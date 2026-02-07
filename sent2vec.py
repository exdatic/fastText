"""
sent2vec -- Pure Python implementation using NumPy and Numba.

A faithful port of the C++ sent2vec / fastText codebase, supporting:
  - Training sent2vec models from a text corpus
  - Loading / saving binary models (compatible with the C++ format)
  - Computing sentence embeddings from trained models

Usage examples:
  # Train a model
  model = Sent2Vec()
  model.train("corpus.txt", "my_model", dim=100, epoch=5)

  # Load a pretrained model and embed sentences
  model = Sent2Vec()
  model.load_model("my_model.bin")
  vec = model.get_sentence_vector("hello world")

Copyright (c) 2016-present, Facebook, Inc.  (original C++ code)
Python port follows the MIT license of the original project.
"""

import struct
import sys
import time
import math
from pathlib import Path

import numpy as np
from numba import njit, types, int32, int64, float32, float64, boolean
from numba.typed import List as NumbaList

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FASTTEXT_VERSION = 12
FASTTEXT_FILEFORMAT_MAGIC_INT32 = 793712314

MAX_VOCAB_SIZE = 30_000_000
MAX_LINE_SIZE = 1024

SIGMOID_TABLE_SIZE = 512
MAX_SIGMOID = 8
LOG_TABLE_SIZE = 512

NEGATIVE_TABLE_SIZE = 10_000_000

# Model types (match C++ enum values)
MODEL_CBOW = 1
MODEL_SG = 2
MODEL_SUP = 3
MODEL_SENT2VEC = 4
MODEL_PVDM = 5

# Loss types
LOSS_HS = 1
LOSS_NS = 2
LOSS_SOFTMAX = 3
LOSS_OVA = 4

# Entry types
ENTRY_WORD = 0
ENTRY_LABEL = 1

# getLine flags
SKIP_EOS = 0x01
SKIP_OOV = 0x02
SKIP_FRQ = 0x04
SKIP_LNG = 0x08

EOS = "</s>"
BOW = "<"
EOW = ">"

# ---------------------------------------------------------------------------
# Numba-accelerated helper functions
# ---------------------------------------------------------------------------

@njit(cache=True)
def _fnv_hash(data):
    """FNV hash matching the C++ fasttext implementation (signed char)."""
    h = np.uint32(2166136261)
    for i in range(len(data)):
        # Cast to int8 then to uint32 to match C++ signed char behaviour
        b = np.int8(data[i])
        h = h ^ np.uint32(b)
        h = np.uint32(h * np.uint32(16777619))
    return h


@njit(cache=True)
def _build_sigmoid_table():
    table = np.empty(SIGMOID_TABLE_SIZE + 1, dtype=np.float32)
    for i in range(SIGMOID_TABLE_SIZE + 1):
        x = float(i) * 2.0 * MAX_SIGMOID / SIGMOID_TABLE_SIZE - MAX_SIGMOID
        table[i] = 1.0 / (1.0 + math.exp(-x))
    return table


@njit(cache=True)
def _build_log_table():
    table = np.empty(LOG_TABLE_SIZE + 1, dtype=np.float32)
    for i in range(LOG_TABLE_SIZE + 1):
        x = (float(i) + 1e-5) / LOG_TABLE_SIZE
        table[i] = math.log(x)
    return table


@njit(cache=True)
def _sigmoid(x, table):
    if x < -MAX_SIGMOID:
        return np.float32(0.0)
    elif x > MAX_SIGMOID:
        return np.float32(1.0)
    else:
        i = int((x + MAX_SIGMOID) * SIGMOID_TABLE_SIZE / MAX_SIGMOID / 2)
        return table[i]


@njit(cache=True)
def _log(x, table):
    if x > 1.0:
        return np.float32(0.0)
    i = int(x * LOG_TABLE_SIZE)
    return table[i]


@njit(cache=True)
def _dot_row(matrix, vec, i, n):
    d = np.float32(0.0)
    offset = i * n
    for j in range(n):
        d += matrix[offset + j] * vec[j]
    return d


@njit(cache=True)
def _add_row_to_vec(matrix, vec, i, n):
    offset = i * n
    for j in range(n):
        vec[j] += matrix[offset + j]


@njit(cache=True)
def _add_row_to_vec_scaled(matrix, vec, i, n, a):
    offset = i * n
    for j in range(n):
        vec[j] += a * matrix[offset + j]


@njit(cache=True)
def _add_vec_to_row(matrix, vec, i, n, a):
    offset = i * n
    for j in range(n):
        matrix[offset + j] += a * vec[j]


@njit(cache=True)
def _compute_hidden(wi, input_ids, hidden, dim):
    """Average the input embedding rows into hidden."""
    n = len(input_ids)
    if n == 0:
        return
    for j in range(dim):
        hidden[j] = np.float32(0.0)
    for k in range(n):
        idx = input_ids[k]
        offset = idx * dim
        for j in range(dim):
            hidden[j] += wi[offset + j]
    inv = np.float32(1.0 / n)
    for j in range(dim):
        hidden[j] *= inv


@njit(cache=True)
def _binary_logistic(wo, hidden, grad, target, label_positive, lr, dim,
                     sigmoid_table, log_table, backprop):
    """Binary logistic loss for a single target. Returns loss."""
    score = _sigmoid(_dot_row(wo, hidden, target, dim), sigmoid_table)
    if backprop:
        alpha = np.float32(lr * (np.float32(label_positive) - score))
        _add_row_to_vec_scaled(wo, grad, target, dim, alpha)
        _add_vec_to_row(wo, hidden, target, dim, alpha)
    if label_positive:
        return -_log(score, log_table)
    else:
        return -_log(np.float32(1.0) - score, log_table)


@njit(cache=True)
def _negative_sampling_forward(wo, wi, hidden, grad, input_ids,
                               target, neg, negatives, neg_size,
                               dim, lr, sigmoid_table, log_table,
                               normalize_gradient, rng_state):
    """Full forward + backward pass for negative sampling loss.

    Returns (loss, rng_state).
    """
    n_input = len(input_ids)
    if n_input == 0:
        return np.float32(0.0), rng_state

    # compute hidden = average of input rows
    _compute_hidden(wi, input_ids, hidden, dim)

    # zero grad
    for j in range(dim):
        grad[j] = np.float32(0.0)

    # positive sample
    loss = _binary_logistic(wo, hidden, grad, target, True, lr, dim,
                            sigmoid_table, log_table, True)

    # negative samples
    for _ in range(neg):
        # LCG matching std::minstd_rand
        rng_state = np.int64((rng_state * 48271) % 2147483647)
        neg_idx = int(np.uint64(rng_state) % np.uint64(neg_size))
        negative = negatives[neg_idx]
        while negative == target:
            rng_state = np.int64((rng_state * 48271) % 2147483647)
            neg_idx = int(np.uint64(rng_state) % np.uint64(neg_size))
            negative = negatives[neg_idx]
        loss += _binary_logistic(wo, hidden, grad, negative, False, lr, dim,
                                 sigmoid_table, log_table, True)

    # normalise gradient if needed
    if normalize_gradient and n_input > 0:
        inv = np.float32(1.0 / n_input)
        for j in range(dim):
            grad[j] *= inv

    # update input embeddings
    for k in range(n_input):
        idx = input_ids[k]
        _add_vec_to_row(wi, grad, idx, dim, np.float32(1.0))

    return loss, rng_state


@njit(cache=True)
def _discard_prob(count, ntokens, t):
    f = float(count) / float(ntokens)
    return math.sqrt(t / f) + t / f


# ---------------------------------------------------------------------------
# Dictionary
# ---------------------------------------------------------------------------

class Dictionary:
    """Vocabulary manager with hashing, subword n-grams, and word n-grams."""

    def __init__(self, args):
        self.args = args
        self.word2int = np.full(MAX_VOCAB_SIZE, -1, dtype=np.int32)
        self.words = []   # list of dicts: {word, count, type, subwords}
        self.pdiscard = np.empty(0, dtype=np.float32)
        self.size = 0
        self.nwords_ = 0
        self.nlabels_ = 0
        self.ntokens_ = 0
        self.pruneidx_size = -1
        self.pruneidx = {}

    # -- hashing --
    @staticmethod
    def hash(word):
        data = word.encode('utf-8')
        h = np.uint32(2166136261)
        with np.errstate(over='ignore'):
            for b in data:
                h = h ^ np.uint32(np.int8(b))
                h = np.uint32(h * np.uint32(16777619))
        return int(h)

    def find(self, word, h=None):
        if h is None:
            h = self.hash(word)
        word2intsize = len(self.word2int)
        idx = int(h % word2intsize)
        while self.word2int[idx] != -1 and self.words[self.word2int[idx]]['word'] != word:
            idx = (idx + 1) % word2intsize
        return idx

    def add(self, word):
        h = self.find(word)
        self.ntokens_ += 1
        if self.word2int[h] == -1:
            entry = {
                'word': word,
                'count': 1,
                'type': ENTRY_LABEL if word.startswith(self.args['label']) else ENTRY_WORD,
                'subwords': [],
            }
            self.words.append(entry)
            self.word2int[h] = self.size
            self.size += 1
        else:
            self.words[self.word2int[h]]['count'] += 1

    def get_id(self, word, h=None):
        idx = self.find(word, h)
        return int(self.word2int[idx])

    def get_type(self, id_or_word):
        if isinstance(id_or_word, str):
            return ENTRY_LABEL if id_or_word.startswith(self.args['label']) else ENTRY_WORD
        return self.words[id_or_word]['type']

    def nwords(self):
        return self.nwords_

    def nlabels(self):
        return self.nlabels_

    def ntokens(self):
        return self.ntokens_

    def get_word(self, wid):
        return self.words[wid]['word']

    def get_token_count(self, wid):
        return self.words[wid]['count']

    # -- discard --
    def discard(self, wid, rand_val):
        if self.args['model'] == MODEL_SUP:
            return False
        return rand_val > self.pdiscard[wid]

    def init_table_discard(self):
        self.pdiscard = np.empty(self.size, dtype=np.float32)
        for i in range(self.size):
            c = self.words[i]['count']
            if c <= 0 or self.ntokens_ <= 0:
                self.pdiscard[i] = 1.0
            else:
                f = float(c) / float(self.ntokens_)
                self.pdiscard[i] = math.sqrt(self.args['t'] / f) + self.args['t'] / f

    # -- subwords --
    def compute_subwords(self, word):
        ngrams = []
        minn = self.args['minn']
        maxn = self.args['maxn']
        if maxn <= 0:
            return ngrams
        word_bytes = word.encode('utf-8')
        # walk over UTF-8 characters
        char_starts = []
        i = 0
        while i < len(word_bytes):
            char_starts.append(i)
            if (word_bytes[i] & 0xC0) == 0x80:
                i += 1
                continue
            i += 1
            while i < len(word_bytes) and (word_bytes[i] & 0xC0) == 0x80:
                i += 1
        char_starts.append(len(word_bytes))  # sentinel

        for ci in range(len(char_starts) - 1):
            if (word_bytes[char_starts[ci]] & 0xC0) == 0x80:
                continue
            for n in range(1, maxn + 1):
                end_ci = ci + n
                if end_ci >= len(char_starts):
                    break
                if n >= minn and not (n == 1 and (ci == 0 or end_ci == len(char_starts) - 1)):
                    ngram = word_bytes[char_starts[ci]:char_starts[end_ci]]
                    h = self.hash(ngram.decode('utf-8', errors='replace')) % self.args['bucket']
                    self.push_hash(ngrams, h)
        return ngrams

    def init_ngrams(self):
        for i in range(self.size):
            word = BOW + self.words[i]['word'] + EOW
            self.words[i]['subwords'] = [i]
            if self.words[i]['word'] != EOS:
                self.words[i]['subwords'].extend(self.compute_subwords(word))

    def get_subwords_by_id(self, wid):
        return self.words[wid]['subwords']

    def get_subwords(self, word):
        wid = self.get_id(word)
        if wid >= 0:
            return list(self.words[wid]['subwords'])
        ngrams = []
        if word != EOS:
            ngrams = self.compute_subwords(BOW + word + EOW)
        return ngrams

    def push_hash(self, hashes, id_val):
        if self.pruneidx_size == 0 or id_val < 0:
            return
        if self.pruneidx_size > 0:
            if id_val in self.pruneidx:
                id_val = self.pruneidx[id_val]
            else:
                return
        hashes.append(self.nwords_ + self.nlabels_ + id_val)

    # -- word n-grams --
    def add_word_ngrams(self, line, hashes, n):
        for i in range(len(hashes)):
            h = np.uint64(hashes[i])
            for j in range(i + 1, min(len(hashes), i + n)):
                h = np.uint64(h * np.uint64(116049371) + np.uint64(hashes[j]))
                bucket_id = int(h % np.uint64(self.args['bucket']))
                self.push_hash(line, bucket_id)

    def add_word_ngrams_dropout(self, line, hashes, n, k, rng):
        size = len(hashes)
        if size <= 2:
            return
        discard = [False] * size
        num_discarded = 0
        while num_discarded < k and size - num_discarded > 2:
            token_to_discard = rng.randint(1, size - 1)
            if not discard[token_to_discard]:
                discard[token_to_discard] = True
                num_discarded += 1
        for i in range(size):
            if discard[i]:
                continue
            h = np.uint64(hashes[i])
            for j in range(i + 1, min(size, i + n)):
                if discard[j]:
                    break
                h = np.uint64(h * np.uint64(116049371) + np.uint64(hashes[j]))
                bucket_id = int(h % np.uint64(self.args['bucket']))
                self.push_hash(line, bucket_id)

    # -- reading --
    @staticmethod
    def read_words(text):
        """Yield words from text line-by-line, inserting EOS at newlines."""
        for line in text.split('\n'):
            tokens = line.split()
            for t in tokens:
                yield t
            yield EOS

    def read_from_file(self, filepath):
        min_threshold = 1
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            text = f.read()
        for word in self.read_words(text):
            self.add(word)
            if self.ntokens_ % 1_000_000 == 0 and self.args.get('verbose', 2) > 1:
                sys.stderr.write(f"\rRead {self.ntokens_ // 1_000_000}M words")
                sys.stderr.flush()
            if self.size > 0.75 * MAX_VOCAB_SIZE:
                min_threshold += 1
                self.threshold(min_threshold, min_threshold)

        # Add placeholder for sent2vec
        if self.args['model'] == MODEL_SENT2VEC:
            h = self.find("<PLACEHOLDER>")
            entry = {
                'word': '<PLACEHOLDER>',
                'count': int(1e18),
                'type': ENTRY_WORD,
                'subwords': [],
            }
            self.words.append(entry)
            self.word2int[h] = self.size
            self.size += 1

        self.threshold(self.args['minCount'], self.args.get('minCountLabel', 0))
        self.init_table_discard()
        self.init_ngrams()

        # Zero-out placeholder count so it doesn't affect discard
        if self.args['model'] == MODEL_SENT2VEC:
            assert self.words[0]['word'] == '<PLACEHOLDER>'
            self.words[0]['count'] = 0

        if self.args.get('verbose', 2) > 0:
            sys.stderr.write(f"\rRead {self.ntokens_ // 1_000_000}M words\n")
            sys.stderr.write(f"Number of words:  {self.nwords_}\n")
            sys.stderr.write(f"Number of labels: {self.nlabels_}\n")
            sys.stderr.flush()

        if self.size == 0:
            raise ValueError("Empty vocabulary. Try a smaller -minCount value.")

    def threshold(self, t, tl):
        # Sort: words first (by count desc), then labels (by count desc)
        self.words.sort(key=lambda e: (e['type'], -e['count']))
        self.words = [e for e in self.words
                      if not ((e['type'] == ENTRY_WORD and e['count'] < t) or
                              (e['type'] == ENTRY_LABEL and e['count'] < tl))]
        self.size = 0
        self.nwords_ = 0
        self.nlabels_ = 0
        self.word2int = np.full(MAX_VOCAB_SIZE, -1, dtype=np.int32)
        for entry in self.words:
            h = self.find(entry['word'])
            self.word2int[h] = self.size
            self.size += 1
            if entry['type'] == ENTRY_WORD:
                self.nwords_ += 1
            else:
                self.nlabels_ += 1

    def get_counts(self, entry_type):
        return [w['count'] for w in self.words if w['type'] == entry_type]

    # -- getLine variants --
    def get_line_sent2vec(self, tokens, rng, flags=0):
        """Parse a sentence for sent2vec training.

        Returns (word_ids, word_hashes, labels).
        """
        words = []
        word_hashes = []
        labels = []
        ntokens = 0
        for token in tokens:
            if flags & SKIP_EOS:
                if token == EOS:
                    break
            h = self.hash(token)
            wid = self.get_id(token, h)
            if flags & SKIP_OOV:
                if wid < 0:
                    continue
            etype = self.get_type(token) if wid < 0 else self.get_type(wid)
            ntokens += 1
            if etype == ENTRY_WORD:
                if flags & SKIP_FRQ:
                    if self.discard(wid, rng.random()):
                        continue
                words.append(wid)
                word_hashes.append(h)
            elif etype == ENTRY_LABEL and wid >= 0:
                labels.append(wid)
            if flags & SKIP_LNG:
                if ntokens > MAX_LINE_SIZE:
                    break
            if token == EOS:
                break
        return words, word_hashes, labels, ntokens

    # -- binary I/O --
    def save(self, f):
        f.write(struct.pack('<i', self.size))
        f.write(struct.pack('<i', self.nwords_))
        f.write(struct.pack('<i', self.nlabels_))
        f.write(struct.pack('<q', self.ntokens_))
        f.write(struct.pack('<q', self.pruneidx_size))
        for i in range(self.size):
            e = self.words[i]
            f.write(e['word'].encode('utf-8'))
            f.write(b'\x00')
            f.write(struct.pack('<q', e['count']))
            f.write(struct.pack('<b', e['type']))
        for first, second in self.pruneidx.items():
            f.write(struct.pack('<i', first))
            f.write(struct.pack('<i', second))

    def load(self, f):
        self.words = []
        self.size = struct.unpack('<i', f.read(4))[0]
        self.nwords_ = struct.unpack('<i', f.read(4))[0]
        self.nlabels_ = struct.unpack('<i', f.read(4))[0]
        self.ntokens_ = struct.unpack('<q', f.read(8))[0]
        self.pruneidx_size = struct.unpack('<q', f.read(8))[0]
        for _ in range(self.size):
            word_bytes = bytearray()
            while True:
                c = f.read(1)
                if c == b'\x00' or c == b'':
                    break
                word_bytes.extend(c)
            word = word_bytes.decode('utf-8', errors='replace')
            count = struct.unpack('<q', f.read(8))[0]
            etype = struct.unpack('<b', f.read(1))[0]
            self.words.append({
                'word': word,
                'count': count,
                'type': etype,
                'subwords': [],
            })
        self.pruneidx = {}
        for _ in range(max(0, self.pruneidx_size)):
            first = struct.unpack('<i', f.read(4))[0]
            second = struct.unpack('<i', f.read(4))[0]
            self.pruneidx[first] = second
        self.init_table_discard()
        self.init_ngrams()
        word2intsize = max(1, math.ceil(self.size / 0.7))
        self.word2int = np.full(word2intsize, -1, dtype=np.int32)
        for i in range(self.size):
            self.word2int[self.find(self.words[i]['word'])] = i


# ---------------------------------------------------------------------------
# Negative sampling table builder
# ---------------------------------------------------------------------------

def build_negative_table(target_counts):
    """Build the unigram^0.5 negative sampling table."""
    z = sum(c ** 0.5 for c in target_counts)
    negatives = []
    for i, c in enumerate(target_counts):
        n_entries = int(c ** 0.5 * NEGATIVE_TABLE_SIZE / z)
        negatives.extend([i] * n_entries)
    if len(negatives) == 0:
        negatives.append(0)
    return np.array(negatives, dtype=np.int32)


# ---------------------------------------------------------------------------
# Sent2Vec model
# ---------------------------------------------------------------------------

class Sent2Vec:
    """Pure Python sent2vec implementation."""

    def __init__(self):
        self.args = None
        self.dict = None
        self.wi = None        # input embeddings  (flat float32 array)
        self.wo = None        # output embeddings (flat float32 array)
        self.quant = False
        self.version = FASTTEXT_VERSION
        self._sigmoid_table = _build_sigmoid_table()
        self._log_table = _build_log_table()

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------

    def train(self, input_path, output_path, **kwargs):
        """Train a sent2vec model.

        Args:
            input_path: Path to training corpus (one sentence per line).
            output_path: Base path for saving model files.
            **kwargs: Hyperparameters (dim, epoch, lr, neg, minCount,
                      wordNgrams, dropoutK, bucket, t, minn, maxn,
                      thread, verbose, seed, minCountLabel).
        """
        self.args = {
            'model': MODEL_SENT2VEC,
            'loss': LOSS_NS,
            'dim': 100,
            'epoch': 5,
            'lr': 0.2,
            'neg': 10,
            'minCount': 5,
            'minCountLabel': 0,
            'wordNgrams': 1,
            'dropoutK': 2,
            'bucket': 2000000,
            'minn': 0,
            'maxn': 0,
            'ws': 5,
            't': 1e-4,
            'lrUpdateRate': 100,
            'label': '__label__',
            'verbose': 2,
            'seed': 0,
            'thread': 1,   # Python uses a single thread
            'input': input_path,
            'output': output_path,
        }
        self.args.update(kwargs)

        # Force model type
        self.args['model'] = MODEL_SENT2VEC
        self.args['loss'] = LOSS_NS

        # If wordNgrams <= 1 and maxn == 0, no bucket needed
        if self.args['wordNgrams'] <= 1 and self.args['maxn'] == 0:
            self.args['bucket'] = 0

        # Build dictionary
        self.dict = Dictionary(self.args)
        self.dict.read_from_file(input_path)

        dim = self.args['dim']
        n_input = self.dict.nwords() + self.dict.nlabels() + self.args['bucket']
        n_output = self.dict.nwords() + self.dict.nlabels()

        # Initialize matrices
        bound = 1.0 / dim
        rng_init = np.random.RandomState(self.args['seed'])
        self.wi = rng_init.uniform(-bound, bound, n_input * dim).astype(np.float32)
        self.wo = np.zeros(n_output * dim, dtype=np.float32)

        # Build negative sampling table
        target_counts = self.dict.get_counts(ENTRY_WORD)
        negatives = build_negative_table(target_counts)

        # Training loop
        self._train_loop(negatives)

        # Save
        self.save_model(output_path + '.bin')
        self.save_vectors(output_path + '.vec')

    def _train_loop(self, negatives):
        """Main training loop (single-threaded)."""
        args = self.args
        dim = args['dim']
        ntokens = self.dict.ntokens()
        total_tokens = args['epoch'] * ntokens
        neg = args['neg']
        neg_size = len(negatives)
        normalize_gradient = True  # sent2vec uses normalizeGradient

        # State
        hidden = np.zeros(dim, dtype=np.float32)
        grad = np.zeros(dim, dtype=np.float32)
        rng = np.random.RandomState(args['seed'])
        rng_state = np.int64(args['seed'] + 1)  # for numba LCG

        token_count = 0
        loss_sum = 0.0
        n_examples = 0
        start_time = time.time()

        # Read corpus into sentences
        sentences = []
        with open(args['input'], 'r', encoding='utf-8', errors='replace') as f:
            for raw_line in f:
                tokens = raw_line.strip().split()
                if tokens:
                    sentences.append(tokens)

        if not sentences:
            raise ValueError("No sentences found in input file")

        n_sentences = len(sentences)

        for epoch in range(args['epoch']):
            for si in range(n_sentences):
                tokens = sentences[si]
                # Parse the sentence
                line, hashes, labels, nt = self.dict.get_line_sent2vec(
                    tokens, rng,
                    flags=SKIP_EOS | SKIP_LNG | SKIP_OOV)
                token_count += nt

                if len(line) <= 1:
                    continue

                # Progress and learning rate
                progress = float(token_count) / float(total_tokens)
                lr = args['lr'] * (1.0 - progress)
                if lr <= 0:
                    break

                # sent2vec training step
                loss_val, rng_state = self._sent2vec_step(
                    line, hashes, lr, neg, negatives, neg_size,
                    hidden, grad, dim, rng, rng_state, normalize_gradient)
                loss_sum += loss_val[0]
                n_examples += loss_val[1]

                # Print progress
                if args.get('verbose', 2) > 1 and si % 1000 == 0:
                    elapsed = time.time() - start_time
                    wps = token_count / max(elapsed, 1e-6)
                    avg_loss = loss_sum / max(n_examples, 1)
                    sys.stderr.write(
                        f"\rProgress: {progress * 100:.1f}%"
                        f"  words/sec: {wps:.0f}"
                        f"  lr: {lr:.6f}"
                        f"  avg.loss: {avg_loss:.6f}")
                    sys.stderr.flush()

            if lr <= 0:
                break

        if args.get('verbose', 2) > 0:
            avg_loss = loss_sum / max(n_examples, 1)
            sys.stderr.write(
                f"\rProgress: 100.0%  avg.loss: {avg_loss:.6f}\n")
            sys.stderr.flush()

    def _sent2vec_step(self, line, hashes, lr, neg, negatives, neg_size,
                       hidden, grad, dim, rng, rng_state, normalize_gradient):
        """One sent2vec training step over a sentence.

        For each word in the sentence, create a context (sentence minus
        that word), compute word n-grams with optional dropout, then do
        a negative-sampling update.
        """
        total_loss = 0.0
        total_examples = 0

        for w in range(len(line)):
            # Discard decision
            wid = line[w]
            if self.dict.discard(wid, rng.random()):
                continue
            if self.dict.get_token_count(wid) < self.args.get('minCountLabel', 0):
                continue

            # Build context: replace target word with PLACEHOLDER (index 0)
            bow = list(line)
            boh = list(hashes)
            bow[w] = 0
            boh[w] = 0

            # Add word n-grams
            if self.args['dropoutK'] > 0:
                self.dict.add_word_ngrams_dropout(
                    bow, boh, self.args['wordNgrams'],
                    self.args['dropoutK'], rng)
            else:
                self.dict.add_word_ngrams(bow, boh, self.args['wordNgrams'])

            # Convert to numpy for numba
            input_ids = np.array(bow, dtype=np.int32)
            target = np.int32(line[w])

            loss, rng_state = _negative_sampling_forward(
                self.wo, self.wi, hidden, grad, input_ids,
                target, neg, negatives, neg_size,
                dim, np.float32(lr),
                self._sigmoid_table, self._log_table,
                normalize_gradient, rng_state)

            total_loss += float(loss)
            total_examples += 1

        return (total_loss, total_examples), rng_state

    # -----------------------------------------------------------------------
    # Inference
    # -----------------------------------------------------------------------

    def get_word_vector(self, word):
        """Get the embedding for a single word."""
        dim = self.args['dim']
        ngrams = self.dict.get_subwords(word)
        vec = np.zeros(dim, dtype=np.float32)
        for idx in ngrams:
            _add_row_to_vec(self.wi, vec, idx, dim)
        if len(ngrams) > 0:
            vec *= 1.0 / len(ngrams)
        return vec

    def get_sentence_vector(self, sentence):
        """Compute the sent2vec embedding for a sentence string.

        Each word vector is L2-normalised before averaging.
        """
        dim = self.args['dim']
        svec = np.zeros(dim, dtype=np.float32)
        words = sentence.strip().split()
        count = 0
        for word in words:
            vec = self.get_word_vector(word)
            norm = np.sqrt(np.sum(vec * vec))
            if norm > 0:
                vec *= 1.0 / norm
                svec += vec
                count += 1
        if count > 0:
            svec *= 1.0 / count
        return svec

    def get_sentence_vectors(self, sentences):
        """Batch compute sentence embeddings.

        Args:
            sentences: list of sentence strings.

        Returns:
            numpy array of shape (len(sentences), dim).
        """
        dim = self.args['dim']
        result = np.zeros((len(sentences), dim), dtype=np.float32)
        for i, sent in enumerate(sentences):
            result[i] = self.get_sentence_vector(sent)
        return result

    # -----------------------------------------------------------------------
    # Model I/O  (compatible with C++ binary format)
    # -----------------------------------------------------------------------

    def save_model(self, filepath):
        """Save model in the C++ fasttext binary format."""
        with open(filepath, 'wb') as f:
            # Magic + version
            f.write(struct.pack('<i', FASTTEXT_FILEFORMAT_MAGIC_INT32))
            f.write(struct.pack('<i', FASTTEXT_VERSION))
            # Args
            self._save_args(f)
            # Dictionary
            self.dict.save(f)
            # quant flag for input
            f.write(struct.pack('<?', False))
            # Input matrix
            n_input = len(self.wi) // self.args['dim']
            dim = self.args['dim']
            f.write(struct.pack('<q', n_input))
            f.write(struct.pack('<q', dim))
            f.write(self.wi.tobytes())
            # qout flag
            f.write(struct.pack('<?', False))
            # Output matrix
            n_output = len(self.wo) // dim
            f.write(struct.pack('<q', n_output))
            f.write(struct.pack('<q', dim))
            f.write(self.wo.tobytes())

    def load_model(self, filepath):
        """Load a model from the C++ fasttext binary format."""
        with open(filepath, 'rb') as f:
            magic = struct.unpack('<i', f.read(4))[0]
            if magic != FASTTEXT_FILEFORMAT_MAGIC_INT32:
                raise ValueError("Invalid model file format (bad magic number)")
            self.version = struct.unpack('<i', f.read(4))[0]
            if self.version > FASTTEXT_VERSION:
                raise ValueError(f"Model version {self.version} > supported {FASTTEXT_VERSION}")

            # Load args
            self.args = self._load_args(f)

            # Load dictionary
            self.dict = Dictionary(self.args)
            self.dict.load(f)

            # Input matrix
            quant_input = struct.unpack('<?', f.read(1))[0]
            if quant_input:
                raise ValueError("Quantised models are not supported in the pure Python version")
            m_in = struct.unpack('<q', f.read(8))[0]
            n_in = struct.unpack('<q', f.read(8))[0]
            self.wi = np.frombuffer(f.read(m_in * n_in * 4), dtype=np.float32).copy()

            # Output matrix
            qout = struct.unpack('<?', f.read(1))[0]
            if qout:
                raise ValueError("Quantised output not supported in the pure Python version")
            m_out = struct.unpack('<q', f.read(8))[0]
            n_out = struct.unpack('<q', f.read(8))[0]
            self.wo = np.frombuffer(f.read(m_out * n_out * 4), dtype=np.float32).copy()

    def save_vectors(self, filepath):
        """Save word vectors in text format (word2vec style)."""
        dim = self.args['dim']
        nwords = self.dict.nwords()
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"{nwords} {dim}\n")
            for i in range(nwords):
                word = self.dict.get_word(i)
                vec = self.get_word_vector(word)
                vec_str = ' '.join(f'{v:.5f}' for v in vec)
                f.write(f"{word} {vec_str}\n")

    def _save_args(self, f):
        """Write args to binary stream in C++ format."""
        a = self.args
        f.write(struct.pack('<i', a['dim']))
        f.write(struct.pack('<i', a['ws']))
        f.write(struct.pack('<i', a['epoch']))
        f.write(struct.pack('<i', a['minCount']))
        f.write(struct.pack('<i', a['neg']))
        f.write(struct.pack('<i', a['wordNgrams']))
        f.write(struct.pack('<i', a['loss']))
        f.write(struct.pack('<i', a['model']))
        f.write(struct.pack('<i', a['bucket']))
        f.write(struct.pack('<i', a['minn']))
        f.write(struct.pack('<i', a['maxn']))
        f.write(struct.pack('<i', a['lrUpdateRate']))
        f.write(struct.pack('<d', a['t']))

    def _load_args(self, f):
        """Read args from binary stream in C++ format."""
        a = {}
        a['dim'] = struct.unpack('<i', f.read(4))[0]
        a['ws'] = struct.unpack('<i', f.read(4))[0]
        a['epoch'] = struct.unpack('<i', f.read(4))[0]
        a['minCount'] = struct.unpack('<i', f.read(4))[0]
        a['neg'] = struct.unpack('<i', f.read(4))[0]
        a['wordNgrams'] = struct.unpack('<i', f.read(4))[0]
        a['loss'] = struct.unpack('<i', f.read(4))[0]
        a['model'] = struct.unpack('<i', f.read(4))[0]
        a['bucket'] = struct.unpack('<i', f.read(4))[0]
        a['minn'] = struct.unpack('<i', f.read(4))[0]
        a['maxn'] = struct.unpack('<i', f.read(4))[0]
        a['lrUpdateRate'] = struct.unpack('<i', f.read(4))[0]
        a['t'] = struct.unpack('<d', f.read(8))[0]
        # Fill in defaults for fields not in the binary format
        a['label'] = '__label__'
        a['verbose'] = 2
        a['lr'] = 0.2
        a['dropoutK'] = 2
        a['minCountLabel'] = 0
        a['seed'] = 0
        a['thread'] = 1
        return a

    # -----------------------------------------------------------------------
    # Dimension accessor
    # -----------------------------------------------------------------------

    @property
    def dim(self):
        return self.args['dim'] if self.args else 0


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _print_usage():
    sys.stderr.write(
        "usage: python sent2vec.py <command> [<args>]\n\n"
        "  sent2vec     Train a sent2vec model\n"
        "  print-vec    Print sentence vectors from a trained model\n\n"
        "Training arguments:\n"
        "  -input        training file path (required)\n"
        "  -output       output file path (required)\n"
        "  -dim          size of word vectors [100]\n"
        "  -epoch        number of epochs [5]\n"
        "  -lr           learning rate [0.2]\n"
        "  -neg          number of negatives sampled [10]\n"
        "  -minCount     minimal number of word occurrences [5]\n"
        "  -wordNgrams   max length of word ngram [1]\n"
        "  -dropoutK     number of ngrams dropped when forming n-gram features [2]\n"
        "  -bucket       number of buckets [2000000]\n"
        "  -t            sampling threshold [1e-4]\n"
        "  -verbose      verbosity level [2]\n"
        "  -seed         random seed [0]\n"
    )


def main():
    if len(sys.argv) < 2:
        _print_usage()
        sys.exit(1)

    command = sys.argv[1]

    if command == 'sent2vec':
        # Parse args
        kwargs = {}
        i = 2
        input_path = None
        output_path = None
        while i < len(sys.argv):
            arg = sys.argv[i]
            if arg == '-input':
                input_path = sys.argv[i + 1]; i += 2
            elif arg == '-output':
                output_path = sys.argv[i + 1]; i += 2
            elif arg == '-dim':
                kwargs['dim'] = int(sys.argv[i + 1]); i += 2
            elif arg == '-epoch':
                kwargs['epoch'] = int(sys.argv[i + 1]); i += 2
            elif arg == '-lr':
                kwargs['lr'] = float(sys.argv[i + 1]); i += 2
            elif arg == '-neg':
                kwargs['neg'] = int(sys.argv[i + 1]); i += 2
            elif arg == '-minCount':
                kwargs['minCount'] = int(sys.argv[i + 1]); i += 2
            elif arg == '-wordNgrams':
                kwargs['wordNgrams'] = int(sys.argv[i + 1]); i += 2
            elif arg == '-dropoutK':
                kwargs['dropoutK'] = int(sys.argv[i + 1]); i += 2
            elif arg == '-bucket':
                kwargs['bucket'] = int(sys.argv[i + 1]); i += 2
            elif arg == '-t':
                kwargs['t'] = float(sys.argv[i + 1]); i += 2
            elif arg == '-verbose':
                kwargs['verbose'] = int(sys.argv[i + 1]); i += 2
            elif arg == '-seed':
                kwargs['seed'] = int(sys.argv[i + 1]); i += 2
            elif arg == '-minn':
                kwargs['minn'] = int(sys.argv[i + 1]); i += 2
            elif arg == '-maxn':
                kwargs['maxn'] = int(sys.argv[i + 1]); i += 2
            else:
                sys.stderr.write(f"Unknown argument: {arg}\n")
                _print_usage()
                sys.exit(1)

        if not input_path or not output_path:
            sys.stderr.write("Error: -input and -output are required.\n")
            _print_usage()
            sys.exit(1)

        model = Sent2Vec()
        model.train(input_path, output_path, **kwargs)

    elif command == 'print-vec':
        if len(sys.argv) < 3:
            sys.stderr.write("usage: python sent2vec.py print-vec <model.bin>\n")
            sys.exit(1)
        model_path = sys.argv[2]
        model = Sent2Vec()
        model.load_model(model_path)
        dim = model.dim
        sys.stderr.write(f"Loaded model: {model.dict.nwords()} words, dim={dim}\n")
        sys.stderr.write("Enter sentences (one per line, Ctrl-D to stop):\n")
        for line in sys.stdin:
            vec = model.get_sentence_vector(line)
            print(' '.join(f'{v:.5f}' for v in vec))
    else:
        sys.stderr.write(f"Unknown command: {command}\n")
        _print_usage()
        sys.exit(1)


if __name__ == '__main__':
    main()
