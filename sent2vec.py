"""
sent2vec -- Pure Python implementation using NumPy and Numba.

Supports training sent2vec models, loading/saving binary models (compatible
with the C++ fasttext format), and computing sentence embeddings.

    >>> model = Sent2Vec.train("corpus.txt", dim=100, epoch=5)
    >>> model.save("my_model.bin")

    >>> model = Sent2Vec.load("my_model.bin")
    >>> vec = model.embed("hello world")
    >>> vecs = model.embed(["sentence one", "sentence two"])

Copyright (c) 2016-present, Facebook, Inc.  (original C++ code)
Python port follows the MIT license of the original project.
"""

from __future__ import annotations

import argparse
import math
import struct
import sys
import time
from dataclasses import dataclass, field

import numpy as np
from numba import njit

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAGIC = 793712314
_VERSION = 12
_MAX_VOCAB = 30_000_000
_MAX_LINE = 1024
_SIG_SIZE = 512
_MAX_SIG = 8
_LOG_SIZE = 512
_NEG_TABLE = 10_000_000
_WORD = 0
_LABEL = 1
_SENT2VEC = 4
_NS = 2

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Entry:
    word: str
    count: int
    type: int          # _WORD or _LABEL
    subwords: list[int] = field(default_factory=list)


@dataclass
class Args:
    """Model hyperparameters. Fields serialised to the binary format are
    marked with ``# [bin]``."""

    dim: int = 100          # [bin]
    ws: int = 5             # [bin]
    epoch: int = 5          # [bin]
    min_count: int = 5      # [bin]
    neg: int = 10           # [bin]
    word_ngrams: int = 1    # [bin]
    loss: int = _NS         # [bin]
    model: int = _SENT2VEC  # [bin]
    bucket: int = 2_000_000 # [bin]
    minn: int = 0           # [bin]
    maxn: int = 0           # [bin]
    lr_update_rate: int = 100  # [bin]
    t: float = 1e-4         # [bin]

    # Not in binary format -- runtime only
    lr: float = 0.2
    dropout_k: int = 2
    min_count_label: int = 0
    label_prefix: str = "__label__"
    verbose: int = 2
    seed: int = 0

    # -- binary I/O (12 ints + 1 double = 56 bytes) --
    _BIN_FMT = "<12id"

    def to_bytes(self) -> bytes:
        return struct.pack(
            self._BIN_FMT,
            self.dim, self.ws, self.epoch, self.min_count, self.neg,
            self.word_ngrams, self.loss, self.model, self.bucket,
            self.minn, self.maxn, self.lr_update_rate, self.t,
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> "Args":
        vals = struct.unpack(cls._BIN_FMT, data)
        return cls(
            dim=vals[0], ws=vals[1], epoch=vals[2], min_count=vals[3],
            neg=vals[4], word_ngrams=vals[5], loss=vals[6], model=vals[7],
            bucket=vals[8], minn=vals[9], maxn=vals[10],
            lr_update_rate=vals[11], t=vals[12],
        )

    BIN_SIZE = struct.calcsize(_BIN_FMT)  # 56


# ---------------------------------------------------------------------------
# FNV-1a hash (matching the C++ fasttext signed-char variant)
# ---------------------------------------------------------------------------

def _fnv(word: str) -> int:
    """FNV hash compatible with the C++ fasttext implementation."""
    h = 2166136261
    for b in word.encode("utf-8"):
        # Replicate C++ ``h ^ uint32_t(int8_t(c))``
        sb = b if b < 128 else b - 256          # signed byte
        h = (h ^ (sb & 0xFFFFFFFF)) & 0xFFFFFFFF
        h = (h * 16777619) & 0xFFFFFFFF
    return h


# ---------------------------------------------------------------------------
# Numba kernels
# ---------------------------------------------------------------------------

@njit(cache=True)
def _build_sigmoid_table():
    t = np.empty(_SIG_SIZE + 1, dtype=np.float32)
    for i in range(_SIG_SIZE + 1):
        x = float(i) * 2.0 * _MAX_SIG / _SIG_SIZE - _MAX_SIG
        t[i] = 1.0 / (1.0 + math.exp(-x))
    return t


@njit(cache=True)
def _build_log_table():
    t = np.empty(_LOG_SIZE + 1, dtype=np.float32)
    for i in range(_LOG_SIZE + 1):
        x = (float(i) + 1e-5) / _LOG_SIZE
        t[i] = math.log(x)
    return t


_SIGMOID_TABLE = _build_sigmoid_table()
_LOG_TABLE = _build_log_table()


@njit(cache=True)
def _sigmoid(x):
    if x < -_MAX_SIG:
        return np.float32(0.0)
    if x > _MAX_SIG:
        return np.float32(1.0)
    return _SIGMOID_TABLE[int((x + _MAX_SIG) * _SIG_SIZE / _MAX_SIG / 2)]


@njit(cache=True)
def _log(x):
    if x > 1.0:
        return np.float32(0.0)
    return _LOG_TABLE[int(x * _LOG_SIZE)]


@njit(cache=True)
def _dot(a, b):
    s = np.float32(0.0)
    for i in range(len(a)):
        s += a[i] * b[i]
    return s


@njit(cache=True)
def _binary_logistic(wo, hidden, grad, target, positive, lr, dim):
    score = _sigmoid(_dot(wo[target], hidden))
    alpha = np.float32(lr * (np.float32(positive) - score))
    grad += alpha * wo[target]
    wo[target] += alpha * hidden
    if positive:
        return -_log(score)
    return -_log(np.float32(1.0) - score)


@njit(cache=True)
def _ns_update(wi, wo, input_ids, target, neg, negatives,
               dim, lr, normalize_gradient, rng_state):
    """Negative-sampling forward + backward for one training example.

    Args:
        wi: input embeddings  (n_input, dim)
        wo: output embeddings (n_output, dim)
        input_ids: int32 array of context token indices
        target: positive target word index
        neg: number of negative samples
        negatives: int32 negative-sampling table
        dim: embedding dimension
        lr: current learning rate
        normalize_gradient: whether to normalise gradient by input size
        rng_state: minstd_rand state (int64)

    Returns:
        (loss, rng_state)
    """
    n = len(input_ids)
    if n == 0:
        return np.float32(0.0), rng_state

    # hidden = mean of input rows
    hidden = np.zeros(dim, dtype=np.float32)
    for k in range(n):
        hidden += wi[input_ids[k]]
    hidden *= np.float32(1.0 / n)

    grad = np.zeros(dim, dtype=np.float32)

    # positive
    loss = _binary_logistic(wo, hidden, grad, target, True, lr, dim)

    # negatives
    neg_size = len(negatives)
    for _ in range(neg):
        rng_state = np.int64((rng_state * 48271) % 2147483647)
        negative = negatives[int(np.uint64(rng_state) % np.uint64(neg_size))]
        while negative == target:
            rng_state = np.int64((rng_state * 48271) % 2147483647)
            negative = negatives[int(np.uint64(rng_state) % np.uint64(neg_size))]
        loss += _binary_logistic(wo, hidden, grad, negative, False, lr, dim)

    # normalise and propagate to input
    if normalize_gradient:
        grad *= np.float32(1.0 / n)
    for k in range(n):
        wi[input_ids[k]] += grad

    return loss, rng_state


# ---------------------------------------------------------------------------
# Dictionary
# ---------------------------------------------------------------------------

class Dictionary:
    """Vocabulary with hashing, subword n-grams, and word n-grams."""

    def __init__(self, args: Args):
        self.args = args
        self._w2i: np.ndarray = np.full(_MAX_VOCAB, -1, dtype=np.int32)
        self.entries: list[Entry] = []
        self.pdiscard: np.ndarray = np.empty(0, dtype=np.float32)
        self.nwords: int = 0
        self.nlabels: int = 0
        self.ntokens: int = 0
        self._pruneidx: dict[int, int] = {}
        self._pruneidx_size: int = -1

    # -- hash / lookup -------------------------------------------------------

    def _slot(self, word: str, h: int | None = None) -> int:
        if h is None:
            h = _fnv(word)
        sz = len(self._w2i)
        idx = h % sz
        while self._w2i[idx] != -1 and self.entries[self._w2i[idx]].word != word:
            idx = (idx + 1) % sz
        return idx

    def get_id(self, word: str, h: int | None = None) -> int:
        return int(self._w2i[self._slot(word, h)])

    # -- build ---------------------------------------------------------------

    def _add(self, word: str):
        slot = self._slot(word)
        self.ntokens += 1
        if self._w2i[slot] == -1:
            etype = _LABEL if word.startswith(self.args.label_prefix) else _WORD
            self.entries.append(Entry(word, 1, etype))
            self._w2i[slot] = len(self.entries) - 1
        else:
            self.entries[self._w2i[slot]].count += 1

    def read_from_file(self, path: str):
        min_thr = 1
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for raw in fh:
                for tok in raw.split():
                    self._add(tok)
                    if self.ntokens % 1_000_000 == 0 and self.args.verbose > 1:
                        sys.stderr.write(f"\rRead {self.ntokens // 1_000_000}M words")
                        sys.stderr.flush()
                    if len(self.entries) > 0.75 * _MAX_VOCAB:
                        min_thr += 1
                        self._threshold(min_thr, min_thr)
                self._add("</s>")

        # sent2vec placeholder
        if self.args.model == _SENT2VEC:
            slot = self._slot("<PLACEHOLDER>")
            self.entries.append(Entry("<PLACEHOLDER>", int(1e18), _WORD))
            self._w2i[slot] = len(self.entries) - 1

        self._threshold(self.args.min_count, self.args.min_count_label)
        self._init_discard()
        self._init_ngrams()

        if self.args.model == _SENT2VEC:
            assert self.entries[0].word == "<PLACEHOLDER>"
            self.entries[0].count = 0

        if self.args.verbose > 0:
            sys.stderr.write(
                f"\rRead {self.ntokens // 1_000_000}M words\n"
                f"Number of words:  {self.nwords}\n"
                f"Number of labels: {self.nlabels}\n")
            sys.stderr.flush()

    def _threshold(self, tw: int, tl: int):
        self.entries.sort(key=lambda e: (e.type, -e.count))
        self.entries = [e for e in self.entries
                        if not ((e.type == _WORD and e.count < tw)
                                or (e.type == _LABEL and e.count < tl))]
        self.nwords = sum(1 for e in self.entries if e.type == _WORD)
        self.nlabels = sum(1 for e in self.entries if e.type == _LABEL)
        self._w2i = np.full(_MAX_VOCAB, -1, dtype=np.int32)
        for i, e in enumerate(self.entries):
            self._w2i[self._slot(e.word)] = i

    def _init_discard(self):
        n = len(self.entries)
        self.pdiscard = np.ones(n, dtype=np.float32)
        for i, e in enumerate(self.entries):
            if e.count > 0 and self.ntokens > 0:
                f = e.count / self.ntokens
                self.pdiscard[i] = math.sqrt(self.args.t / f) + self.args.t / f

    def _init_ngrams(self):
        for i, e in enumerate(self.entries):
            e.subwords = [i]
            if e.word != "</s>":
                e.subwords.extend(self._compute_subwords("<" + e.word + ">"))

    def _compute_subwords(self, word: str) -> list[int]:
        minn, maxn = self.args.minn, self.args.maxn
        if maxn <= 0:
            return []
        raw = word.encode("utf-8")
        starts = [i for i in range(len(raw)) if i == 0 or (raw[i] & 0xC0) != 0x80]
        starts.append(len(raw))
        ngrams: list[int] = []
        for ci in range(len(starts) - 1):
            for n in range(1, maxn + 1):
                end = ci + n
                if end >= len(starts):
                    break
                if n >= minn and not (n == 1 and (ci == 0 or end == len(starts) - 1)):
                    sub = raw[starts[ci]:starts[end]].decode("utf-8", "replace")
                    h = _fnv(sub) % self.args.bucket
                    self._push_hash(ngrams, h)
        return ngrams

    def _push_hash(self, out: list[int], h: int):
        if self._pruneidx_size == 0 or h < 0:
            return
        if self._pruneidx_size > 0:
            if h in self._pruneidx:
                h = self._pruneidx[h]
            else:
                return
        out.append(self.nwords + self.nlabels + h)

    # -- subwords / word n-grams ---------------------------------------------

    def get_subwords(self, word: str) -> list[int]:
        wid = self.get_id(word)
        if wid >= 0:
            return list(self.entries[wid].subwords)
        if word != "</s>":
            return self._compute_subwords("<" + word + ">")
        return []

    def add_word_ngrams(self, line: list[int], hashes: list[int], n: int):
        for i in range(len(hashes)):
            h = hashes[i] & 0xFFFFFFFFFFFFFFFF
            for j in range(i + 1, min(len(hashes), i + n)):
                h = ((h * 116049371) + hashes[j]) & 0xFFFFFFFFFFFFFFFF
                self._push_hash(line, h % self.args.bucket)

    def add_word_ngrams_dropout(self, line: list[int], hashes: list[int],
                                n: int, k: int, rng: np.random.RandomState):
        sz = len(hashes)
        if sz <= 2:
            return
        drop = [False] * sz
        nd = 0
        while nd < k and sz - nd > 2:
            t = rng.randint(1, sz - 1)
            if not drop[t]:
                drop[t] = True
                nd += 1
        for i in range(sz):
            if drop[i]:
                continue
            h = hashes[i] & 0xFFFFFFFFFFFFFFFF
            for j in range(i + 1, min(sz, i + n)):
                if drop[j]:
                    break
                h = ((h * 116049371) + hashes[j]) & 0xFFFFFFFFFFFFFFFF
                self._push_hash(line, h % self.args.bucket)

    # -- training line parser ------------------------------------------------

    def get_line(self, tokens: list[str], rng: np.random.RandomState,
                 *, skip_oov=False, skip_freq=False
                 ) -> tuple[list[int], list[int], int]:
        """Parse tokens into (word_ids, word_hashes, ntokens)."""
        ids: list[int] = []
        hashes: list[int] = []
        nt = 0
        for tok in tokens:
            if tok == "</s>":
                break
            h = _fnv(tok)
            wid = self.get_id(tok, h)
            if skip_oov and wid < 0:
                continue
            etype = self.entries[wid].type if wid >= 0 else (
                _LABEL if tok.startswith(self.args.label_prefix) else _WORD)
            nt += 1
            if etype == _WORD:
                if skip_freq and rng.random() > self.pdiscard[wid]:
                    continue
                ids.append(wid)
                hashes.append(h)
            if nt > _MAX_LINE:
                break
        return ids, hashes, nt

    # -- binary I/O ----------------------------------------------------------

    def save(self, f):
        n = len(self.entries)
        # C++ layout: int32, int32, int32, int64, int64 = 28 bytes
        f.write(struct.pack("<i", n))
        f.write(struct.pack("<i", self.nwords))
        f.write(struct.pack("<i", self.nlabels))
        f.write(struct.pack("<q", self.ntokens))
        f.write(struct.pack("<q", self._pruneidx_size))
        for e in self.entries:
            f.write(e.word.encode("utf-8") + b"\x00")
            f.write(struct.pack("<qb", e.count, e.type))
        for k, v in self._pruneidx.items():
            f.write(struct.pack("<ii", k, v))

    def load(self, f):
        # C++ layout: int32, int32, int32, int64, int64 = 28 bytes
        n = struct.unpack("<i", f.read(4))[0]
        self.nwords = struct.unpack("<i", f.read(4))[0]
        self.nlabels = struct.unpack("<i", f.read(4))[0]
        self.ntokens = struct.unpack("<q", f.read(8))[0]
        self._pruneidx_size = struct.unpack("<q", f.read(8))[0]
        self.entries = []
        for _ in range(n):
            wb = bytearray()
            while (c := f.read(1)) != b"\x00" and c:
                wb.extend(c)
            count, etype = struct.unpack("<qb", f.read(9))
            self.entries.append(Entry(wb.decode("utf-8", "replace"), count, etype))
        self._pruneidx = {}
        for _ in range(max(0, self._pruneidx_size)):
            k, v = struct.unpack("<ii", f.read(8))
            self._pruneidx[k] = v
        self._init_discard()
        self._init_ngrams()
        sz = max(1, math.ceil(n / 0.7))
        self._w2i = np.full(sz, -1, dtype=np.int32)
        for i, e in enumerate(self.entries):
            self._w2i[self._slot(e.word)] = i

    def word_counts(self) -> list[int]:
        return [e.count for e in self.entries if e.type == _WORD]


# ---------------------------------------------------------------------------
# Sent2Vec
# ---------------------------------------------------------------------------

class Sent2Vec:
    """Pure Python sent2vec model.

    Two ways to create::

        model = Sent2Vec.train("corpus.txt", dim=100)
        model = Sent2Vec.load("model.bin")

    Then embed::

        vec = model.embed("a sentence")
        mat = model.embed(["sent one", "sent two"])  # (n, dim) array
    """

    def __init__(self, args: Args, dictionary: Dictionary,
                 wi: np.ndarray, wo: np.ndarray):
        self.args = args
        self.dictionary = dictionary
        self.wi = wi  # (n_input, dim)  float32
        self.wo = wo  # (n_output, dim) float32

    @property
    def dim(self) -> int:
        return self.args.dim

    # -----------------------------------------------------------------------
    # Embedding
    # -----------------------------------------------------------------------

    def word_vector(self, word: str) -> np.ndarray:
        """Return the embedding for a single word."""
        ngrams = self.dictionary.get_subwords(word)
        if not ngrams:
            return np.zeros(self.dim, dtype=np.float32)
        return self.wi[ngrams].mean(axis=0)

    def embed(self, sentences: str | list[str]) -> np.ndarray:
        """Compute sent2vec embeddings.

        Args:
            sentences: A single sentence string, or a list of sentences.

        Returns:
            A 1-D array (dim,) for a single sentence, or a 2-D array
            (n, dim) for a list.
        """
        if isinstance(sentences, str):
            return self._embed_one(sentences)
        out = np.zeros((len(sentences), self.dim), dtype=np.float32)
        for i, s in enumerate(sentences):
            out[i] = self._embed_one(s)
        return out

    def _embed_one(self, sentence: str) -> np.ndarray:
        svec = np.zeros(self.dim, dtype=np.float32)
        count = 0
        for word in sentence.split():
            wv = self.word_vector(word)
            norm = np.linalg.norm(wv)
            if norm > 0:
                svec += wv / norm
                count += 1
        if count > 0:
            svec /= count
        return svec

    # keep old name as alias
    get_sentence_vector = _embed_one

    # -----------------------------------------------------------------------
    # I/O  (binary-compatible with C++ fasttext)
    # -----------------------------------------------------------------------

    def save(self, path: str):
        """Save model in the C++ fasttext binary format."""
        with open(path, "wb") as f:
            f.write(struct.pack("<ii", _MAGIC, _VERSION))
            f.write(self.args.to_bytes())
            self.dictionary.save(f)
            f.write(struct.pack("<?", False))   # quant_input
            f.write(struct.pack("<qq", *self.wi.shape))
            self.wi.tofile(f)
            f.write(struct.pack("<?", False))   # qout
            f.write(struct.pack("<qq", *self.wo.shape))
            self.wo.tofile(f)

    @classmethod
    def load(cls, path: str) -> "Sent2Vec":
        """Load a model from the C++ fasttext binary format."""
        with open(path, "rb") as f:
            magic, version = struct.unpack("<ii", f.read(8))
            if magic != _MAGIC:
                raise ValueError("Bad magic number -- not a fasttext model")
            if version > _VERSION:
                raise ValueError(f"Model version {version} not supported")

            args = Args.from_bytes(f.read(Args.BIN_SIZE))
            d = Dictionary(args)
            d.load(f)

            if struct.unpack("<?", f.read(1))[0]:
                raise ValueError("Quantised input not supported")
            m, n = struct.unpack("<qq", f.read(16))
            wi = np.frombuffer(f.read(m * n * 4), np.float32).reshape(m, n).copy()

            if struct.unpack("<?", f.read(1))[0]:
                raise ValueError("Quantised output not supported")
            m, n = struct.unpack("<qq", f.read(16))
            wo = np.frombuffer(f.read(m * n * 4), np.float32).reshape(m, n).copy()

        return cls(args, d, wi, wo)

    # keep old name as alias
    load_model = classmethod(lambda cls, p: cls.load(p))
    save_model = save

    def save_vectors(self, path: str):
        """Save word vectors in word2vec text format."""
        nw = self.dictionary.nwords
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"{nw} {self.dim}\n")
            for i in range(nw):
                w = self.dictionary.entries[i].word
                v = self.word_vector(w)
                f.write(f"{w} {' '.join(f'{x:.5f}' for x in v)}\n")

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------

    @classmethod
    def train(cls, input_path: str, *, output_path: str | None = None,
              **kwargs) -> "Sent2Vec":
        """Train a new sent2vec model.

        Args:
            input_path: Training corpus (one sentence per line).
            output_path: If given, save .bin and .vec after training.
            **kwargs: Any ``Args`` field (dim, epoch, lr, neg, min_count,
                      word_ngrams, dropout_k, bucket, t, seed, verbose, ...).

        Returns:
            Trained ``Sent2Vec`` model.
        """
        a = Args(**{k: v for k, v in kwargs.items() if hasattr(Args, k)})
        a.model = _SENT2VEC
        a.loss = _NS
        if a.word_ngrams <= 1 and a.maxn == 0:
            a.bucket = 0

        d = Dictionary(a)
        d.read_from_file(input_path)

        n_in = d.nwords + d.nlabels + a.bucket
        n_out = d.nwords + d.nlabels
        bound = 1.0 / a.dim
        rng_init = np.random.RandomState(a.seed)
        wi = rng_init.uniform(-bound, bound, (n_in, a.dim)).astype(np.float32)
        wo = np.zeros((n_out, a.dim), dtype=np.float32)

        negatives = _build_neg_table(d.word_counts())

        model = cls(a, d, wi, wo)
        model._run_training(input_path, negatives)

        if output_path:
            model.save(output_path + ".bin")
            model.save_vectors(output_path + ".vec")
        return model

    def _run_training(self, input_path: str, negatives: np.ndarray):
        a = self.args
        d = self.dictionary
        dim = a.dim
        total = a.epoch * d.ntokens
        rng = np.random.RandomState(a.seed)
        rng_state = np.int64(a.seed + 1)

        sentences = []
        with open(input_path, "r", encoding="utf-8", errors="replace") as fh:
            for raw in fh:
                toks = raw.split()
                if toks:
                    sentences.append(toks)
        if not sentences:
            raise ValueError("No sentences in input file")

        tok_count = 0
        loss_sum = 0.0
        n_ex = 0
        t0 = time.time()

        for epoch in range(a.epoch):
            for si, toks in enumerate(sentences):
                ids, hashes, nt = d.get_line(toks, rng, skip_oov=True)
                tok_count += nt
                if len(ids) <= 1:
                    continue

                progress = tok_count / total
                lr = a.lr * (1.0 - progress)
                if lr <= 0:
                    break

                ls, n, rng_state = self._sent2vec_step(
                    ids, hashes, lr, negatives, rng, rng_state)
                loss_sum += ls
                n_ex += n

                if a.verbose > 1 and si % 1000 == 0:
                    elapsed = time.time() - t0
                    sys.stderr.write(
                        f"\rProgress: {progress*100:.1f}%"
                        f"  words/sec: {tok_count/max(elapsed,1e-6):.0f}"
                        f"  lr: {lr:.6f}"
                        f"  avg.loss: {loss_sum/max(n_ex,1):.6f}")
                    sys.stderr.flush()
            else:
                continue
            break  # lr <= 0

        if a.verbose > 0:
            sys.stderr.write(
                f"\rProgress: 100.0%  avg.loss: {loss_sum/max(n_ex,1):.6f}\n")
            sys.stderr.flush()

    def _sent2vec_step(self, ids, hashes, lr, negatives, rng, rng_state):
        d = self.dictionary
        a = self.args
        total_loss = 0.0
        total_n = 0

        for w in range(len(ids)):
            wid = ids[w]
            if rng.random() > d.pdiscard[wid]:
                continue
            if d.entries[wid].count < a.min_count_label:
                continue

            bow = list(ids)
            boh = list(hashes)
            bow[w] = 0
            boh[w] = 0

            if a.dropout_k > 0:
                d.add_word_ngrams_dropout(bow, boh, a.word_ngrams, a.dropout_k, rng)
            else:
                d.add_word_ngrams(bow, boh, a.word_ngrams)

            ctx = np.array(bow, dtype=np.int32)
            loss, rng_state = _ns_update(
                self.wi, self.wo, ctx, np.int32(ids[w]),
                a.neg, negatives, a.dim, np.float32(lr), True, rng_state)
            total_loss += float(loss)
            total_n += 1

        return total_loss, total_n, rng_state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_neg_table(counts: list[int]) -> np.ndarray:
    z = sum(c ** 0.5 for c in counts)
    table: list[int] = []
    for i, c in enumerate(counts):
        table.extend([i] * int(c ** 0.5 * _NEG_TABLE / z))
    if not table:
        table.append(0)
    return np.array(table, dtype=np.int32)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli():
    top = argparse.ArgumentParser(prog="sent2vec",
                                  description="sent2vec (pure Python)")
    sub = top.add_subparsers(dest="command")

    # -- train ---------------------------------------------------------------
    tr = sub.add_parser("train", help="Train a sent2vec model")
    tr.add_argument("-input", required=True, dest="input_path")
    tr.add_argument("-output", required=True, dest="output_path")
    tr.add_argument("-dim", type=int, default=100)
    tr.add_argument("-epoch", type=int, default=5)
    tr.add_argument("-lr", type=float, default=0.2)
    tr.add_argument("-neg", type=int, default=10)
    tr.add_argument("-minCount", type=int, default=5, dest="min_count")
    tr.add_argument("-wordNgrams", type=int, default=1, dest="word_ngrams")
    tr.add_argument("-dropoutK", type=int, default=2, dest="dropout_k")
    tr.add_argument("-bucket", type=int, default=2_000_000)
    tr.add_argument("-t", type=float, default=1e-4)
    tr.add_argument("-minn", type=int, default=0)
    tr.add_argument("-maxn", type=int, default=0)
    tr.add_argument("-verbose", type=int, default=2)
    tr.add_argument("-seed", type=int, default=0)

    # -- embed ---------------------------------------------------------------
    em = sub.add_parser("embed", aliases=["print-vec"],
                        help="Print sentence vectors")
    em.add_argument("model_path")

    parsed = top.parse_args()
    if parsed.command is None:
        top.print_help()
        sys.exit(1)

    if parsed.command == "train":
        kw = {k: v for k, v in vars(parsed).items()
              if k not in ("command", "input_path", "output_path")}
        Sent2Vec.train(parsed.input_path, output_path=parsed.output_path, **kw)

    elif parsed.command in ("embed", "print-vec"):
        model = Sent2Vec.load(parsed.model_path)
        sys.stderr.write(
            f"Loaded: {model.dictionary.nwords} words, dim={model.dim}\n"
            f"Enter sentences (one per line, Ctrl-D to stop):\n")
        for line in sys.stdin:
            vec = model.embed(line.strip())
            print(" ".join(f"{v:.5f}" for v in vec))


if __name__ == "__main__":
    _cli()
