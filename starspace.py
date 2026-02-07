"""starspace — StarSpace-compatible "Embed All The Things" in pure Python.

Implements trainMode=0 (classification/tagging) with:
- Hinge (margin ranking) loss + negative sampling
- Cosine similarity (L2-normalized embeddings)
- AdaGrad optimisation
- Shared LHS/RHS embedding matrix
- Word n-grams

::

    # From a file (convenience):
    model = StarSpace.train("train.txt", dim=100, epoch=5)
    model.test("test.txt")                     # → (N, P@1, R@1)

    # From any iterable of token lists:
    lines = [["__label__pos", "love", "this"], ["__label__neg", "awful"]]
    model = StarSpace.train(lines, dim=100, epoch=5)
    model.test(iter_lines("test.txt"))

    model.predict("the food was great")        # → [("__label__pos", 0.92)]

Requires only **numpy** and **numba** (no C compiler, no scipy).
"""

from __future__ import annotations

import argparse, math, os, sys, tempfile, time
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Iterator

import numpy as np
from numba import njit

# ── public helpers ────────────────────────────────────────────────────────────

def iter_lines(path: str) -> Iterator[list[str]]:
    """Yield tokenized lines from a text file.

    Each yielded item is a list of tokens (words and/or ``__label__`` tags).
    This is the bridge between file-based I/O and the iterator-based core API::

        model = StarSpace.train(iter_lines("train.txt"))
    """
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            tokens = line.split()
            if tokens:
                yield tokens

# ── deterministic hash (matches C++ StarSpace / fasttext) ────────────────────

@njit(cache=True)
def _fnv1a_bytes(data):
    """FNV-1a 32-bit over a uint8 array with signed-char XOR."""
    h = np.uint32(2166136261)
    for i in range(len(data)):
        b = data[i]
        sb = np.uint32(b) if b < 128 else np.uint32(np.int32(np.int8(b)))
        h = (h ^ sb) * np.uint32(16777619)
    return np.int32(h)

# ── monolithic epoch kernel (hinge loss, cosine similarity, AdaGrad) ─────────
#
# ALL training for one epoch in a single @njit call.
# Per-example:  LHS = bag-of-words + n-grams, RHS = single label.
# Hinge loss: max(0, margin - cos(lhs, rhs+) + cos(lhs, rhs-))
# AdaGrad: per-row accumulated gradient for adaptive learning rates.

@njit(fastmath=True, cache=True)
def _train_epoch(emb, adagrad, flat_lhs, flat_hashes, flat_labels,
                 lhs_offsets, label_offsets, n_examples,
                 neg_pool, neg_pool_size,
                 nwords, nlabels, dim, word_ngrams, bucket,
                 margin, neg_search_limit, base_lr, total_tokens,
                 rng_state, tok_count, norm_limit):
    """Train one full epoch using hinge loss with cosine similarity.

    Returns (loss_sum, n_steps, tok_count, rng_state).
    """
    loss_sum = np.float64(0.0)
    n_steps = np.int32(0)
    _M = np.uint64(0xFFFFFFFFFFFFFFFF)

    # Size buffers for longest sentence
    max_n = np.int32(0)
    for s in range(n_examples):
        slen = np.int32(lhs_offsets[s + 1] - lhs_offsets[s])
        if slen > max_n:
            max_n = slen
    wng = max(word_ngrams, np.int32(1))
    ctx_buf = np.empty(max_n * wng, np.int32)

    lhs_vec = np.empty(dim, np.float32)
    rhs_pos = np.empty(dim, np.float32)
    rhs_neg = np.empty(dim, np.float32)
    neg_mean = np.empty(dim, np.float32)
    grad_w = np.empty(dim, np.float32)
    neg_ids = np.empty(neg_search_limit, np.int32)
    neg_flags = np.empty(neg_search_limit, np.int32)
    ngram_base = nwords + nlabels

    for s in range(n_examples):
        in_start = lhs_offsets[s]
        in_end = lhs_offsets[s + 1]
        n_words = np.int32(in_end - in_start)

        lb_start = label_offsets[s]
        lb_end = label_offsets[s + 1]
        n_lb = np.int32(lb_end - lb_start)

        if n_words == 0 or n_lb == 0:
            continue

        tok_count += np.int64(n_words)

        progress = np.float64(tok_count) / np.float64(total_tokens)
        cur_lr = np.float32(np.float64(base_lr) * (1.0 - progress))
        if cur_lr <= np.float32(0.0):
            break

        # Randomly select one label
        rng_state = np.int64((rng_state * np.int64(48271)) % np.int64(2147483647))
        target = flat_labels[lb_start + np.int32(
            np.uint64(rng_state) % np.uint64(n_lb))]

        # ── build LHS features (words + word n-grams) ──
        n_ctx = np.int32(0)
        for k in range(n_words):
            ctx_buf[n_ctx] = flat_lhs[in_start + k]
            n_ctx += 1

        if word_ngrams > 1 and bucket > 0:
            for i in range(n_words):
                hv = np.uint64(np.int64(flat_hashes[in_start + i])) & _M
                for j in range(i + 1, min(n_words, i + word_ngrams)):
                    hv = (hv * np.uint64(116049371) +
                          (np.uint64(np.int64(flat_hashes[in_start + j]))
                           & _M)) & _M
                    ctx_buf[n_ctx] = np.int32(
                        ngram_base + np.int32(hv % np.uint64(bucket)))
                    n_ctx += 1

        if n_ctx == 0:
            continue

        # ── LHS embedding: sum + L2 normalise ──
        for d in range(dim):
            lhs_vec[d] = np.float32(0.0)
        for k in range(n_ctx):
            row = ctx_buf[k]
            for d in range(dim):
                lhs_vec[d] += emb[row, d]
        norm_sq = np.float32(0.0)
        for d in range(dim):
            norm_sq += lhs_vec[d] * lhs_vec[d]
        inv_norm = np.float32(1.0 / np.float32(
            np.sqrt(np.float64(norm_sq)) + 1e-10))
        for d in range(dim):
            lhs_vec[d] *= inv_norm

        # ── RHS positive: L2 normalise ──
        norm_sq = np.float32(0.0)
        for d in range(dim):
            rhs_pos[d] = emb[target, d]
            norm_sq += rhs_pos[d] * rhs_pos[d]
        inv_norm = np.float32(1.0 / np.float32(
            np.sqrt(np.float64(norm_sq)) + 1e-10))
        for d in range(dim):
            rhs_pos[d] *= inv_norm

        pos_sim = np.float32(0.0)
        for d in range(dim):
            pos_sim += lhs_vec[d] * rhs_pos[d]

        # ── negative sampling + hinge loss ──
        for d in range(dim):
            neg_mean[d] = np.float32(0.0)
        n_valid = np.int32(0)

        for ni in range(neg_search_limit):
            rng_state = np.int64(
                (rng_state * np.int64(48271)) % np.int64(2147483647))
            neg_label = neg_pool[np.int64(
                np.uint64(rng_state) % np.uint64(neg_pool_size))]
            neg_ids[ni] = neg_label

            if neg_label == target:
                neg_flags[ni] = np.int32(0)
                continue

            # Negative embedding: L2 normalise
            norm_sq = np.float32(0.0)
            for d in range(dim):
                rhs_neg[d] = emb[neg_label, d]
                norm_sq += rhs_neg[d] * rhs_neg[d]
            inv_norm = np.float32(1.0 / np.float32(
                np.sqrt(np.float64(norm_sq)) + 1e-10))
            for d in range(dim):
                rhs_neg[d] *= inv_norm

            neg_sim = np.float32(0.0)
            for d in range(dim):
                neg_sim += lhs_vec[d] * rhs_neg[d]

            triplet_loss = margin - pos_sim + neg_sim
            if triplet_loss > np.float32(0.0):
                for d in range(dim):
                    neg_mean[d] += rhs_neg[d]
                n_valid += 1
                neg_flags[ni] = np.int32(1)
                loss_sum += np.float64(triplet_loss)
            else:
                neg_flags[ni] = np.int32(0)

        if n_valid == 0:
            n_steps += 1
            continue

        # ── gradient: gradW = mean(negatives) - positive ──
        for d in range(dim):
            grad_w[d] = neg_mean[d] / np.float32(n_valid) - rhs_pos[d]

        # ── update LHS features (AdaGrad) ──
        n1 = np.float32(1.0) / np.float32(n_ctx)
        for k in range(n_ctx):
            row = ctx_buf[k]
            adagrad[row] += n1 / np.float32(dim)
            eff_lr = cur_lr / np.float32(
                np.sqrt(np.float64(adagrad[row]) + 1e-6))
            for d in range(dim):
                emb[row, d] -= eff_lr * grad_w[d]

        # ── update RHS positive: push toward LHS (AdaGrad) ──
        pos_rate = (np.float32(np.float64(cur_lr) /
                    np.float64(neg_search_limit)) * np.float32(n_valid))
        adagrad[target] += np.float32(1.0) / np.float32(dim)
        eff_lr = pos_rate / np.float32(
            np.sqrt(np.float64(adagrad[target]) + 1e-6))
        for d in range(dim):
            emb[target, d] += eff_lr * lhs_vec[d]

        # ── update RHS negatives: push away from LHS (AdaGrad) ──
        neg_rate = np.float32(np.float64(cur_lr) /
                              np.float64(neg_search_limit))
        for ni in range(neg_search_limit):
            if neg_flags[ni] == np.int32(1):
                nl = neg_ids[ni]
                adagrad[nl] += np.float32(1.0) / np.float32(dim)
                eff_lr = neg_rate / np.float32(
                    np.sqrt(np.float64(adagrad[nl]) + 1e-6))
                for d in range(dim):
                    emb[nl, d] -= eff_lr * lhs_vec[d]

        # ── norm clipping for updated rows ──
        if norm_limit > np.float32(0.0):
            for k in range(n_ctx):
                row = ctx_buf[k]
                rnorm = np.float32(0.0)
                for d in range(dim):
                    rnorm += emb[row, d] * emb[row, d]
                rnorm = np.float32(np.sqrt(np.float64(rnorm)))
                if rnorm > norm_limit:
                    scale = norm_limit / rnorm
                    for d in range(dim):
                        emb[row, d] *= scale
            rnorm = np.float32(0.0)
            for d in range(dim):
                rnorm += emb[target, d] * emb[target, d]
            rnorm = np.float32(np.sqrt(np.float64(rnorm)))
            if rnorm > norm_limit:
                scale = norm_limit / rnorm
                for d in range(dim):
                    emb[target, d] *= scale

        n_steps += 1

    return loss_sum, n_steps, tok_count, rng_state

# ── vocabulary ───────────────────────────────────────────────────────────────

@dataclass
class Vocab:
    words: list[str]            = field(default_factory=list)
    labels: list[str]           = field(default_factory=list)
    w2i: dict[str, int]         = field(default_factory=dict)
    l2i: dict[str, int]         = field(default_factory=dict)
    whash: dict[str, int]       = field(default_factory=dict)
    ntokens: int                = 0
    bucket: int                 = 0
    word_ngrams: int            = 1
    label_prefix: str           = "__label__"

    @classmethod
    def build(cls, data: Iterable[list[str]], *, min_count=1,
              bucket=2_000_000, word_ngrams=1, label_prefix="__label__",
              verbose=2) -> Vocab:
        word_freq: Counter[str] = Counter()
        label_freq: Counter[str] = Counter()
        ntokens = 0
        for tokens in data:
            for tok in tokens:
                ntokens += 1
                if tok.startswith(label_prefix):
                    label_freq[tok] += 1
                else:
                    word_freq[tok] += 1
                if verbose > 1 and ntokens % 1_000_000 == 0:
                    print(f"\rRead {ntokens // 1_000_000}M words",
                          end="", file=sys.stderr)

        real_words = [w for w, c in word_freq.most_common() if c >= min_count]
        labels = [l for l, _ in label_freq.most_common()]

        w2i = {w: i for i, w in enumerate(real_words)}
        l2i = {l: i for i, l in enumerate(labels)}
        whash = {w: int(_fnv1a_bytes(
            np.frombuffer(w.encode("utf-8"), dtype=np.uint8)))
            for w in real_words}

        bkt = bucket if word_ngrams > 1 else 0

        if verbose > 0:
            print(f"\rRead {ntokens // 1_000_000}M words — "
                  f"vocab {len(real_words)} words, {len(labels)} labels "
                  f"(min_count={min_count})", file=sys.stderr)

        return cls(words=real_words, labels=labels, w2i=w2i, l2i=l2i,
                   whash=whash, ntokens=ntokens, bucket=bkt,
                   word_ngrams=word_ngrams, label_prefix=label_prefix)

    @property
    def nwords(self) -> int:
        return len(self.words)

    @property
    def nlabels(self) -> int:
        return len(self.labels)

    def tokenise_line(self, tokens: list[str]
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Parse tokens → (word_ids, word_hashes, label_emb_ids).

        word_ids are [0, nwords).
        label_emb_ids are global embedding indices [nwords, nwords+nlabels).
        """
        w2i, whash, l2i = self.w2i, self.whash, self.l2i
        prefix = self.label_prefix
        nw = self.nwords

        word_ids, word_hashes, label_ids = [], [], []
        for tok in tokens:
            if tok.startswith(prefix):
                lid = l2i.get(tok)
                if lid is not None:
                    label_ids.append(nw + lid)  # global embedding index
            else:
                wid = w2i.get(tok, -1)
                if wid >= 0:
                    word_ids.append(wid)
                    word_hashes.append(whash[tok])

        return (np.array(word_ids, dtype=np.int32),
                np.array(word_hashes, dtype=np.int32),
                np.array(label_ids, dtype=np.int32))

# ── model ────────────────────────────────────────────────────────────────────

class StarSpace:
    """Pure-Python StarSpace classifier/embedder.

    ::

        model = StarSpace.train("train.txt", dim=100, epoch=5)
        model.predict("the food was great")
    """

    __slots__ = ("emb", "vocab", "dim", "word_ngrams", "margin",
                 "neg_search_limit", "lr", "epoch", "seed",
                 "norm_limit", "verbose")

    def __init__(self, *, vocab: Vocab, emb: np.ndarray,
                 dim: int, word_ngrams: int = 1, margin: float = 0.05,
                 neg_search_limit: int = 50, lr: float = 0.01,
                 epoch: int = 5, seed: int = 0, norm_limit: float = 1.0,
                 verbose: int = 2):
        self.vocab, self.emb = vocab, emb
        self.dim = dim
        self.word_ngrams = word_ngrams
        self.margin = margin
        self.neg_search_limit = neg_search_limit
        self.lr, self.epoch, self.seed = lr, epoch, seed
        self.norm_limit = norm_limit
        self.verbose = verbose

    # ── prediction ────────────────────────────────────────────────────────

    def predict(self, text: str, k: int = 1) -> list[tuple[str, float]]:
        """Predict top-k labels. Returns [(label, cosine_similarity), ...]."""
        v = self.vocab
        word_ids, word_hashes, _ = v.tokenise_line(text.split())
        if len(word_ids) == 0:
            return []

        # Build input features: words + n-gram buckets
        input_ids = list(word_ids)
        if v.word_ngrams > 1 and v.bucket > 0:
            _M = np.uint64(0xFFFFFFFFFFFFFFFF)
            ngram_base = v.nwords + v.nlabels
            for i in range(len(word_hashes)):
                hv = np.uint64(np.int64(word_hashes[i])) & _M
                for j in range(i + 1, min(len(word_hashes),
                                          i + v.word_ngrams)):
                    hv = (hv * np.uint64(116049371) +
                          (np.uint64(np.int64(word_hashes[j])) & _M)) & _M
                    input_ids.append(ngram_base + int(hv % np.uint64(v.bucket)))

        # LHS embedding: sum + L2 normalise
        lhs = np.zeros(self.dim, np.float32)
        for idx in input_ids:
            lhs += self.emb[idx]
        n = np.linalg.norm(lhs)
        if n > 0:
            lhs /= n

        # Cosine similarity with all label embeddings
        nw = v.nwords
        label_embs = self.emb[nw:nw + v.nlabels]
        norms = np.linalg.norm(label_embs, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        sims = (label_embs / norms) @ lhs

        top_k = np.argsort(sims)[::-1][:k]
        return [(v.labels[i], float(sims[i])) for i in top_k]

    def test(self, data, k: int = 1) -> tuple[int, float, float]:
        """Evaluate on labeled data. Returns (N, precision@k, recall@k).

        *data* is a file path (str) or an iterable of token lists.
        """
        if isinstance(data, str):
            data = iter_lines(data)
        v = self.vocab
        n = 0
        p_sum = 0.0
        r_sum = 0.0

        for tokens in data:
            true_labels = set()
            text_tokens = []
            for tok in tokens:
                if tok.startswith(v.label_prefix):
                    true_labels.add(tok)
                else:
                    text_tokens.append(tok)

            if not true_labels or not text_tokens:
                continue

            preds = self.predict(" ".join(text_tokens), k=k)
            pred_labels = {label for label, _ in preds}

            matches = len(pred_labels & true_labels)
            p_sum += matches / max(len(pred_labels), 1)
            r_sum += matches / len(true_labels)
            n += 1

        precision = p_sum / max(n, 1)
        recall = r_sum / max(n, 1)
        return n, precision, recall

    # ── I/O ──────────────────────────────────────────────────────────────

    def save(self, path: str):
        np.savez_compressed(
            path,
            emb=self.emb,
            words=np.array(self.vocab.words, dtype=object),
            labels=np.array(self.vocab.labels, dtype=object),
            meta=np.array([self.dim, self.word_ngrams, self.epoch, self.seed,
                           self.vocab.ntokens, self.vocab.bucket,
                           self.neg_search_limit]),
            fmeta=np.array([self.lr, self.margin, self.norm_limit]),
            label_prefix=np.array([self.vocab.label_prefix]),
        )

    @classmethod
    def load(cls, path: str) -> StarSpace:
        d = np.load(path, allow_pickle=True)
        words = list(d["words"])
        labels = list(d["labels"])
        m = d["meta"]
        fm = d["fmeta"]
        lp = str(d["label_prefix"][0])

        whash = {w: int(_fnv1a_bytes(
            np.frombuffer(w.encode("utf-8"), dtype=np.uint8)))
            for w in words}
        vocab = Vocab(
            words=words, labels=labels,
            w2i={w: i for i, w in enumerate(words)},
            l2i={l: i for i, l in enumerate(labels)},
            whash=whash,
            ntokens=int(m[4]),
            bucket=int(m[5]),
            word_ngrams=int(m[1]),
            label_prefix=lp,
        )
        return cls(
            vocab=vocab, emb=d["emb"],
            dim=int(m[0]), word_ngrams=int(m[1]),
            epoch=int(m[2]), seed=int(m[3]),
            neg_search_limit=int(m[6]),
            lr=float(fm[0]), margin=float(fm[1]),
            norm_limit=float(fm[2]), verbose=2,
        )

    # ── training ─────────────────────────────────────────────────────────

    @classmethod
    def train(cls, data, *, dim=100, epoch=5, lr=0.01,
              margin=0.05, neg_search_limit=50, min_count=1,
              word_ngrams=1, bucket=2_000_000, norm_limit=1.0,
              init_rand_sd=0.001, seed=0, verbose=2) -> StarSpace:
        """Train a StarSpace model.

        *data* is a file path (str) or an iterable of token lists, where each
        token list mixes ``__label__*`` tags with ordinary words::

            model = StarSpace.train("train.txt")
            model = StarSpace.train([["__label__pos", "great", "movie"]])
        """
        if isinstance(data, str):
            # File path — can iterate twice (vocab + training) cheaply.
            vocab = Vocab.build(iter_lines(data), min_count=min_count,
                                bucket=bucket, word_ngrams=word_ngrams,
                                verbose=verbose)
            train_data = iter_lines(data)
        else:
            # Arbitrary iterable — materialise for two passes.
            lines = data if isinstance(data, (list, tuple)) else list(data)
            vocab = Vocab.build(lines, min_count=min_count, bucket=bucket,
                                word_ngrams=word_ngrams, verbose=verbose)
            train_data = lines

        # Shared embedding: [words | labels | n-gram buckets]
        n_emb = vocab.nwords + vocab.nlabels + vocab.bucket
        rng = np.random.RandomState(seed)
        emb = (rng.normal(0, init_rand_sd, (n_emb, dim))).astype(np.float32)

        model = cls(vocab=vocab, emb=emb, dim=dim, word_ngrams=word_ngrams,
                    margin=margin, neg_search_limit=neg_search_limit,
                    lr=lr, epoch=epoch, seed=seed, norm_limit=norm_limit,
                    verbose=verbose)
        model._fit(train_data)
        return model

    def _fit(self, data: Iterable[list[str]]):
        v = self.vocab
        total = self.epoch * v.ntokens
        rng_state = np.int64(self.seed + 1)

        # AdaGrad accumulator (per-row)
        adagrad = np.zeros(self.emb.shape[0], np.float32)

        # Tokenise → stream to temp binary files (mmap'd)
        tmp_dir = tempfile.mkdtemp(prefix="ss_")
        ids_path = os.path.join(tmp_dir, "ids.bin")
        hash_path = os.path.join(tmp_dir, "hash.bin")
        lbl_path = os.path.join(tmp_dir, "lbl.bin")
        ioff_path = os.path.join(tmp_dir, "ioff.bin")
        loff_path = os.path.join(tmp_dir, "loff.bin")
        neg_path = os.path.join(tmp_dir, "neg.bin")

        f_ids = open(ids_path, "wb")
        f_hash = open(hash_path, "wb")
        f_lbl = open(lbl_path, "wb")
        f_ioff = open(ioff_path, "wb")
        f_loff = open(loff_path, "wb")
        f_neg = open(neg_path, "wb")
        f_ioff.write(np.int64(0).tobytes())
        f_loff.write(np.int64(0).tobytes())

        for tokens in data:
            word_ids, word_hashes, label_ids = v.tokenise_line(tokens)
            if len(word_ids) > 0 and len(label_ids) > 0:
                f_ids.write(word_ids.tobytes())
                f_hash.write(word_hashes.tobytes())
                f_lbl.write(label_ids.tobytes())
                f_ioff.write(np.int64(f_ids.tell() // 4).tobytes())
                f_loff.write(np.int64(f_lbl.tell() // 4).tobytes())
                f_neg.write(label_ids.tobytes())

        f_ids.close()
        f_hash.close()
        f_lbl.close()
        f_ioff.close()
        f_loff.close()
        f_neg.close()

        flat_ids = np.memmap(ids_path, dtype=np.int32, mode="r")
        flat_hashes = np.memmap(hash_path, dtype=np.int32, mode="r")
        flat_labels = np.memmap(lbl_path, dtype=np.int32, mode="r")
        input_offsets = np.memmap(ioff_path, dtype=np.int64, mode="r")
        label_offsets = np.memmap(loff_path, dtype=np.int64, mode="r")
        neg_pool = np.memmap(neg_path, dtype=np.int32, mode="r")
        n_examples = len(input_offsets) - 1
        neg_pool_size = len(neg_pool)

        tok_count = np.int64(0)
        loss_acc, n_acc = 0.0, 0
        t0 = time.time()
        ep = 0

        while tok_count < np.int64(total):
            loss, steps, tok_count, rng_state = _train_epoch(
                self.emb, adagrad,
                flat_ids, flat_hashes, flat_labels,
                input_offsets, label_offsets,
                np.int32(n_examples), neg_pool, np.int64(neg_pool_size),
                np.int32(v.nwords), np.int32(v.nlabels), np.int32(self.dim),
                np.int32(self.word_ngrams), np.int32(v.bucket),
                np.float32(self.margin), np.int32(self.neg_search_limit),
                np.float32(self.lr), np.int64(total),
                rng_state, tok_count, np.float32(self.norm_limit))
            loss_acc += float(loss)
            n_acc += int(steps)
            ep += 1

            if self.verbose > 0:
                elapsed = max(time.time() - t0, 1e-6)
                wps = int(tok_count) / elapsed
                pct = min(int(tok_count) / total * 100, 100.0)
                avg = loss_acc / max(n_acc, 1) / self.neg_search_limit
                print(f"\r{pct:5.1f}%  {wps:,.0f} w/s  pass={ep}"
                      f"  loss={avg:.4f}",
                      end="", file=sys.stderr)

        if self.verbose > 0:
            avg = loss_acc / max(n_acc, 1) / self.neg_search_limit
            print(f"\rDone — avg loss {avg:.4f}"
                  f"  ({time.time() - t0:.1f}s)", file=sys.stderr)

        # clean up mmap temp files
        del flat_ids, flat_hashes, flat_labels
        del input_offsets, label_offsets, neg_pool
        for p in (ids_path, hash_path, lbl_path, ioff_path, loff_path,
                  neg_path):
            try:
                os.unlink(p)
            except OSError:
                pass
        try:
            os.rmdir(tmp_dir)
        except OSError:
            pass

# ── CLI ──────────────────────────────────────────────────────────────────────

def _cli():
    p = argparse.ArgumentParser(prog="starspace")
    sub = p.add_subparsers(dest="cmd")

    tr = sub.add_parser("train")
    tr.add_argument("corpus")
    tr.add_argument("-o", "--output", required=True)
    tr.add_argument("--dim",              type=int,   default=100)
    tr.add_argument("--epoch",            type=int,   default=5)
    tr.add_argument("--lr",               type=float, default=0.01)
    tr.add_argument("--margin",           type=float, default=0.05)
    tr.add_argument("--neg-search-limit", type=int,   default=50)
    tr.add_argument("--min-count",        type=int,   default=1)
    tr.add_argument("--word-ngrams",      type=int,   default=1)
    tr.add_argument("--bucket",           type=int,   default=2_000_000)
    tr.add_argument("--norm-limit",       type=float, default=1.0)
    tr.add_argument("--init-rand-sd",     type=float, default=0.001)
    tr.add_argument("--seed",             type=int,   default=0)

    ts = sub.add_parser("test")
    ts.add_argument("model")
    ts.add_argument("test_file")
    ts.add_argument("-k", type=int, default=1)

    pr = sub.add_parser("predict")
    pr.add_argument("model")
    pr.add_argument("-k", type=int, default=1)

    args = p.parse_args()
    if args.cmd == "train":
        m = StarSpace.train(
            iter_lines(args.corpus),
            dim=args.dim, epoch=args.epoch, lr=args.lr,
            margin=args.margin, neg_search_limit=args.neg_search_limit,
            min_count=args.min_count, word_ngrams=args.word_ngrams,
            bucket=args.bucket, norm_limit=args.norm_limit,
            init_rand_sd=args.init_rand_sd, seed=args.seed)
        m.save(args.output)
    elif args.cmd == "test":
        m = StarSpace.load(args.model)
        n, prec, rec = m.test(iter_lines(args.test_file), k=args.k)
        print(f"N\t{n}")
        print(f"P@{args.k}\t{prec:.4f}")
        print(f"R@{args.k}\t{rec:.4f}")
    elif args.cmd == "predict":
        m = StarSpace.load(args.model)
        for line in sys.stdin:
            preds = m.predict(line.strip(), k=args.k)
            print(" ".join(f"{label} {sim:.4f}" for label, sim in preds))
    else:
        p.print_help()


if __name__ == "__main__":
    _cli()
