"""sent2vec — sentence embeddings via unsupervised compositional n-gram features.

    model = Sent2Vec.train("corpus.txt", dim=100, epoch=5)
    model.save("model.npz")

    model = Sent2Vec.load("model.npz")
    model.embed("a sentence")                          # → (dim,)
    model.embed(["sentence one", "sentence two"])       # → (n, dim)

Requires only **numpy** and **numba** (no C compiler, no scipy).
"""

from __future__ import annotations

import argparse, math, struct, sys, time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numba import njit

# ── lookup tables (built once at import) ─────────────────────────────────────

@njit(cache=True)
def _make_tables():
    S, L = 512, 512
    sig = np.empty(S + 1, np.float32)
    log = np.empty(L + 1, np.float32)
    for i in range(S + 1):
        sig[i] = 1.0 / (1.0 + math.exp(-(float(i) * 16.0 / S - 8.0)))
    for i in range(L + 1):
        log[i] = math.log((float(i) + 1e-5) / L)
    return sig, log

_SIG, _LOG = _make_tables()

# ── deterministic hash ───────────────────────────────────────────────────────

_FNV_OFFSET = 0xcbf29ce484222325
_FNV_PRIME  = 0x00000100000001B3
_MASK64     = 0xFFFFFFFFFFFFFFFF

def _fnv1a(s: str) -> int:
    """FNV-1a 64-bit hash — deterministic across processes, unlike hash()."""
    h = _FNV_OFFSET
    for b in s.encode("utf-8"):
        h = ((h ^ b) * _FNV_PRIME) & _MASK64
    return h

# ── numba kernels ────────────────────────────────────────────────────────────

@njit(cache=True)
def _sigmoid(x):
    if x < -8.0:  return np.float32(0.0)
    if x >  8.0:  return np.float32(1.0)
    return _SIG[int((x + 8.0) * 32.0)]                # 512 / 16 = 32

@njit(cache=True)
def _logp(x):
    if x > 1.0: return np.float32(0.0)
    return _LOG[int(x * 512.0)]

@njit(cache=True)
def _dot(a, b):
    s = np.float32(0.0)
    for i in range(len(a)):
        s += a[i] * b[i]
    return s

@njit(cache=True)
def _ns_step(wi, wo, ctx, target, neg, neg_table, dim, lr, rng):
    """One negative-sampling SGD step.  Returns (loss, rng)."""
    n = len(ctx)
    if n == 0:
        return np.float32(0.0), rng

    # hidden = mean of context embeddings
    h = np.zeros(dim, np.float32)
    for k in range(n):
        h += wi[ctx[k]]
    h *= np.float32(1.0 / n)

    g = np.zeros(dim, np.float32)
    neg_sz = len(neg_table)

    # positive example
    score = _sigmoid(_dot(wo[target], h))
    alpha = np.float32(lr * (1.0 - score))
    g += alpha * wo[target]
    wo[target] += alpha * h
    loss = -_logp(score)

    # negative examples
    for _ in range(neg):
        rng = np.int64((rng * 48271) % 2147483647)
        ni = neg_table[int(np.uint64(rng) % np.uint64(neg_sz))]
        while ni == target:
            rng = np.int64((rng * 48271) % 2147483647)
            ni = neg_table[int(np.uint64(rng) % np.uint64(neg_sz))]
        score = _sigmoid(_dot(wo[ni], h))
        alpha = np.float32(lr * (0.0 - score))
        g += alpha * wo[ni]
        wo[ni] += alpha * h
        loss += -_logp(np.float32(1.0) - score)

    # propagate to input, normalised by context size
    g *= np.float32(1.0 / n)
    for k in range(n):
        wi[ctx[k]] += g

    return loss, rng

# ── vocabulary ───────────────────────────────────────────────────────────────

@dataclass
class Vocab:
    words: list[str]         = field(default_factory=list)
    counts: np.ndarray       = field(default_factory=lambda: np.empty(0, np.int64))
    w2i: dict[str, int]      = field(default_factory=dict)
    ntokens: int             = 0
    bucket: int              = 0
    word_ngrams: int         = 1
    t: float                 = 1e-4

    @classmethod
    def build(cls, path: str, *, min_count=5, bucket=2_000_000,
              word_ngrams=1, t=1e-4, verbose=2) -> Vocab:
        freq: Counter[str] = Counter()
        ntokens = 0
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                for tok in line.split():
                    freq[tok] += 1
                    ntokens += 1
                    if verbose > 1 and ntokens % 1_000_000 == 0:
                        print(f"\rRead {ntokens // 1_000_000}M words",
                              end="", file=sys.stderr)

        # filter, sort by frequency desc, prepend placeholder at index 0
        words  = ["<PLACEHOLDER>"] + [w for w, c in freq.most_common() if c >= min_count]
        counts = np.array([0] + [freq[w] for w in words[1:]], dtype=np.int64)
        w2i    = {w: i for i, w in enumerate(words)}

        if verbose > 0:
            print(f"\rRead {ntokens // 1_000_000}M words — "
                  f"vocab {len(words) - 1} (after min_count={min_count})",
                  file=sys.stderr)

        return cls(words=words, counts=counts, w2i=w2i, ntokens=ntokens,
                   bucket=(bucket if word_ngrams > 1 else 0),
                   word_ngrams=word_ngrams, t=t)

    def __len__(self) -> int:
        return len(self.words)

    @property
    def discard_prob(self) -> np.ndarray:
        """Subsampling keep-probabilities (higher = more likely to keep)."""
        f = self.counts / max(self.ntokens, 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            p = np.sqrt(self.t / f) + self.t / f
        p[0] = 1.0                                     # placeholder always kept
        return p.astype(np.float32)

    def tokenise(self, tokens: list[str]) -> tuple[list[int], list[int]]:
        """Map raw tokens → (word_ids, hashes_for_ngrams).
        Unknown words are silently skipped."""
        ids, hashes = [], []
        for tok in tokens:
            wid = self.w2i.get(tok, -1)
            if wid >= 0:
                ids.append(wid)
                hashes.append(_fnv1a(tok))
        return ids, hashes

    def word_ngram_ids(self, hashes: list[int], *, drop: set[int] | None = None
                       ) -> list[int]:
        """Compute bucket indices for word n-gram features."""
        if self.word_ngrams <= 1 or self.bucket == 0:
            return []
        out: list[int] = []
        n = self.word_ngrams
        sz = len(hashes)
        for i in range(sz):
            if drop and i in drop:
                continue
            h = hashes[i]
            for j in range(i + 1, min(sz, i + n)):
                if drop and j in drop:
                    break
                h = ((h * 116049371) + hashes[j]) & 0xFFFFFFFFFFFFFFFF
                out.append(len(self) + int(h % self.bucket))
        return out

# ── model ────────────────────────────────────────────────────────────────────

class Sent2Vec:
    """Pure-Python sent2vec.

    ::

        model = Sent2Vec.train("corpus.txt", dim=100)
        model.embed("the cat sat on the mat")
    """

    __slots__ = ("wi", "wo", "vocab", "dim", "neg", "word_ngrams",
                 "dropout_k", "lr", "epoch", "seed", "verbose")

    def __init__(self, *, vocab: Vocab, wi: np.ndarray, wo: np.ndarray,
                 dim: int, neg: int = 10, word_ngrams: int = 1,
                 dropout_k: int = 2, lr: float = 0.2, epoch: int = 5,
                 seed: int = 0, verbose: int = 2):
        self.vocab, self.wi, self.wo = vocab, wi, wo
        self.dim = dim
        self.neg, self.word_ngrams, self.dropout_k = neg, word_ngrams, dropout_k
        self.lr, self.epoch, self.seed, self.verbose = lr, epoch, seed, verbose

    # ── embedding ────────────────────────────────────────────────────────────

    def word_vector(self, word: str) -> np.ndarray:
        wid = self.vocab.w2i.get(word, -1)
        if wid < 0:
            return np.zeros(self.dim, np.float32)
        return self.wi[wid].copy()

    def embed(self, text: str | list[str]) -> np.ndarray:
        """Embed a sentence or list of sentences.

        Returns (dim,) for a single string, (n, dim) for a list.
        """
        if isinstance(text, str):
            return self._embed_one(text)
        out = np.empty((len(text), self.dim), np.float32)
        for i, s in enumerate(text):
            out[i] = self._embed_one(s)
        return out

    def _embed_one(self, sentence: str) -> np.ndarray:
        vec = np.zeros(self.dim, np.float32)
        n = 0
        for word in sentence.split():
            wv = self.word_vector(word)
            norm = np.linalg.norm(wv)
            if norm > 0:
                vec += wv / norm
                n += 1
        return vec / n if n else vec

    # ── I/O ──────────────────────────────────────────────────────────────────

    def save(self, path: str):
        """Save to a single .npz file."""
        np.savez_compressed(
            path,
            wi=self.wi, wo=self.wo,
            words=np.array(self.vocab.words, dtype=object),
            counts=self.vocab.counts,
            meta=np.array([self.dim, self.neg, self.vocab.word_ngrams,
                           self.dropout_k, self.epoch, self.seed,
                           self.vocab.ntokens, self.vocab.bucket]),
            fmeta=np.array([self.lr, self.vocab.t]),
        )

    @classmethod
    def load(cls, path: str) -> Sent2Vec:
        """Load from .npz produced by ``save()``."""
        d = np.load(path, allow_pickle=True)
        words = list(d["words"])
        counts = d["counts"]
        m = d["meta"]
        fm = d["fmeta"]
        vocab = Vocab(
            words=words,
            counts=counts,
            w2i={w: i for i, w in enumerate(words)},
            ntokens=int(m[6]),
            bucket=int(m[7]),
            word_ngrams=int(m[2]),
            t=float(fm[1]),
        )
        return cls(
            vocab=vocab, wi=d["wi"], wo=d["wo"],
            dim=int(m[0]), neg=int(m[1]), word_ngrams=int(m[2]),
            dropout_k=int(m[3]), epoch=int(m[4]), seed=int(m[5]),
            lr=float(fm[0]), verbose=2,
        )

    # ── training ─────────────────────────────────────────────────────────────

    @classmethod
    def train(cls, corpus: str, *, dim=100, epoch=5, lr=0.2, neg=10,
              min_count=5, word_ngrams=1, dropout_k=2, bucket=2_000_000,
              t=1e-4, seed=0, verbose=2) -> Sent2Vec:
        """Train a new model from a text file (one sentence per line)."""

        vocab = Vocab.build(corpus, min_count=min_count, bucket=bucket,
                            word_ngrams=word_ngrams, t=t, verbose=verbose)

        n_in  = len(vocab) + vocab.bucket
        n_out = len(vocab)
        rng   = np.random.RandomState(seed)
        wi    = (rng.uniform(-1, 1, (n_in, dim)) / dim).astype(np.float32)
        wo    = np.zeros((n_out, dim), np.float32)

        neg_table = _build_neg_table(vocab.counts[1:])  # skip placeholder

        model = cls(vocab=vocab, wi=wi, wo=wo, dim=dim, neg=neg, lr=lr,
                    word_ngrams=word_ngrams, dropout_k=dropout_k,
                    epoch=epoch, seed=seed, verbose=verbose)
        model._fit(corpus, neg_table)
        return model

    def _fit(self, corpus: str, neg_table: np.ndarray):
        v = self.vocab
        pdiscard = v.discard_prob
        total = self.epoch * v.ntokens
        rng = np.random.RandomState(self.seed)
        rng_state = np.int64(self.seed + 1)

        # pre-load corpus
        sentences = [line.split() for line in
                     open(corpus, encoding="utf-8", errors="replace")
                     if line.strip()]

        tok_count = 0
        loss_acc, n_acc = 0.0, 0
        t0 = time.time()

        for ep in range(self.epoch):
            for si, toks in enumerate(sentences):
                ids, hashes = v.tokenise(toks)
                tok_count += len(ids)
                if len(ids) <= 1:
                    continue

                progress = tok_count / total
                cur_lr = self.lr * (1.0 - progress)
                if cur_lr <= 0:
                    break

                for w in range(len(ids)):
                    if rng.random() > pdiscard[ids[w]]:
                        continue

                    # context = sentence with target replaced by placeholder
                    ctx = list(ids)
                    ctx_h = list(hashes)
                    ctx[w], ctx_h[w] = 0, 0

                    # word n-grams (with optional dropout)
                    drop = None
                    if self.dropout_k > 0 and len(ctx) > 2:
                        drop = set()
                        while len(drop) < self.dropout_k and len(ctx) - len(drop) > 2:
                            drop.add(rng.randint(1, len(ctx) - 1))
                    ngrams = v.word_ngram_ids(ctx_h, drop=drop)
                    ctx_arr = np.array(ctx + ngrams, dtype=np.int32)

                    loss, rng_state = _ns_step(
                        self.wi, self.wo, ctx_arr, np.int32(ids[w]),
                        self.neg, neg_table, self.dim,
                        np.float32(cur_lr), rng_state)
                    loss_acc += float(loss)
                    n_acc += 1

                if self.verbose > 1 and si % 1000 == 0:
                    _progress(tok_count, total, t0, cur_lr, loss_acc, n_acc)
            else:
                continue
            break

        if self.verbose > 0:
            print(f"\rDone — avg loss {loss_acc / max(n_acc, 1):.4f}"
                  f"  ({time.time() - t0:.1f}s)", file=sys.stderr)

# ── helpers ──────────────────────────────────────────────────────────────────

def _build_neg_table(counts: np.ndarray, size: int = 10_000_000) -> np.ndarray:
    sqrt_c = np.sqrt(counts.astype(np.float64))
    prob = sqrt_c / sqrt_c.sum()
    table = np.zeros(size, np.int32)
    i, cum = 0, prob[0]
    for j in range(size):
        while i < len(prob) - 1 and j / size > cum:
            i += 1
            cum += prob[i]
        table[j] = i + 1                               # +1 to skip placeholder
    return table

def _progress(tok, total, t0, lr, loss, n):
    pct = tok / total * 100
    wps = tok / max(time.time() - t0, 1e-6)
    avg = loss / max(n, 1)
    print(f"\r{pct:5.1f}%  {wps:,.0f} w/s  lr={lr:.5f}  loss={avg:.4f}",
          end="", file=sys.stderr)

# ── CLI ──────────────────────────────────────────────────────────────────────

def _cli():
    p = argparse.ArgumentParser(prog="sent2vec")
    sub = p.add_subparsers(dest="cmd")

    tr = sub.add_parser("train")
    tr.add_argument("corpus")
    tr.add_argument("-o", "--output", required=True)
    tr.add_argument("--dim",         type=int,   default=100)
    tr.add_argument("--epoch",       type=int,   default=5)
    tr.add_argument("--lr",          type=float, default=0.2)
    tr.add_argument("--neg",         type=int,   default=10)
    tr.add_argument("--min-count",   type=int,   default=5)
    tr.add_argument("--word-ngrams", type=int,   default=1)
    tr.add_argument("--dropout-k",   type=int,   default=2)
    tr.add_argument("--bucket",      type=int,   default=2_000_000)
    tr.add_argument("--seed",        type=int,   default=0)

    em = sub.add_parser("embed")
    em.add_argument("model")

    args = p.parse_args()
    if args.cmd == "train":
        m = Sent2Vec.train(
            args.corpus, dim=args.dim, epoch=args.epoch, lr=args.lr,
            neg=args.neg, min_count=args.min_count,
            word_ngrams=args.word_ngrams, dropout_k=args.dropout_k,
            bucket=args.bucket, seed=args.seed)
        m.save(args.output)
    elif args.cmd == "embed":
        m = Sent2Vec.load(args.model)
        print(f"Loaded {len(m.vocab)} words, dim={m.dim}", file=sys.stderr)
        for line in sys.stdin:
            v = m.embed(line.strip())
            print(" ".join(f"{x:.5f}" for x in v))
    else:
        p.print_help()

if __name__ == "__main__":
    _cli()
