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

# ── deterministic hash (matches C++ fasttext/sent2vec exactly) ────────────────

@njit(cache=True)
def _fnv1a_bytes(data):
    """FNV-1a 32-bit over a uint8 array with signed-char XOR.
    Identical to C++ Dictionary::hash()."""
    h = np.uint32(2166136261)
    for i in range(len(data)):
        b = data[i]
        # C++ does: h ^ uint32_t(int8_t(c)) — sign-extends bytes >= 0x80
        sb = np.uint32(b) if b < 128 else np.uint32(np.int32(np.int8(b)))
        h = (h ^ sb) * np.uint32(16777619)
    # return as signed int32 to match C++ int32_t storage
    return np.int32(h)

def _fnv1a(s: str) -> int:
    """FNV-1a 32-bit — thin wrapper that encodes str to bytes then calls numba."""
    return int(_fnv1a_bytes(np.frombuffer(s.encode("utf-8"), dtype=np.uint8)))

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

@njit(cache=True)
def _word_ngram_ids(hashes, n, nwords, bucket, drop):
    """Compute bucket indices for word n-gram features (numba-accelerated).
    drop: int32 array of positions to skip (-1 terminated or empty)."""
    _M = np.uint64(0xFFFFFFFFFFFFFFFF)
    sz = len(hashes)
    # worst case: sz * (n-1) entries
    buf = np.empty(sz * n, np.int32)
    pos = 0
    for i in range(sz):
        # check if i is in drop set
        skip = False
        for d in range(len(drop)):
            if drop[d] == i:
                skip = True
                break
        if skip:
            continue
        h = np.uint64(np.int64(hashes[i])) & _M  # sign-extend int32→uint64
        for j in range(i + 1, min(sz, i + n)):
            skip_j = False
            for d in range(len(drop)):
                if drop[d] == j:
                    skip_j = True
                    break
            if skip_j:
                break
            h = (h * np.uint64(116049371) + (np.uint64(np.int64(hashes[j])) & _M)) & _M
            buf[pos] = np.int32(nwords + np.int32(h % np.uint64(bucket)))
            pos += 1
    return buf[:pos]

@njit(cache=True)
def _train_sentence(wi, wo, ids, hashes, pdiscard, neg_table,
                    word_ngrams, bucket, dropout_k, nwords, dim,
                    neg, lr, rng_state, rng_discard):
    """Process one sentence: iterate over targets, build context, run SGD.
    Returns (loss_sum, n_steps, rng_state, rng_discard)."""
    n = len(ids)
    loss_sum = np.float32(0.0)
    n_steps = np.int32(0)
    empty_drop = np.empty(0, np.int32)
    _MASK48 = np.uint64(0xFFFFFFFFFFFF)
    _MULT = np.uint64(25214903917)
    _INC = np.uint64(11)
    _MASK16 = np.uint64(0xFFFF)

    for w in range(n):
        rng_discard = (rng_discard * _MULT + _INC) & _MASK48
        p = np.float64(rng_discard & _MASK16) / 65536.0
        if p > np.float64(pdiscard[ids[w]]):
            continue

        target = ids[w]
        # save & mask target position (avoids copying entire arrays)
        old_id, old_h = ids[w], hashes[w]
        ids[w] = np.int32(0)
        hashes[w] = np.int32(0)

        # word n-grams with optional dropout
        if word_ngrams > 1 and bucket > 0:
            if dropout_k > 0 and n > 2:
                drop_buf = np.empty(dropout_k, np.int32)
                n_drop = np.int32(0)
                while n_drop < dropout_k and n - n_drop > 2:
                    rng_discard = (rng_discard * _MULT + _INC) & _MASK48
                    pos = np.int32(1 + (rng_discard % np.uint64(n - 1)))
                    already = False
                    for d in range(n_drop):
                        if drop_buf[d] == pos:
                            already = True
                            break
                    if not already:
                        drop_buf[n_drop] = pos
                        n_drop += 1
                drop = drop_buf[:n_drop]
            else:
                drop = empty_drop
            ngrams = _word_ngram_ids(hashes, word_ngrams, nwords, bucket, drop)
            ctx_arr = np.empty(n + len(ngrams), np.int32)
            for k in range(n):
                ctx_arr[k] = ids[k]
            for k in range(len(ngrams)):
                ctx_arr[n + k] = ngrams[k]
        else:
            ctx_arr = ids

        loss, rng_state = _ns_step(wi, wo, ctx_arr, target,
                                   neg, neg_table, dim,
                                   np.float32(lr), rng_state)
        # restore
        ids[w] = old_id
        hashes[w] = old_h
        loss_sum += loss
        n_steps += 1

    return loss_sum, n_steps, rng_state, rng_discard



# ── vocabulary ───────────────────────────────────────────────────────────────

@dataclass
class Vocab:
    words: list[str]         = field(default_factory=list)
    counts: np.ndarray       = field(default_factory=lambda: np.empty(0, np.int64))
    w2i: dict[str, int]      = field(default_factory=dict)
    whash: dict[str, int]    = field(default_factory=dict)  # pre-computed hashes
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
        # pre-hash every word once — encode to bytes then call numba kernel
        whash  = {w: int(_fnv1a_bytes(np.frombuffer(w.encode("utf-8"), dtype=np.uint8)))
                  for w in words}

        if verbose > 0:
            print(f"\rRead {ntokens // 1_000_000}M words — "
                  f"vocab {len(words) - 1} (after min_count={min_count})",
                  file=sys.stderr)

        bkt = bucket if word_ngrams > 1 else 0
        return cls(words=words, counts=counts, w2i=w2i, whash=whash,
                   ntokens=ntokens, bucket=bkt, word_ngrams=word_ngrams, t=t)

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

    def tokenise(self, tokens: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Map raw tokens → (word_ids, hashes) as int32 arrays.
        Unknown words are silently skipped."""
        w2i, whash = self.w2i, self.whash
        ids, hashes = [], []
        for tok in tokens:
            wid = w2i.get(tok, -1)
            if wid >= 0:
                ids.append(wid)
                hashes.append(whash[tok])
        return np.array(ids, dtype=np.int32), np.array(hashes, dtype=np.int32)

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
        whash = {w: int(_fnv1a_bytes(np.frombuffer(w.encode("utf-8"), dtype=np.uint8)))
                 for w in words}
        vocab = Vocab(
            words=words,
            counts=counts,
            w2i={w: i for i, w in enumerate(words)},
            whash=whash,
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

        neg_table = _build_neg_table(vocab.counts)  # includes placeholder (count=0)

        model = cls(vocab=vocab, wi=wi, wo=wo, dim=dim, neg=neg, lr=lr,
                    word_ngrams=word_ngrams, dropout_k=dropout_k,
                    epoch=epoch, seed=seed, verbose=verbose)
        model._fit(corpus, neg_table)
        return model

    def _fit(self, corpus: str, neg_table: np.ndarray):
        v = self.vocab
        pdiscard = v.discard_prob
        total = self.epoch * v.ntokens
        nwords = np.int32(len(v))
        rng_state = np.int64(self.seed + 1)
        rng_discard = np.uint64(self.seed)

        # pre-tokenise entire corpus to numpy arrays (once, not per epoch)
        corpus_ids = []
        corpus_hashes = []
        with open(corpus, encoding="utf-8", errors="replace") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                ids, hashes = v.tokenise(stripped.split())
                if len(ids) > 1:
                    corpus_ids.append(ids)
                    corpus_hashes.append(hashes)

        tok_count = 0
        loss_acc, n_acc = 0.0, 0
        t0 = time.time()

        # warm up numba on a tiny call
        _dummy_ids = np.array([1, 2], np.int32)
        _dummy_h = np.array([0, 0], np.int32)
        _train_sentence(self.wi, self.wo, _dummy_ids, _dummy_h,
                        pdiscard, neg_table, self.word_ngrams,
                        np.int32(v.bucket), np.int32(self.dropout_k),
                        nwords, np.int32(self.dim), np.int32(self.neg),
                        np.float32(self.lr), rng_state, rng_discard)

        for ep in range(self.epoch):
            for si in range(len(corpus_ids)):
                ids = corpus_ids[si]
                hashes = corpus_hashes[si]
                tok_count += len(ids)

                progress = tok_count / total
                cur_lr = self.lr * (1.0 - progress)
                if cur_lr <= 0:
                    break

                loss, steps, rng_state, rng_discard = _train_sentence(
                    self.wi, self.wo, ids, hashes, pdiscard, neg_table,
                    np.int32(self.word_ngrams), np.int32(v.bucket),
                    np.int32(self.dropout_k), nwords, np.int32(self.dim),
                    np.int32(self.neg), np.float32(cur_lr),
                    rng_state, rng_discard)
                loss_acc += float(loss)
                n_acc += int(steps)

                if self.verbose > 1 and si % 1000 == 0:
                    _progress(tok_count, total, t0, cur_lr, loss_acc, n_acc)
            else:
                continue
            break

        if self.verbose > 0:
            print(f"\rDone — avg loss {loss_acc / max(n_acc, 1):.4f}"
                  f"  ({time.time() - t0:.1f}s)", file=sys.stderr)

# ── helpers ──────────────────────────────────────────────────────────────────

_NEG_TABLE_SIZE = 10_000_000

def _build_neg_table(counts: np.ndarray) -> np.ndarray:
    """Build negative-sampling table — matches C++ NegativeSamplingLoss exactly.

    counts includes placeholder at index 0 (count=0).
    Table stores 0-based word indices; placeholder gets 0 entries."""
    sqrt_c = np.power(counts.astype(np.float64), 0.5)
    z = sqrt_c.sum()
    # each word i gets floor(sqrt(c_i) * TABLE_SIZE / z) entries — matches C++ loop
    slots = (sqrt_c * _NEG_TABLE_SIZE / z).astype(np.intp)
    return np.repeat(np.arange(len(counts), dtype=np.int32), slots)

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
