"""sent2vec — sentence embeddings via unsupervised compositional n-gram features.

    model = Sent2Vec.train("corpus.txt", dim=100, epoch=5)
    model.save("model.npz")

    model = Sent2Vec.load("model.npz")
    model.embed("a sentence")                          # → (dim,)
    model.embed(["sentence one", "sentence two"])       # → (n, dim)

Requires only **numpy** and **numba** (no C compiler, no scipy).
"""

from __future__ import annotations

import argparse, math, sys, time
from collections import Counter
from dataclasses import dataclass, field

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

# ── monolithic epoch kernel ──────────────────────────────────────────────────
#
# ALL training logic in one @njit function: sentence iteration, subsampling,
# context building, word n-grams, negative-sampling SGD.
# Uses flat arrays + scalar indexing (no slices, no copies).
# Called once per epoch from Python — eliminates ~100K dispatch calls.

@njit(fastmath=True, cache=True)
def _train_epoch(wi, wo, flat_ids, flat_hashes, offsets, n_sentences,
                 pdiscard, neg_table, word_ngrams, bucket, dropout_k,
                 nwords, dim, neg, base_lr, total_tokens,
                 rng_state, rng_discard, tok_count,
                 sig_table, log_table):
    """Train one full epoch over all sentences.

    Returns (loss_sum, n_steps, tok_count, rng_state, rng_discard).
    """
    neg_sz = len(neg_table)
    loss_sum = np.float32(0.0)
    n_steps = np.int32(0)

    # LCG constants for subsampling RNG
    _MASK48 = np.uint64(0xFFFFFFFFFFFF)
    _MULT   = np.uint64(25214903917)
    _INC    = np.uint64(11)
    _MASK16 = np.uint64(0xFFFF)
    # n-gram hash mask
    _M = np.uint64(0xFFFFFFFFFFFFFFFF)

    # Pre-allocate reusable buffers sized for the longest sentence
    max_n = np.int32(0)
    for s in range(n_sentences):
        slen = np.int32(offsets[s + 1] - offsets[s])
        if slen > max_n:
            max_n = slen
    wng = max(word_ngrams, np.int32(1))
    ctx_buf  = np.empty(max_n * wng, np.int32)
    drop_buf = np.empty(max(dropout_k, np.int32(1)), np.int32)
    h = np.empty(dim, np.float32)
    g = np.empty(dim, np.float32)

    for s in range(n_sentences):
        off = offsets[s]
        n = np.int32(offsets[s + 1] - off)
        tok_count += np.int64(n)

        progress = np.float64(tok_count) / np.float64(total_tokens)
        lr = np.float32(np.float64(base_lr) * (1.0 - progress))
        if lr <= np.float32(0.0):
            break

        for w in range(n):
            wid = flat_ids[off + w]
            rng_discard = (rng_discard * _MULT + _INC) & _MASK48
            pv = np.float64(rng_discard & _MASK16) / 65536.0
            if pv > np.float64(pdiscard[wid]):
                continue

            target = wid

            # save & mask target position
            old_id = flat_ids[off + w]
            old_h  = flat_hashes[off + w]
            flat_ids[off + w]    = np.int32(0)
            flat_hashes[off + w] = np.int32(0)

            # ── build context (scalar indexing into flat arrays) ──
            n_ctx = np.int32(0)
            for k in range(n):
                ctx_buf[n_ctx] = flat_ids[off + k]
                n_ctx += 1

            # word n-gram features with optional dropout
            if word_ngrams > 1 and bucket > 0:
                n_drop = np.int32(0)
                if dropout_k > 0 and n > 2:
                    while n_drop < dropout_k and n - n_drop > 2:
                        rng_discard = (rng_discard * _MULT + _INC) & _MASK48
                        pos = np.int32(1 + np.int32(rng_discard % np.uint64(n - 1)))
                        already = False
                        for di in range(n_drop):
                            if drop_buf[di] == pos:
                                already = True
                                break
                        if not already:
                            drop_buf[n_drop] = pos
                            n_drop += 1

                for i in range(n):
                    skip = False
                    for di in range(n_drop):
                        if drop_buf[di] == i:
                            skip = True
                            break
                    if skip:
                        continue
                    hv = np.uint64(np.int64(flat_hashes[off + i])) & _M
                    for j in range(i + 1, min(n, i + word_ngrams)):
                        skip_j = False
                        for di in range(n_drop):
                            if drop_buf[di] == j:
                                skip_j = True
                                break
                        if skip_j:
                            break
                        hv = (hv * np.uint64(116049371) + (np.uint64(np.int64(flat_hashes[off + j])) & _M)) & _M
                        ctx_buf[n_ctx] = np.int32(nwords + np.int32(hv % np.uint64(bucket)))
                        n_ctx += 1

            # restore target position
            flat_ids[off + w]    = old_id
            flat_hashes[off + w] = old_h

            if n_ctx == 0:
                continue

            # ── negative-sampling SGD (fully inlined) ──
            inv_n = np.float32(1.0 / np.float32(n_ctx))

            # hidden = mean of context embeddings (C-style loops for SIMD)
            for d in range(dim):
                h[d] = np.float32(0.0)
            for k in range(n_ctx):
                row = ctx_buf[k]
                for d in range(dim):
                    h[d] += wi[row, d]
            for d in range(dim):
                h[d] *= inv_n

            # gradient accumulator
            for d in range(dim):
                g[d] = np.float32(0.0)

            # positive example
            dot_val = np.float32(0.0)
            for d in range(dim):
                dot_val += wo[target, d] * h[d]
            if dot_val < np.float32(-8.0):
                score = np.float32(0.0)
            elif dot_val > np.float32(8.0):
                score = np.float32(1.0)
            else:
                score = sig_table[np.int32((dot_val + np.float32(8.0)) * np.float32(32.0))]
            alpha = lr * (np.float32(1.0) - score)
            for d in range(dim):
                g[d] += alpha * wo[target, d]
                wo[target, d] += alpha * h[d]
            if score > np.float32(1.0):
                loss = np.float32(0.0)
            else:
                loss = -log_table[np.int32(score * np.float32(512.0))]

            # negative examples
            for _neg_i in range(neg):
                rng_state = np.int64((rng_state * np.int64(48271)) % np.int64(2147483647))
                ni = neg_table[np.int64(np.uint64(rng_state) % np.uint64(neg_sz))]
                while ni == target:
                    rng_state = np.int64((rng_state * np.int64(48271)) % np.int64(2147483647))
                    ni = neg_table[np.int64(np.uint64(rng_state) % np.uint64(neg_sz))]

                dot_val = np.float32(0.0)
                for d in range(dim):
                    dot_val += wo[ni, d] * h[d]
                if dot_val < np.float32(-8.0):
                    score = np.float32(0.0)
                elif dot_val > np.float32(8.0):
                    score = np.float32(1.0)
                else:
                    score = sig_table[np.int32((dot_val + np.float32(8.0)) * np.float32(32.0))]
                alpha = lr * (np.float32(0.0) - score)
                for d in range(dim):
                    g[d] += alpha * wo[ni, d]
                    wo[ni, d] += alpha * h[d]
                one_m = np.float32(1.0) - score
                if one_m > np.float32(1.0):
                    lp = np.float32(0.0)
                else:
                    lp = log_table[np.int32(one_m * np.float32(512.0))]
                loss -= lp

            # propagate gradient to input embeddings
            for d in range(dim):
                g[d] *= inv_n
            for k in range(n_ctx):
                row = ctx_buf[k]
                for d in range(dim):
                    wi[row, d] += g[d]

            loss_sum += loss
            n_steps += 1

    return loss_sum, n_steps, tok_count, rng_state, rng_discard


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
        nlines = 0
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                for tok in line.split():
                    freq[tok] += 1
                    ntokens += 1
                    if verbose > 1 and ntokens % 1_000_000 == 0:
                        print(f"\rRead {ntokens // 1_000_000}M words",
                              end="", file=sys.stderr)
                nlines += 1

        # C++ counts </s> (one per line) in ntokens
        ntokens += nlines

        # vocab: <PLACEHOLDER> at 0, </s> at 1 (highest count), then by freq desc
        # C++ sorts by count desc; placeholder gets count=1e18 to land at [0],
        # </s> (count=nlines) naturally follows, then real words.
        real_words = [w for w, c in freq.most_common() if c >= min_count]
        words  = ["<PLACEHOLDER>", "</s>"] + real_words
        counts = np.array([0, nlines] + [freq[w] for w in real_words], dtype=np.int64)
        w2i    = {w: i for i, w in enumerate(words)}
        # pre-hash every word once — encode to bytes then call numba kernel
        whash  = {w: int(_fnv1a_bytes(np.frombuffer(w.encode("utf-8"), dtype=np.uint8)))
                  for w in words}

        if verbose > 0:
            print(f"\rRead {ntokens // 1_000_000}M words — "
                  f"vocab {len(words) - 2} (after min_count={min_count})",
                  file=sys.stderr)

        bkt = bucket if word_ngrams > 1 else 0
        return cls(words=words, counts=counts, w2i=w2i, whash=whash,
                   ntokens=ntokens, bucket=bkt, word_ngrams=word_ngrams, t=t)

    def __len__(self) -> int:
        return len(self.words)

    @property
    def discard_prob(self) -> np.ndarray:
        """Subsampling discard table: pdiscard = sqrt(t/f) + t/f.

        Matches C++ initTableDiscard(). In C++ the placeholder is given
        count=1e18 during this computation (making pdiscard~0), then reset
        to 0 afterwards. Since placeholder never appears in training
        sentences, we approximate by giving it pdiscard=0 (always discarded
        if encountered) which has the same no-op effect."""
        counts = self.counts.copy()
        counts[0] = int(1e18)                          # match C++ placeholder count
        f = counts / max(self.ntokens, 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            p = np.sqrt(self.t / f) + self.t / f
        return p.astype(np.float32)

    def tokenise(self, tokens: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Map raw tokens → (word_ids, hashes) as int32 arrays.
        Unknown words and </s> are silently skipped (matches C++ SKIP_EOS|SKIP_OOV)."""
        w2i, whash = self.w2i, self.whash
        ids, hashes = [], []
        for tok in tokens:
            if tok == "</s>":
                break                                  # SKIP_EOS: stop on EOS
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

        # pre-tokenise entire corpus and flatten into contiguous arrays
        sentences_ids = []
        sentences_hashes = []
        with open(corpus, encoding="utf-8", errors="replace") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                ids, hashes = v.tokenise(stripped.split())
                if len(ids) > 1:
                    sentences_ids.append(ids)
                    sentences_hashes.append(hashes)

        n_sentences = len(sentences_ids)
        offsets = np.empty(n_sentences + 1, np.int64)
        offsets[0] = 0
        for i in range(n_sentences):
            offsets[i + 1] = offsets[i] + len(sentences_ids[i])
        total_toks = int(offsets[n_sentences])
        flat_ids = np.empty(total_toks, np.int32)
        flat_hashes = np.empty(total_toks, np.int32)
        for i in range(n_sentences):
            a, b = int(offsets[i]), int(offsets[i + 1])
            flat_ids[a:b] = sentences_ids[i]
            flat_hashes[a:b] = sentences_hashes[i]
        del sentences_ids, sentences_hashes

        tok_count = np.int64(0)
        loss_acc, n_acc = 0.0, 0
        t0 = time.time()
        ep = 0

        # C++ loops: while tokenCount < epoch * ntokens, wrapping file at EOF.
        # Since ntokens includes </s> but training tokens don't, this requires
        # more than `epoch` passes. We call _train_epoch (one full pass per call)
        # in a while-loop until tok_count reaches total.
        while tok_count < np.int64(total):
            loss, steps, tok_count, rng_state, rng_discard = _train_epoch(
                self.wi, self.wo, flat_ids, flat_hashes, offsets,
                np.int32(n_sentences), pdiscard, neg_table,
                np.int32(self.word_ngrams), np.int32(v.bucket),
                np.int32(self.dropout_k), nwords, np.int32(self.dim),
                np.int32(self.neg), np.float32(self.lr),
                np.int64(total), rng_state, rng_discard,
                tok_count, _SIG, _LOG)
            loss_acc += float(loss)
            n_acc += int(steps)
            ep += 1

            if self.verbose > 0:
                elapsed = max(time.time() - t0, 1e-6)
                wps = int(tok_count) / elapsed
                pct = min(int(tok_count) / total * 100, 100.0)
                avg = loss_acc / max(n_acc, 1)
                print(f"\r{pct:5.1f}%  {wps:,.0f} w/s  pass={ep}"
                      f"  loss={avg:.4f}",
                      end="", file=sys.stderr)

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
