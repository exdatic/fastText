"""starspace — StarSpace "Embed All The Things" in pure Python.

Implements all six training modes (trainMode 0-5) with:
- Hinge (margin ranking) loss + negative sampling
- Cosine similarity (L2-normalized embeddings)
- AdaGrad optimisation
- Shared LHS/RHS embedding matrix
- Word n-grams

Training modes::

    0  Classification   — LHS = words,             RHS = 1 random label
    1  Label from rest  — LHS = words + rest labels, RHS = 1 random label
    2  Inverted labels  — LHS = words + 1 label,   RHS = rest labels
    3  Pair prediction  — LHS = words + 1 label,   RHS = 1 other label
    4  Fixed pair       — LHS = words + label[0],  RHS = label[1]
    5  Word embedding   — LHS = context words,     RHS = target word

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

import argparse, math, os, random as _random, sys, tempfile, time
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
# Supports extra LHS features (labels in LHS for modes 1-4) and
# multi-label RHS (mode 2: sum all RHS labels).
#
# Hinge loss: max(0, margin - cos(lhs, rhs+) + cos(lhs, rhs-))
# AdaGrad: per-row accumulated gradient for adaptive learning rates.

@njit(fastmath=True, cache=True)
def _train_epoch(emb, adagrad, flat_lhs, flat_hashes, flat_labels,
                 flat_lhs_extra, lhs_offsets, label_offsets, extra_offsets,
                 n_examples, neg_pool, neg_pool_size,
                 nwords, nlabels, dim, word_ngrams, bucket,
                 margin, neg_search_limit, base_lr, total_tokens,
                 rng_state, tok_count, norm_limit, multi_rhs):
    """Train one full epoch.  Returns (loss_sum, n_steps, tok_count, rng_state)."""
    loss_sum = np.float64(0.0)
    n_steps = np.int32(0)
    _M = np.uint64(0xFFFFFFFFFFFFFFFF)

    # Size buffers for longest example
    max_n = np.int32(0)
    max_extra = np.int32(0)
    max_rhs = np.int32(0)
    for s in range(n_examples):
        slen = np.int32(lhs_offsets[s + 1] - lhs_offsets[s])
        if slen > max_n:
            max_n = slen
        elen = np.int32(extra_offsets[s + 1] - extra_offsets[s])
        if elen > max_extra:
            max_extra = elen
        rlen = np.int32(label_offsets[s + 1] - label_offsets[s])
        if rlen > max_rhs:
            max_rhs = rlen
    wng = max(word_ngrams, np.int32(1))
    ctx_buf = np.empty(max_n * wng + max_extra, np.int32)
    target_buf = np.empty(max(max_rhs, np.int32(1)), np.int32)

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

        if n_lb == 0:
            continue

        tok_count += np.int64(n_words)

        progress = np.float64(tok_count) / np.float64(total_tokens)
        cur_lr = np.float32(np.float64(base_lr) * (1.0 - progress))
        if cur_lr <= np.float32(0.0):
            break

        # ── resolve RHS target(s) ──
        n_targets = np.int32(0)
        if multi_rhs:
            for k in range(lb_start, lb_end):
                target_buf[n_targets] = flat_labels[k]
                n_targets += 1
        else:
            rng_state = np.int64(
                (rng_state * np.int64(48271)) % np.int64(2147483647))
            target_buf[0] = flat_labels[lb_start + np.int32(
                np.uint64(rng_state) % np.uint64(n_lb))]
            n_targets = np.int32(1)

        # ── build LHS features (words + word n-grams + extra labels) ──
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

        # Append extra LHS features (label embeddings for modes 1-4)
        ex_start = extra_offsets[s]
        ex_end = extra_offsets[s + 1]
        for k in range(ex_start, ex_end):
            ctx_buf[n_ctx] = flat_lhs_extra[k]
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

        # ── RHS positive: sum target embeddings + L2 normalise ──
        for d in range(dim):
            rhs_pos[d] = np.float32(0.0)
        for t in range(n_targets):
            tgt = target_buf[t]
            for d in range(dim):
                rhs_pos[d] += emb[tgt, d]
        norm_sq = np.float32(0.0)
        for d in range(dim):
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

            # Skip if negative is any of the positive targets
            is_pos = False
            for t in range(n_targets):
                if neg_label == target_buf[t]:
                    is_pos = True
                    break
            if is_pos:
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
        for t in range(n_targets):
            tgt = target_buf[t]
            adagrad[tgt] += np.float32(1.0) / np.float32(dim)
            eff_lr = pos_rate / np.float32(
                np.sqrt(np.float64(adagrad[tgt]) + 1e-6))
            for d in range(dim):
                emb[tgt, d] += eff_lr * lhs_vec[d]

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
            for t in range(n_targets):
                tgt = target_buf[t]
                rnorm = np.float32(0.0)
                for d in range(dim):
                    rnorm += emb[tgt, d] * emb[tgt, d]
                rnorm = np.float32(np.sqrt(np.float64(rnorm)))
                if rnorm > norm_limit:
                    scale = norm_limit / rnorm
                    for d in range(dim):
                        emb[tgt, d] *= scale

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
    """Pure-Python StarSpace classifier/embedder (trainMode 0-5).

    ::

        model = StarSpace.train("train.txt", dim=100, epoch=5)
        model.predict("the food was great")
    """

    __slots__ = ("emb", "vocab", "dim", "word_ngrams", "margin",
                 "neg_search_limit", "lr", "epoch", "seed",
                 "norm_limit", "verbose", "train_mode", "ws")

    def __init__(self, *, vocab: Vocab, emb: np.ndarray,
                 dim: int, word_ngrams: int = 1, margin: float = 0.05,
                 neg_search_limit: int = 50, lr: float = 0.01,
                 epoch: int = 5, seed: int = 0, norm_limit: float = 1.0,
                 verbose: int = 2, train_mode: int = 0, ws: int = 5):
        self.vocab, self.emb = vocab, emb
        self.dim = dim
        self.word_ngrams = word_ngrams
        self.margin = margin
        self.neg_search_limit = neg_search_limit
        self.lr, self.epoch, self.seed = lr, epoch, seed
        self.norm_limit = norm_limit
        self.verbose = verbose
        self.train_mode = train_mode
        self.ws = ws

    # ── prediction ────────────────────────────────────────────────────────

    def predict(self, text: str, k: int = 1) -> list[tuple[str, float]]:
        """Predict top-k labels. Returns [(label, cosine_similarity), ...]."""
        v = self.vocab
        if v.nlabels == 0:
            return []
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
        Not available for trainMode 5 (word embedding).
        """
        if self.train_mode == 5:
            raise ValueError("test() is not defined for trainMode 5")
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
                           self.neg_search_limit, self.train_mode, self.ws]),
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
        train_mode = int(m[7]) if len(m) > 7 else 0
        ws = int(m[8]) if len(m) > 8 else 5
        return cls(
            vocab=vocab, emb=d["emb"],
            dim=int(m[0]), word_ngrams=int(m[1]),
            epoch=int(m[2]), seed=int(m[3]),
            neg_search_limit=int(m[6]),
            lr=float(fm[0]), margin=float(fm[1]),
            norm_limit=float(fm[2]), verbose=2,
            train_mode=train_mode, ws=ws,
        )

    # ── training ─────────────────────────────────────────────────────────

    @staticmethod
    def _build_vocab_and_cache(data, cache_dir, *, min_count=1,
                               bucket=2_000_000, word_ngrams=1,
                               label_prefix="__label__", verbose=2):
        """Single pass over *data*: count frequencies + write provisional ID cache.

        Writes ``prov.bin`` (int32 per token) and ``off.bin`` (int64 per line)
        to *cache_dir*.

        Returns ``(vocab, remap, final_hash)`` where *remap[prov_id]* gives
        the final embedding index (or -1 if filtered by min_count) and
        *final_hash[final_word_id]* gives the FNV-1a hash.
        """
        tok2prov: dict[str, int] = {}
        prov_is_label: list[bool] = []
        prov_hash: list[int] = []
        prov_freq: list[int] = []

        prov_path = os.path.join(cache_dir, "prov.bin")
        off_path = os.path.join(cache_dir, "off.bin")
        f_prov = open(prov_path, "wb")
        f_off = open(off_path, "wb")
        f_off.write(np.int64(0).tobytes())

        ntokens = 0
        for tokens in data:
            for tok in tokens:
                ntokens += 1
                pid = tok2prov.get(tok)
                if pid is None:
                    pid = len(tok2prov)
                    tok2prov[tok] = pid
                    is_lbl = tok.startswith(label_prefix)
                    prov_is_label.append(is_lbl)
                    prov_hash.append(
                        0 if is_lbl else int(_fnv1a_bytes(
                            np.frombuffer(tok.encode("utf-8"),
                                          dtype=np.uint8))))
                    prov_freq.append(0)
                prov_freq[pid] += 1
                f_prov.write(np.int32(pid).tobytes())
            f_off.write(np.int64(f_prov.tell() // 4).tobytes())
            if verbose > 1 and ntokens % 1_000_000 == 0:
                print(f"\rRead {ntokens // 1_000_000}M words",
                      end="", file=sys.stderr)

        f_prov.close()
        f_off.close()

        # Separate surviving words and labels, sorted by frequency desc
        prov2str = {v: k for k, v in tok2prov.items()}

        word_items: list[tuple[int, str, int]] = []
        label_items: list[tuple[int, str, int]] = []
        for pid in range(len(prov_freq)):
            s = prov2str[pid]
            if prov_is_label[pid]:
                label_items.append((pid, s, prov_freq[pid]))
            elif prov_freq[pid] >= min_count:
                word_items.append((pid, s, prov_freq[pid]))

        word_items.sort(key=lambda x: -x[2])
        label_items.sort(key=lambda x: -x[2])

        words = [s for _, s, _ in word_items]
        labels = [s for _, s, _ in label_items]
        nwords = len(words)

        # Remap: prov_id → final embedding index, -1 for filtered
        remap = np.full(len(tok2prov), -1, dtype=np.int32)
        for final_wid, (pid, _, _) in enumerate(word_items):
            remap[pid] = final_wid
        for final_lid, (pid, _, _) in enumerate(label_items):
            remap[pid] = nwords + final_lid

        # Hash table: final_word_id → fnv1a
        final_hash = np.array([prov_hash[pid] for pid, _, _ in word_items],
                              dtype=np.int32)

        w2i = {w: i for i, w in enumerate(words)}
        l2i = {l: i for i, l in enumerate(labels)}
        whash = {words[i]: int(final_hash[i]) for i in range(nwords)}
        bkt = bucket if word_ngrams > 1 else 0

        if verbose > 0:
            print(f"\rRead {ntokens // 1_000_000}M words — "
                  f"vocab {nwords} words, {len(labels)} labels "
                  f"(min_count={min_count})", file=sys.stderr)

        vocab = Vocab(words=words, labels=labels, w2i=w2i, l2i=l2i,
                      whash=whash, ntokens=ntokens, bucket=bkt,
                      word_ngrams=word_ngrams, label_prefix=label_prefix)

        return vocab, remap, final_hash

    @classmethod
    def train(cls, data, *, dim=100, epoch=5, lr=0.01,
              margin=0.05, neg_search_limit=50, min_count=1,
              word_ngrams=1, bucket=2_000_000, norm_limit=1.0,
              init_rand_sd=0.001, seed=0, verbose=2,
              train_mode=0, ws=5) -> StarSpace:
        """Train a StarSpace model.

        *data* is a file path (str) or an iterable of token lists, where each
        token list mixes ``__label__*`` tags with ordinary words::

            model = StarSpace.train("train.txt")
            model = StarSpace.train([["__label__pos", "great", "movie"]])

        *train_mode* selects the LHS/RHS construction (0-5).
        *ws* is the context window size for trainMode 5.
        """
        if isinstance(data, str):
            data = iter_lines(data)

        cache_dir = tempfile.mkdtemp(prefix="ss_")
        try:
            vocab, remap, final_hash = cls._build_vocab_and_cache(
                data, cache_dir, min_count=min_count, bucket=bucket,
                word_ngrams=word_ngrams, verbose=verbose)

            n_emb = vocab.nwords + vocab.nlabels + vocab.bucket
            rng = np.random.RandomState(seed)
            emb = (rng.normal(0, init_rand_sd, (n_emb, dim))).astype(
                np.float32)

            model = cls(vocab=vocab, emb=emb, dim=dim,
                        word_ngrams=word_ngrams, margin=margin,
                        neg_search_limit=neg_search_limit,
                        lr=lr, epoch=epoch, seed=seed, norm_limit=norm_limit,
                        verbose=verbose, train_mode=train_mode, ws=ws)
            model._fit(cache_dir, remap, final_hash)
        finally:
            for fn in os.listdir(cache_dir):
                try:
                    os.unlink(os.path.join(cache_dir, fn))
                except OSError:
                    pass
            try:
                os.rmdir(cache_dir)
            except OSError:
                pass

        return model

    def _fit(self, cache_dir, remap, final_hash):
        v = self.vocab
        total = self.epoch * v.ntokens
        rng_state = np.int64(self.seed + 1)
        mode = self.train_mode
        nw = v.nwords

        adagrad = np.zeros(self.emb.shape[0], np.float32)

        def _mmap_or_empty(p, dt):
            if os.path.getsize(p) == 0:
                return np.empty(0, dtype=dt)
            return np.memmap(p, dtype=dt, mode="r")

        # ── Read provisional cache ──
        prov_ids = _mmap_or_empty(
            os.path.join(cache_dir, "prov.bin"), np.int32)
        line_offsets = _mmap_or_empty(
            os.path.join(cache_dir, "off.bin"), np.int64)
        n_lines = len(line_offsets) - 1

        # ── Build training mmaps from cache + remap ──
        tmp_dir = tempfile.mkdtemp(prefix="ss_tr_")
        names = ("ids", "hash", "lbl", "extra", "ioff", "loff", "eoff", "neg")
        paths = {n: os.path.join(tmp_dir, f"{n}.bin") for n in names}

        f = {n: open(p, "wb") for n, p in paths.items()}
        f["ioff"].write(np.int64(0).tobytes())
        f["loff"].write(np.int64(0).tobytes())
        f["eoff"].write(np.int64(0).tobytes())

        rng_py = _random.Random(self.seed)

        for li in range(n_lines):
            start = int(line_offsets[li])
            end = int(line_offsets[li + 1])

            # Remap provisional IDs → (word_ids, word_hashes, label_ids)
            word_ids_l: list[int] = []
            word_hashes_l: list[int] = []
            label_ids_l: list[int] = []
            for j in range(start, end):
                fid = int(remap[prov_ids[j]])
                if fid < 0:
                    continue
                if fid < nw:
                    word_ids_l.append(fid)
                    word_hashes_l.append(int(final_hash[fid]))
                else:
                    label_ids_l.append(fid)

            word_ids = np.array(word_ids_l, dtype=np.int32)
            word_hashes = np.array(word_hashes_l, dtype=np.int32)
            label_ids = np.array(label_ids_l, dtype=np.int32)
            nw_line = len(word_ids)
            nl = len(label_ids)

            if mode == 5:
                for wi in range(nw_line):
                    ctx_start = max(0, wi - self.ws)
                    ctx_end = min(nw_line, wi + self.ws + 1)
                    ctx_w = []
                    ctx_h = []
                    for ci in range(ctx_start, ctx_end):
                        if ci != wi:
                            ctx_w.append(word_ids[ci])
                            ctx_h.append(word_hashes[ci])
                    if not ctx_w:
                        continue
                    cw = np.array(ctx_w, dtype=np.int32)
                    ch = np.array(ctx_h, dtype=np.int32)
                    tw = np.array([word_ids[wi]], dtype=np.int32)
                    f["ids"].write(cw.tobytes())
                    f["hash"].write(ch.tobytes())
                    f["lbl"].write(tw.tobytes())
                    f["extra"].write(b"")
                    f["ioff"].write(np.int64(
                        f["ids"].tell() // 4).tobytes())
                    f["loff"].write(np.int64(
                        f["lbl"].tell() // 4).tobytes())
                    f["eoff"].write(np.int64(
                        f["extra"].tell() // 4).tobytes())
                    f["neg"].write(tw.tobytes())
                continue

            if mode == 0:
                if nw_line == 0 or nl == 0:
                    continue
                lhs_w, lhs_h = word_ids, word_hashes
                lhs_extra = np.empty(0, np.int32)
                rhs = label_ids
            elif mode == 1:
                if nw_line == 0 or nl < 2:
                    continue
                idx = rng_py.randrange(nl)
                lhs_w, lhs_h = word_ids, word_hashes
                lhs_extra = np.array(
                    [label_ids[i] for i in range(nl) if i != idx],
                    dtype=np.int32)
                rhs = np.array([label_ids[idx]], dtype=np.int32)
            elif mode == 2:
                if nw_line == 0 or nl < 2:
                    continue
                idx = rng_py.randrange(nl)
                lhs_w, lhs_h = word_ids, word_hashes
                lhs_extra = np.array([label_ids[idx]], dtype=np.int32)
                rhs = np.array(
                    [label_ids[i] for i in range(nl) if i != idx],
                    dtype=np.int32)
            elif mode == 3:
                if nl < 2:
                    continue
                idx1 = rng_py.randrange(nl)
                idx2 = idx1
                while idx2 == idx1:
                    idx2 = rng_py.randrange(nl)
                lhs_w, lhs_h = word_ids, word_hashes
                lhs_extra = np.array([label_ids[idx1]], dtype=np.int32)
                rhs = np.array([label_ids[idx2]], dtype=np.int32)
            elif mode == 4:
                if nl < 2:
                    continue
                lhs_w, lhs_h = word_ids, word_hashes
                lhs_extra = np.array([label_ids[0]], dtype=np.int32)
                rhs = np.array([label_ids[1]], dtype=np.int32)
            else:
                raise ValueError(f"Unknown train_mode {mode}")

            f["ids"].write(lhs_w.tobytes())
            f["hash"].write(lhs_h.tobytes())
            f["lbl"].write(rhs.tobytes())
            f["extra"].write(lhs_extra.tobytes())
            f["ioff"].write(np.int64(f["ids"].tell() // 4).tobytes())
            f["loff"].write(np.int64(f["lbl"].tell() // 4).tobytes())
            f["eoff"].write(np.int64(f["extra"].tell() // 4).tobytes())
            f["neg"].write(rhs.tobytes())

        del prov_ids, line_offsets

        for fh in f.values():
            fh.close()

        flat_ids = _mmap_or_empty(paths["ids"], np.int32)
        flat_hashes = _mmap_or_empty(paths["hash"], np.int32)
        flat_labels = _mmap_or_empty(paths["lbl"], np.int32)
        flat_extra = _mmap_or_empty(paths["extra"], np.int32)
        input_offsets = _mmap_or_empty(paths["ioff"], np.int64)
        label_offsets = _mmap_or_empty(paths["loff"], np.int64)
        extra_offsets = _mmap_or_empty(paths["eoff"], np.int64)
        neg_pool = _mmap_or_empty(paths["neg"], np.int32)
        n_examples = len(input_offsets) - 1
        neg_pool_size = len(neg_pool)

        if n_examples == 0 or neg_pool_size == 0:
            if self.verbose > 0:
                print("No training examples.", file=sys.stderr)
            self._cleanup_dir(tmp_dir)
            return

        multi_rhs = np.int32(1) if mode == 2 else np.int32(0)

        tok_count = np.int64(0)
        loss_acc, n_acc = 0.0, 0
        t0 = time.time()
        ep = 0

        while tok_count < np.int64(total):
            loss, steps, tok_count, rng_state = _train_epoch(
                self.emb, adagrad,
                flat_ids, flat_hashes, flat_labels,
                flat_extra, input_offsets, label_offsets, extra_offsets,
                np.int32(n_examples), neg_pool, np.int64(neg_pool_size),
                np.int32(v.nwords), np.int32(v.nlabels), np.int32(self.dim),
                np.int32(self.word_ngrams), np.int32(v.bucket),
                np.float32(self.margin), np.int32(self.neg_search_limit),
                np.float32(self.lr), np.int64(total),
                rng_state, tok_count, np.float32(self.norm_limit),
                multi_rhs)
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

        del flat_ids, flat_hashes, flat_labels, flat_extra
        del input_offsets, label_offsets, extra_offsets, neg_pool
        self._cleanup_dir(tmp_dir)

    @staticmethod
    def _cleanup_dir(tmp_dir):
        for fn in os.listdir(tmp_dir):
            try:
                os.unlink(os.path.join(tmp_dir, fn))
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
    tr.add_argument("--train-mode",       type=int,   default=0)
    tr.add_argument("--ws",               type=int,   default=5)

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
            init_rand_sd=args.init_rand_sd, seed=args.seed,
            train_mode=args.train_mode, ws=args.ws)
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
