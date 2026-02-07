"""supervised — fastText-compatible supervised text classification.

    model = Supervised.train("train.txt", dim=100, epoch=25, lr=0.1)
    model.save("model.npz")

    model = Supervised.load("model.npz")
    model.predict("the food was great")        # → [("__label__pos", 0.98)]
    model.test("test.txt")                     # → (N, P@1, R@1)

Requires only **numpy** and **numba** (no C compiler, no scipy).
"""

from __future__ import annotations

import argparse, math, os, sys, tempfile, time
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from numba import njit

# ── deterministic hash (matches C++ fasttext exactly) ────────────────────────

@njit(cache=True)
def _fnv1a_bytes(data):
    """FNV-1a 32-bit over a uint8 array with signed-char XOR."""
    h = np.uint32(2166136261)
    for i in range(len(data)):
        b = data[i]
        sb = np.uint32(b) if b < 128 else np.uint32(np.int32(np.int8(b)))
        h = (h ^ sb) * np.uint32(16777619)
    return np.int32(h)

# ── monolithic epoch kernel (softmax loss) ───────────────────────────────────
#
# One @njit call per epoch. Computes:
# - bag-of-words hidden = mean of input embeddings (words + word n-grams)
# - softmax over all labels
# - cross-entropy loss on randomly selected label
# - gradient back-propagated to wo and wi

@njit(fastmath=True, cache=True)
def _train_epoch(wi, wo, flat_ids, flat_hashes, flat_labels,
                 input_offsets, label_offsets,
                 n_examples, nlabels, dim, word_ngrams, bucket, nwords,
                 base_lr, total_tokens, rng_state, tok_count):
    """Train one full epoch using softmax loss.

    Returns (loss_sum, n_steps, tok_count, rng_state).
    """
    loss_sum = np.float64(0.0)
    n_steps = np.int32(0)
    _M = np.uint64(0xFFFFFFFFFFFFFFFF)

    # Pre-allocate buffers
    h = np.empty(dim, np.float32)
    g = np.empty(dim, np.float32)
    probs = np.empty(nlabels, np.float32)

    # Size ctx_buf for longest sentence including n-grams
    max_n = np.int32(0)
    for s in range(n_examples):
        slen = np.int32(input_offsets[s + 1] - input_offsets[s])
        if slen > max_n:
            max_n = slen
    wng = max(word_ngrams, np.int32(1))
    ctx_buf = np.empty(max_n * wng, np.int32)

    for s in range(n_examples):
        in_start = input_offsets[s]
        in_end = input_offsets[s + 1]
        n_words = np.int32(in_end - in_start)

        lb_start = label_offsets[s]
        lb_end = label_offsets[s + 1]
        n_lb = np.int32(lb_end - lb_start)

        if n_words == 0 or n_lb == 0:
            continue

        tok_count += np.int64(n_words)

        progress = np.float64(tok_count) / np.float64(total_tokens)
        lr = np.float32(np.float64(base_lr) * (1.0 - progress))
        if lr <= np.float32(0.0):
            break

        # Randomly select one label
        rng_state = np.int64((rng_state * np.int64(48271)) % np.int64(2147483647))
        target = flat_labels[lb_start + np.int32(
            np.uint64(rng_state) % np.uint64(n_lb))]

        # ── build input features: word IDs + word n-gram buckets ──
        n_ctx = np.int32(0)
        for k in range(n_words):
            ctx_buf[n_ctx] = flat_ids[in_start + k]
            n_ctx += 1

        if word_ngrams > 1 and bucket > 0:
            for i in range(n_words):
                hv = np.uint64(np.int64(flat_hashes[in_start + i])) & _M
                for j in range(i + 1, min(n_words, i + word_ngrams)):
                    hv = (hv * np.uint64(116049371) +
                          (np.uint64(np.int64(flat_hashes[in_start + j])) & _M)) & _M
                    ctx_buf[n_ctx] = np.int32(
                        nwords + np.int32(hv % np.uint64(bucket)))
                    n_ctx += 1

        if n_ctx == 0:
            continue

        # ── hidden = mean of input embeddings ──
        inv_n = np.float32(1.0 / np.float32(n_ctx))
        for d in range(dim):
            h[d] = np.float32(0.0)
        for k in range(n_ctx):
            row = ctx_buf[k]
            for d in range(dim):
                h[d] += wi[row, d]
        for d in range(dim):
            h[d] *= inv_n

        # ── softmax ──
        max_logit = np.float32(-1e30)
        for i in range(nlabels):
            dot_val = np.float32(0.0)
            for d in range(dim):
                dot_val += wo[i, d] * h[d]
            probs[i] = dot_val
            if dot_val > max_logit:
                max_logit = dot_val

        z = np.float32(0.0)
        for i in range(nlabels):
            probs[i] = np.float32(np.exp(np.float64(probs[i] - max_logit)))
            z += probs[i]
        for i in range(nlabels):
            probs[i] /= z

        # ── loss = -log(P(target)) ──
        p_target = np.float64(probs[target])
        if p_target > 1e-10:
            loss_sum -= np.log(p_target)
        else:
            loss_sum += 30.0

        # ── backprop through softmax ──
        for d in range(dim):
            g[d] = np.float32(0.0)

        for i in range(nlabels):
            label_val = np.float32(1.0) if i == target else np.float32(0.0)
            alpha = lr * (label_val - probs[i])
            for d in range(dim):
                g[d] += alpha * wo[i, d]
                wo[i, d] += alpha * h[d]

        # ── gradient to input embeddings (normalized) ──
        for d in range(dim):
            g[d] *= inv_n
        for k in range(n_ctx):
            row = ctx_buf[k]
            for d in range(dim):
                wi[row, d] += g[d]

        loss_sum += np.float64(0.0)
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
    def build(cls, path: str, *, min_count=1, bucket=2_000_000,
              word_ngrams=1, label_prefix="__label__", verbose=2) -> Vocab:
        word_freq: Counter[str] = Counter()
        label_freq: Counter[str] = Counter()
        ntokens = 0
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                for tok in line.split():
                    ntokens += 1
                    if tok.startswith(label_prefix):
                        label_freq[tok] += 1
                    else:
                        word_freq[tok] += 1
                    if verbose > 1 and ntokens % 1_000_000 == 0:
                        print(f"\rRead {ntokens // 1_000_000}M words",
                              end="", file=sys.stderr)

        # C++ sorts words by count desc, labels by count desc (separate)
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

    def tokenise_line(self, tokens: list[str]) -> tuple[np.ndarray, np.ndarray,
                                                         np.ndarray]:
        """Parse tokens → (word_ids, word_hashes, label_ids).

        word_ids and word_hashes are parallel arrays of in-vocab words.
        label_ids are 0-based label indices.
        """
        w2i, whash, l2i = self.w2i, self.whash, self.l2i
        prefix = self.label_prefix

        word_ids, word_hashes, label_ids = [], [], []
        for tok in tokens:
            if tok.startswith(prefix):
                lid = l2i.get(tok)
                if lid is not None:
                    label_ids.append(lid)
            else:
                wid = w2i.get(tok, -1)
                if wid >= 0:
                    word_ids.append(wid)
                    word_hashes.append(whash[tok])

        return (np.array(word_ids, dtype=np.int32),
                np.array(word_hashes, dtype=np.int32),
                np.array(label_ids, dtype=np.int32))

# ── model ────────────────────────────────────────────────────────────────────

class Supervised:
    """Pure-Python fastText supervised classifier.

    ::

        model = Supervised.train("train.txt", dim=100, epoch=25, lr=0.1)
        model.predict("the food was great")
    """

    __slots__ = ("wi", "wo", "vocab", "dim", "word_ngrams",
                 "lr", "epoch", "seed", "verbose")

    def __init__(self, *, vocab: Vocab, wi: np.ndarray, wo: np.ndarray,
                 dim: int, word_ngrams: int = 1, lr: float = 0.1,
                 epoch: int = 5, seed: int = 0, verbose: int = 2):
        self.vocab, self.wi, self.wo = vocab, wi, wo
        self.dim = dim
        self.word_ngrams = word_ngrams
        self.lr, self.epoch, self.seed, self.verbose = lr, epoch, seed, verbose

    # ── prediction ────────────────────────────────────────────────────────

    def predict(self, text: str, k: int = 1) -> list[tuple[str, float]]:
        """Predict top-k labels. Returns [(label, probability), ...]."""
        v = self.vocab
        word_ids, word_hashes, _ = v.tokenise_line(text.split())
        if len(word_ids) == 0:
            return []

        # Build input features: words + n-gram buckets
        input_ids = list(word_ids)
        if v.word_ngrams > 1 and v.bucket > 0:
            _M = np.uint64(0xFFFFFFFFFFFFFFFF)
            nw = v.nwords
            for i in range(len(word_hashes)):
                hv = np.uint64(np.int64(word_hashes[i])) & _M
                for j in range(i + 1, min(len(word_hashes),
                                          i + v.word_ngrams)):
                    hv = (hv * np.uint64(116049371) +
                          (np.uint64(np.int64(word_hashes[j])) & _M)) & _M
                    input_ids.append(nw + int(hv % np.uint64(v.bucket)))

        # Hidden = mean of input embeddings
        h = np.zeros(self.dim, np.float32)
        for idx in input_ids:
            h += self.wi[idx]
        h /= len(input_ids)

        # Softmax
        logits = self.wo @ h
        logits -= logits.max()
        exp_l = np.exp(logits)
        probs = exp_l / exp_l.sum()

        top_k = np.argsort(probs)[::-1][:k]
        return [(v.labels[i], float(probs[i])) for i in top_k]

    def test(self, path: str, k: int = 1) -> tuple[int, float, float]:
        """Evaluate on a labeled file. Returns (N, precision@k, recall@k)."""
        v = self.vocab
        n = 0
        p_sum = 0.0
        r_sum = 0.0

        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                tokens = line.split()
                if not tokens:
                    continue

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
            wi=self.wi, wo=self.wo,
            words=np.array(self.vocab.words, dtype=object),
            labels=np.array(self.vocab.labels, dtype=object),
            meta=np.array([self.dim, self.word_ngrams, self.epoch, self.seed,
                           self.vocab.ntokens, self.vocab.bucket]),
            fmeta=np.array([self.lr]),
            label_prefix=np.array([self.vocab.label_prefix]),
        )

    @classmethod
    def load(cls, path: str) -> Supervised:
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
            vocab=vocab, wi=d["wi"], wo=d["wo"],
            dim=int(m[0]), word_ngrams=int(m[1]),
            epoch=int(m[2]), seed=int(m[3]),
            lr=float(fm[0]), verbose=2,
        )

    # ── training ─────────────────────────────────────────────────────────

    @classmethod
    def train(cls, corpus: str, *, dim=100, epoch=5, lr=0.1, min_count=1,
              word_ngrams=1, bucket=2_000_000, seed=0,
              verbose=2) -> Supervised:
        """Train from a labeled text file (__label__... tokens per line)."""

        vocab = Vocab.build(corpus, min_count=min_count, bucket=bucket,
                            word_ngrams=word_ngrams, verbose=verbose)

        n_in = vocab.nwords + vocab.bucket
        n_out = vocab.nlabels
        rng = np.random.RandomState(seed)
        wi = (rng.uniform(-1, 1, (n_in, dim)) / dim).astype(np.float32)
        wo = np.zeros((n_out, dim), np.float32)

        model = cls(vocab=vocab, wi=wi, wo=wo, dim=dim,
                    word_ngrams=word_ngrams, lr=lr, epoch=epoch,
                    seed=seed, verbose=verbose)
        model._fit(corpus)
        return model

    def _fit(self, corpus: str):
        v = self.vocab
        total = self.epoch * v.ntokens
        rng_state = np.int64(self.seed + 1)

        # Tokenise → stream to temp binary files (mmap'd, never in RAM)
        tmp_dir = tempfile.mkdtemp(prefix="sup_")
        ids_path = os.path.join(tmp_dir, "ids.bin")
        hash_path = os.path.join(tmp_dir, "hash.bin")
        lbl_path = os.path.join(tmp_dir, "lbl.bin")
        ioff_path = os.path.join(tmp_dir, "ioff.bin")
        loff_path = os.path.join(tmp_dir, "loff.bin")

        f_ids = open(ids_path, "wb")
        f_hash = open(hash_path, "wb")
        f_lbl = open(lbl_path, "wb")
        f_ioff = open(ioff_path, "wb")
        f_loff = open(loff_path, "wb")
        f_ioff.write(np.int64(0).tobytes())
        f_loff.write(np.int64(0).tobytes())

        with open(corpus, encoding="utf-8", errors="replace") as f:
            for line in f:
                tokens = line.split()
                if not tokens:
                    continue
                word_ids, word_hashes, label_ids = v.tokenise_line(tokens)
                if len(word_ids) > 0 and len(label_ids) > 0:
                    f_ids.write(word_ids.tobytes())
                    f_hash.write(word_hashes.tobytes())
                    f_lbl.write(label_ids.tobytes())
                    f_ioff.write(np.int64(f_ids.tell() // 4).tobytes())
                    f_loff.write(np.int64(f_lbl.tell() // 4).tobytes())

        f_ids.close()
        f_hash.close()
        f_lbl.close()
        f_ioff.close()
        f_loff.close()

        flat_ids = np.memmap(ids_path, dtype=np.int32, mode="r")
        flat_hashes = np.memmap(hash_path, dtype=np.int32, mode="r")
        flat_labels = np.memmap(lbl_path, dtype=np.int32, mode="r")
        input_offsets = np.memmap(ioff_path, dtype=np.int64, mode="r")
        label_offsets = np.memmap(loff_path, dtype=np.int64, mode="r")
        n_examples = len(input_offsets) - 1

        tok_count = np.int64(0)
        loss_acc, n_acc = 0.0, 0
        t0 = time.time()
        ep = 0

        while tok_count < np.int64(total):
            loss, steps, tok_count, rng_state = _train_epoch(
                self.wi, self.wo, flat_ids, flat_hashes, flat_labels,
                input_offsets, label_offsets,
                np.int32(n_examples), np.int32(v.nlabels),
                np.int32(self.dim), np.int32(self.word_ngrams),
                np.int32(v.bucket), np.int32(v.nwords),
                np.float32(self.lr), np.int64(total),
                rng_state, tok_count)
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

        # clean up mmap temp files
        del flat_ids, flat_hashes, flat_labels, input_offsets, label_offsets
        for p in (ids_path, hash_path, lbl_path, ioff_path, loff_path):
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
    p = argparse.ArgumentParser(prog="supervised")
    sub = p.add_subparsers(dest="cmd")

    tr = sub.add_parser("train")
    tr.add_argument("corpus")
    tr.add_argument("-o", "--output", required=True)
    tr.add_argument("--dim",         type=int,   default=100)
    tr.add_argument("--epoch",       type=int,   default=5)
    tr.add_argument("--lr",          type=float, default=0.1)
    tr.add_argument("--min-count",   type=int,   default=1)
    tr.add_argument("--word-ngrams", type=int,   default=1)
    tr.add_argument("--bucket",      type=int,   default=2_000_000)
    tr.add_argument("--seed",        type=int,   default=0)

    ts = sub.add_parser("test")
    ts.add_argument("model")
    ts.add_argument("test_file")
    ts.add_argument("-k", type=int, default=1)

    pr = sub.add_parser("predict")
    pr.add_argument("model")
    pr.add_argument("-k", type=int, default=1)

    args = p.parse_args()
    if args.cmd == "train":
        m = Supervised.train(
            args.corpus, dim=args.dim, epoch=args.epoch, lr=args.lr,
            min_count=args.min_count, word_ngrams=args.word_ngrams,
            bucket=args.bucket, seed=args.seed)
        m.save(args.output)
    elif args.cmd == "test":
        m = Supervised.load(args.model)
        n, prec, rec = m.test(args.test_file, k=args.k)
        print(f"N\t{n}")
        print(f"P@{args.k}\t{prec:.4f}")
        print(f"R@{args.k}\t{rec:.4f}")
    elif args.cmd == "predict":
        m = Supervised.load(args.model)
        for line in sys.stdin:
            preds = m.predict(line.strip(), k=args.k)
            print(" ".join(f"{label} {prob:.4f}" for label, prob in preds))
    else:
        p.print_help()


if __name__ == "__main__":
    _cli()
