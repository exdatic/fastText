"""Benchmark: Python (numba+mmap) sent2vec vs C++ sent2vec (single-threaded).

Usage:
    python sent2vec_benchmark.py                          # auto-downloads NLTK corpora
    python sent2vec_benchmark.py corpus.txt
    python sent2vec_benchmark.py corpus.txt --dim 200 --word-ngrams 2

Requires the C++ binary at ./fasttext (build with `make opt`) for comparison.
"""

import argparse, os, subprocess, sys, time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sent2vec import Sent2Vec

_DEFAULT_CORPUS = "/tmp/s2v_nltk_corpus.txt"


def _cosine(a, b):
    d = np.dot(a, b)
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return d / n if n > 0 else 0.0


def _download_corpus(path):
    """Download Gutenberg + Brown + Reuters from NLTK as a single text file."""
    print("Downloading NLTK corpora (gutenberg, brown, reuters)...")
    import nltk
    for pkg in ("gutenberg", "brown", "reuters", "punkt_tab"):
        nltk.download(pkg, quiet=True)
    from nltk.corpus import gutenberg, brown, reuters

    n_lines = 0
    with open(path, "w") as f:
        for corpus in (gutenberg, brown, reuters):
            for sent in corpus.sents():
                f.write(" ".join(sent) + "\n")
                n_lines += 1
    size_mb = os.path.getsize(path) / 1e6
    print(f"  Saved {path} ({size_mb:.1f} MB, {n_lines:,} sentences)\n")
    return path


def _analogy(model, a, b, c, top_n=5):
    """Solve a:b :: c:? by vector arithmetic."""
    va = model.word_vector(a)
    vb = model.word_vector(b)
    vc = model.word_vector(c)
    if not (np.linalg.norm(va) and np.linalg.norm(vb) and np.linalg.norm(vc)):
        return []
    target = vb - va + vc
    tn = np.linalg.norm(target)
    if tn == 0:
        return []
    target /= tn

    exclude = {a, b, c}
    results = []
    for w in model.vocab.words:
        if w in exclude or w.startswith("<"):
            continue
        wv = model.word_vector(w)
        wn = np.linalg.norm(wv)
        if wn == 0:
            continue
        results.append((w, np.dot(target, wv / wn)))
    results.sort(key=lambda x: -x[1])
    return results[:top_n]


def run(corpus, *, dim, epoch, lr, neg, min_count, word_ngrams, dropout_k,
        seed, runs, fasttext_bin):

    params = dict(dim=dim, epoch=epoch, lr=lr, neg=neg, min_count=min_count,
                  word_ngrams=word_ngrams, dropout_k=dropout_k, seed=seed)

    corpus_mb = os.path.getsize(corpus) / 1e6
    wc = sum(len(line.split()) for line in open(corpus))

    # ── warm JIT cache ────────────────────────────────────────────────────
    print("Warming JIT cache...", end="", flush=True)
    Sent2Vec.train(corpus, **params, verbose=0)
    print(" done\n")

    # ── Python runs ───────────────────────────────────────────────────────
    py_times = []
    py_model = None
    for i in range(runs):
        t0 = time.perf_counter()
        py_model = Sent2Vec.train(corpus, **params, verbose=0)
        py_times.append(time.perf_counter() - t0)

    total_tok = epoch * py_model.vocab.ntokens
    py_best = min(py_times)
    py_wps = total_tok / py_best

    # ── C++ runs ──────────────────────────────────────────────────────────
    cpp_times = []
    has_cpp = os.path.isfile(fasttext_bin)
    if has_cpp:
        for i in range(runs):
            t0 = time.perf_counter()
            subprocess.run([
                fasttext_bin, "sent2vec",
                "-input", corpus, "-output", "/tmp/s2v_bench_cpp",
                "-dim", str(dim), "-epoch", str(epoch), "-lr", str(lr),
                "-neg", str(neg), "-minCount", str(min_count),
                "-wordNgrams", str(word_ngrams), "-dropoutK", str(dropout_k),
                "-thread", "1",
            ], capture_output=True)
            cpp_times.append(time.perf_counter() - t0)
        cpp_best = min(cpp_times)
        cpp_wps = total_tok / cpp_best
    else:
        cpp_best = cpp_wps = 0

    # ── report ────────────────────────────────────────────────────────────
    print("=" * 65)
    print(f"{'SENT2VEC BENCHMARK':^65}")
    print(f"{'─' * 65}")
    print(f"  Corpus:  {corpus}  ({corpus_mb:.1f} MB, {wc:,} words)")
    print(f"  Params:  dim={dim} epoch={epoch} lr={lr} neg={neg} "
          f"minCount={min_count} wordNgrams={word_ngrams} dropoutK={dropout_k}")
    print(f"  Vocab:   {len(py_model.vocab)} words, "
          f"ntokens={py_model.vocab.ntokens:,}")
    print(f"  Runs:    {runs} (best of)")
    print(f"{'=' * 65}\n")

    hdr_py = "Python (numba+mmap)"
    hdr_cpp = "C++ (1 thread)"
    print(f"  {'':22s} {hdr_py:>18s}  {hdr_cpp:>18s}")
    print(f"  {'─' * 22} {'─' * 18}  {'─' * 18}")
    print(f"  {'Wall time':22s} {py_best:>17.2f}s", end="")
    if has_cpp:
        print(f"  {cpp_best:>17.2f}s")
    else:
        print(f"  {'(no binary)':>18s}")
    print(f"  {'Throughput':22s} {py_wps:>15,.0f} w/s", end="")
    if has_cpp:
        print(f"  {cpp_wps:>15,.0f} w/s")
    else:
        print(f"  {'':>18s}")
    if has_cpp and cpp_wps > 0:
        print(f"  {'Speed ratio':22s} {py_wps / cpp_wps:>17.2f}x  "
              f"{'1.0x (baseline)':>18s}")
    print(f"\n  {'Run times (Python)':22s} "
          f"{', '.join(f'{t:.3f}s' for t in py_times)}")
    if has_cpp:
        print(f"  {'Run times (C++)':22s} "
              f"{', '.join(f'{t:.3f}s' for t in cpp_times)}")

    # ── memory ────────────────────────────────────────────────────────────
    wi_mb = py_model.wi.nbytes / 1e6
    wo_mb = py_model.wo.nbytes / 1e6
    print(f"\n  Memory:  wi={wi_mb:.1f} MB  wo={wo_mb:.1f} MB  "
          f"(corpus mmap'd, not in RAM)")

    # ── quality: word similarities ────────────────────────────────────────
    pairs = [("man", "woman"), ("king", "queen"), ("good", "bad"),
             ("city", "town"), ("dog", "cat"), ("war", "peace")]
    shown = []
    for w1, w2 in pairs:
        v1, v2 = py_model.word_vector(w1), py_model.word_vector(w2)
        if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
            shown.append((w1, w2, _cosine(v1, v2)))
    if shown:
        print(f"\n  Word similarities (cosine):")
        for w1, w2, sim in shown:
            print(f"    {w1:>10s} — {w2:<10s}  {sim:.4f}")

    # ── quality: analogies ────────────────────────────────────────────────
    analogies = [
        ("man", "woman", "king"),       # king - man + woman ≈ queen
        ("paris", "france", "london"),   # london - paris + france ≈ england
        ("good", "better", "bad"),       # bad - good + better ≈ worse
    ]
    any_shown = False
    for a, b, c in analogies:
        results = _analogy(py_model, a, b, c)
        if results:
            if not any_shown:
                print(f"\n  Analogies (a:b :: c:?):")
                any_shown = True
            top = ", ".join(f"{w} ({s:.3f})" for w, s in results[:3])
            print(f"    {a}:{b} :: {c}:?  →  {top}")

    print("=" * 65)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("corpus", nargs="?", default=None,
                   help="Training corpus (one sentence per line). "
                        "If omitted, downloads NLTK corpora automatically.")
    p.add_argument("--dim", type=int, default=100)
    p.add_argument("--epoch", type=int, default=5)
    p.add_argument("--lr", type=float, default=0.2)
    p.add_argument("--neg", type=int, default=10)
    p.add_argument("--min-count", type=int, default=5)
    p.add_argument("--word-ngrams", type=int, default=1)
    p.add_argument("--dropout-k", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--runs", type=int, default=3, help="Number of runs (best of)")
    p.add_argument("--fasttext", default="./fasttext",
                   help="Path to C++ fasttext binary (default: ./fasttext)")
    args = p.parse_args()

    corpus = args.corpus
    if corpus is None:
        if not os.path.isfile(_DEFAULT_CORPUS):
            corpus = _download_corpus(_DEFAULT_CORPUS)
        else:
            corpus = _DEFAULT_CORPUS
            print(f"Using cached corpus: {corpus}\n")

    run(corpus, dim=args.dim, epoch=args.epoch, lr=args.lr,
        neg=args.neg, min_count=args.min_count, word_ngrams=args.word_ngrams,
        dropout_k=args.dropout_k, seed=args.seed, runs=args.runs,
        fasttext_bin=args.fasttext)


if __name__ == "__main__":
    main()
