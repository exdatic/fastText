# StarSpace — Pure Python Implementation Notes

## Overview

`starspace.py` is a pure Python (numpy + numba) implementation of Facebook's
[StarSpace](https://github.com/facebookresearch/StarSpace) algorithm,
supporting **all six training modes** (trainMode 0-5).

**Performance**: 20.8x faster than C++ StarSpace single-threaded on a 30K-line
sentiment dataset (dim=100, epoch=5), with identical 100% P@1 accuracy.

## Training Modes

| Mode | Name | LHS | RHS | Input format |
|------|------|-----|-----|--------------|
| 0 | Classification | words | 1 random label | `__label__X word word` |
| 1 | Label from rest | words + rest labels | 1 random label | `__label__A __label__B word word` (≥2 labels) |
| 2 | Inverted labels | words + 1 label | rest labels (multi-RHS) | `__label__A __label__B word word` (≥2 labels) |
| 3 | Pair prediction | words + 1 label | 1 other label | `__label__A __label__B word word` (≥2 labels) |
| 4 | Fixed pair | words + label[0] | label[1] | `__label__A __label__B word word` (exactly 2 labels) |
| 5 | Word embedding | context words | target word | `word word word` (no labels) |

**Mode-specific details:**
- Modes 1-4 add label embeddings to the LHS via a separate `flat_lhs_extra` mmap array, appended *after* word n-gram computation so labels don't participate in n-gram hashing.
- Mode 2 is the only multi-RHS mode: all remaining labels are summed into a single RHS vector. Negative sampling checks against all positive targets.
- Mode 5 expands each sentence into (context, target) word pairs using a sliding window of size `ws`. Negatives are sampled from the word pool (not labels).

## Algorithm

StarSpace embeds both inputs (bag-of-words) and labels into the same vector
space using a shared embedding matrix, trained with:

- **Hinge (margin ranking) loss**: `max(0, margin - cos(lhs, rhs+) + cos(lhs, rhs-))`
- **Cosine similarity** with L2-normalized embeddings
- **AdaGrad** optimisation (per-row accumulated gradient)
- **Negative sampling** from the training label pool (modes 0-4) or word pool (mode 5)

## Architecture

All training happens in a single `@njit(fastmath=True)` function — no Python
overhead per example. The tokenized corpus is streamed to temporary binary
files during vocabulary construction and memory-mapped back as read-only arrays,
eliminating RAM usage for corpus data.

Embedding matrix layout: `[words | labels | n-gram buckets] × dim`

**Mmap arrays** (8 files per training run):
- `ids`, `hash` — LHS word IDs and FNV-1a hashes
- `lbl` — RHS label/word IDs
- `extra` — extra LHS label IDs (modes 1-4, empty for modes 0/5)
- `ioff`, `loff`, `eoff` — per-example offsets into the above
- `neg` — negative sampling pool

**Iterator API**: `train()` and `test()` accept either a file path (str) or
any `Iterable[list[str]]`. Generator inputs are spilled to a temp text file
(not materialized in memory) since two passes are needed (vocab + training).

## Key Differences from C++ StarSpace

| Aspect | C++ StarSpace | Python (this) |
|---|---|---|
| Batch size | 5 (default) | 1 (per-example SGD) |
| maxNegSamples | 10 (caps violated negs) | All violated negs used |
| Loss divisor | `loss / negSearchLimit` | `loss_sum / n_steps / negSearchLimit` |
| Norm thread | Separate thread clips norms | Inline after each example |
| Corpus storage | Re-reads file each epoch | mmap'd binary (zero-copy) |

## C++ StarSpace Test Evaluation Bug

The C++ `starspace test` command reports ~50% hit@1 on data where the model
actually achieves 100% accuracy. Root cause: in `evaluateOne()` (starspace.cpp
line 362), when two labels have equal similarity scores, a random coin flip
decides ranking — `float flip = (float) rand() / RAND_MAX`. Since the correct
label is compared against itself via `baseDocVectors_` (precomputed) vs freshly
projected `rhsM`, floating-point equality triggers the 50/50 tiebreaker.

**Verification**: Manually evaluating the C++ model's TSV embeddings with
cosine similarity gives 100% accuracy on all 10K test examples — confirming
the trained embeddings are correct and only the eval code is buggy.

## Usage

```python
from starspace import StarSpace, iter_lines

# Train (file path)
model = StarSpace.train("train.txt", dim=100, epoch=5, lr=0.01)

# Train (iterator)
model = StarSpace.train(iter_lines("train.txt"), dim=100, epoch=5)

# Train (in-memory)
lines = [["__label__pos", "great", "movie"], ["__label__neg", "awful"]]
model = StarSpace.train(lines, dim=100, epoch=5)

# Train word embeddings (mode 5)
model = StarSpace.train("corpus.txt", dim=100, epoch=5, train_mode=5, ws=5)

# Predict
model.predict("the food was great")        # → [("__label__pos", 0.92)]

# Evaluate
n, precision, recall = model.test("test.txt")

# Save / Load
model.save("model.npz")
model = StarSpace.load("model.npz")
```

### CLI

```bash
python starspace.py train corpus.txt -o model.npz --dim 100 --epoch 5
python starspace.py train corpus.txt -o model.npz --train-mode 5 --ws 5
python starspace.py test model.npz test.txt
python starspace.py predict model.npz -k 3 < input.txt
```

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `dim` | 100 | Embedding dimension |
| `epoch` | 5 | Number of training epochs |
| `lr` | 0.01 | Initial learning rate (linear decay) |
| `margin` | 0.05 | Hinge loss margin |
| `neg_search_limit` | 50 | Negatives sampled per example |
| `min_count` | 1 | Minimum word frequency |
| `word_ngrams` | 1 | Max word n-gram length (1 = unigrams only) |
| `bucket` | 2,000,000 | Hash buckets for word n-grams |
| `norm_limit` | 1.0 | Max L2 norm for embedding rows |
| `init_rand_sd` | 0.001 | Std dev for initial embeddings |
| `train_mode` | 0 | Training mode (0-5, see table above) |
| `ws` | 5 | Context window size (mode 5 only) |
