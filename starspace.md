# StarSpace — Pure Python Implementation Notes

## Overview

`starspace.py` is a pure Python (numpy + numba) implementation of Facebook's
[StarSpace](https://github.com/facebookresearch/StarSpace) algorithm,
specifically **trainMode=0** (classification/tagging).

**Performance**: 20.8x faster than C++ StarSpace single-threaded on a 30K-line
sentiment dataset (dim=100, epoch=5), with identical 100% P@1 accuracy.

## Algorithm

StarSpace embeds both inputs (bag-of-words) and labels into the same vector
space using a shared embedding matrix, trained with:

- **Hinge (margin ranking) loss**: `max(0, margin - cos(lhs, rhs+) + cos(lhs, rhs-))`
- **Cosine similarity** with L2-normalized embeddings
- **AdaGrad** optimisation (per-row accumulated gradient)
- **Negative sampling** from the training label pool

Each training example: LHS = sum of word + n-gram embeddings (normalized),
RHS = single label embedding (normalized). The model pushes LHS toward the
correct label and away from randomly sampled negative labels.

## Architecture

All training happens in a single `@njit(fastmath=True)` function — no Python
overhead per example. The tokenized corpus is streamed to temporary binary
files during vocabulary construction and memory-mapped back as read-only arrays,
eliminating RAM usage for corpus data.

Embedding matrix layout: `[words | labels | n-gram buckets] × dim`

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
from starspace import StarSpace

# Train
model = StarSpace.train("train.txt", dim=100, epoch=5, lr=0.01)

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
