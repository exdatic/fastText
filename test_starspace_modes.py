"""Test all 6 StarSpace training modes (0-5)."""

import sys, os, tempfile, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from starspace import StarSpace, iter_lines

# ── test data ──────────────────────────────────────────────────────────────────

# Classification data: __label__X  word word word
CLASSIFICATION_TRAIN = [
    ["__label__pos", "love", "this", "great", "movie"],
    ["__label__pos", "excellent", "wonderful", "film"],
    ["__label__pos", "amazing", "fantastic", "great"],
    ["__label__pos", "best", "movie", "ever", "love"],
    ["__label__pos", "brilliant", "love", "wonderful"],
    ["__label__neg", "terrible", "awful", "bad", "movie"],
    ["__label__neg", "worst", "horrible", "boring"],
    ["__label__neg", "hate", "this", "awful", "film"],
    ["__label__neg", "bad", "terrible", "waste"],
    ["__label__neg", "boring", "dull", "hate"],
]

CLASSIFICATION_TEST = [
    ["__label__pos", "love", "great", "film"],
    ["__label__neg", "terrible", "bad", "boring"],
    ["__label__pos", "wonderful", "amazing"],
    ["__label__neg", "awful", "hate", "waste"],
]

# Multi-label data (modes 1-4): items have 2+ labels
MULTILABEL_TRAIN = [
    ["__label__action", "__label__comedy", "fun", "exciting", "chase"],
    ["__label__drama", "__label__romance", "love", "emotional", "tears"],
    ["__label__action", "__label__scifi", "space", "battle", "hero"],
    ["__label__comedy", "__label__romance", "funny", "love", "date"],
    ["__label__drama", "__label__action", "intense", "fight", "war"],
    ["__label__scifi", "__label__drama", "future", "dark", "robot"],
    ["__label__comedy", "__label__action", "fun", "chase", "silly"],
    ["__label__romance", "__label__drama", "tears", "love", "sad"],
    ["__label__action", "__label__comedy", "exciting", "fun", "fight"],
    ["__label__drama", "__label__romance", "emotional", "love", "heart"],
    ["__label__scifi", "__label__action", "battle", "space", "laser"],
    ["__label__comedy", "__label__romance", "funny", "date", "love"],
] * 5  # Repeat for more data

# Word embedding data (mode 5): plain text, no labels
WORD_EMB_TRAIN = [
    "the cat sat on the mat".split(),
    "the dog ran in the park".split(),
    "a cat and a dog played together".split(),
    "the bird flew over the house".split(),
    "a man walked with his dog".split(),
    "the woman sat in the park".split(),
    "cats and dogs are friends".split(),
    "the boy ran to the house".split(),
    "she walked with her cat".split(),
    "he sat on the bench in the park".split(),
] * 10  # Repeat for more data


def test_mode(mode, train_data, test_data=None, *, description, dim=30,
              epoch=10, extra_kwargs=None):
    """Test a single training mode and return pass/fail."""
    kwargs = dict(dim=dim, epoch=epoch, lr=0.01, seed=42, verbose=0,
                  train_mode=mode, min_count=1)
    if extra_kwargs:
        kwargs.update(extra_kwargs)

    print(f"\n{'='*60}")
    print(f"  Mode {mode}: {description}")
    print(f"{'='*60}")

    t0 = time.time()
    try:
        model = StarSpace.train(train_data, **kwargs)
    except Exception as e:
        print(f"  FAIL: Training raised {type(e).__name__}: {e}")
        return False

    elapsed = time.time() - t0
    print(f"  Trained in {elapsed:.2f}s")
    print(f"  Embedding shape: {model.emb.shape}")
    print(f"  Vocab: {model.vocab.nwords} words, {model.vocab.nlabels} labels")

    # Test save/load roundtrip
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tf:
        tmp_path = tf.name
    try:
        model.save(tmp_path)
        loaded = StarSpace.load(tmp_path)
        assert loaded.train_mode == mode, \
            f"train_mode mismatch: {loaded.train_mode} != {mode}"
        assert loaded.emb.shape == model.emb.shape, \
            f"emb shape mismatch: {loaded.emb.shape} != {model.emb.shape}"
        print(f"  Save/load roundtrip: OK")
    except Exception as e:
        print(f"  FAIL: Save/load raised {type(e).__name__}: {e}")
        return False
    finally:
        os.unlink(tmp_path)

    # Test prediction (modes 0-4 have labels)
    if mode < 5 and model.vocab.nlabels > 0:
        try:
            preds = model.predict("love great film", k=3)
            print(f"  predict('love great film'): {preds}")
        except Exception as e:
            print(f"  FAIL: predict raised {type(e).__name__}: {e}")
            return False

    # Test evaluation (modes 0-4)
    if test_data is not None and mode < 5:
        try:
            n, prec, rec = model.test(test_data)
            print(f"  test(): N={n}, P@1={prec:.4f}, R@1={rec:.4f}")
        except Exception as e:
            print(f"  FAIL: test raised {type(e).__name__}: {e}")
            return False

    # Mode 5: test() should raise
    if mode == 5:
        try:
            model.test([["__label__x", "word"]])
            print(f"  FAIL: test() should raise ValueError for mode 5")
            return False
        except ValueError:
            print(f"  test() correctly raises ValueError for mode 5")

    print(f"  PASS")
    return True


def test_file_input(mode=0):
    """Test that file path input works."""
    print(f"\n{'='*60}")
    print(f"  File path input test (mode 0)")
    print(f"{'='*60}")

    with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False) as tf:
        for tokens in CLASSIFICATION_TRAIN:
            tf.write(" ".join(tokens) + "\n")
        train_path = tf.name

    try:
        model = StarSpace.train(train_path, dim=30, epoch=5, lr=0.01,
                                seed=42, verbose=0, min_count=1)
        preds = model.predict("love great", k=2)
        print(f"  predict('love great'): {preds}")
        assert len(preds) > 0
        print(f"  PASS")
        return True
    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {e}")
        return False
    finally:
        os.unlink(train_path)


def test_iter_lines_input():
    """Test that iter_lines works as input."""
    print(f"\n{'='*60}")
    print(f"  iter_lines() input test")
    print(f"{'='*60}")

    with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False) as tf:
        for tokens in CLASSIFICATION_TRAIN:
            tf.write(" ".join(tokens) + "\n")
        train_path = tf.name

    try:
        model = StarSpace.train(iter_lines(train_path), dim=30, epoch=5,
                                lr=0.01, seed=42, verbose=0, min_count=1)
        preds = model.predict("love great", k=2)
        print(f"  predict('love great'): {preds}")
        assert len(preds) > 0
        print(f"  PASS")
        return True
    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {e}")
        return False
    finally:
        os.unlink(train_path)


def main():
    print("StarSpace All-Modes Test Suite")
    print("=" * 60)

    results = {}

    # Warm numba JIT
    print("\nWarming numba JIT (first compile)...")
    t0 = time.time()
    StarSpace.train(CLASSIFICATION_TRAIN[:4], dim=10, epoch=1,
                    lr=0.01, seed=0, verbose=0, min_count=1)
    print(f"JIT warm-up done in {time.time()-t0:.1f}s\n")

    # Mode 0: Classification
    results[0] = test_mode(0, CLASSIFICATION_TRAIN, CLASSIFICATION_TEST,
                           description="Classification (LHS=words, RHS=1 label)")

    # Mode 1: Label from rest
    results[1] = test_mode(1, MULTILABEL_TRAIN,
                           description="Label from rest (LHS=words+rest, RHS=1)")

    # Mode 2: Inverted labels (multi-RHS)
    results[2] = test_mode(2, MULTILABEL_TRAIN,
                           description="Inverted labels (LHS=words+1, RHS=rest)")

    # Mode 3: Pair prediction
    results[3] = test_mode(3, MULTILABEL_TRAIN,
                           description="Pair prediction (LHS=words+1, RHS=1 other)")

    # Mode 4: Fixed pair
    results[4] = test_mode(4, MULTILABEL_TRAIN,
                           description="Fixed pair (LHS=words+label[0], RHS=label[1])")

    # Mode 5: Word embedding
    results[5] = test_mode(5, WORD_EMB_TRAIN,
                           description="Word embedding (context→target)",
                           extra_kwargs=dict(ws=3))

    # File input
    results["file"] = test_file_input()

    # iter_lines input
    results["iter"] = test_iter_lines_input()

    # Summary
    print(f"\n\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    all_pass = True
    for key, passed in results.items():
        status = "PASS" if passed else "FAIL"
        label = f"Mode {key}" if isinstance(key, int) else key
        print(f"  {label:20s} {status}")
        if not passed:
            all_pass = False

    print(f"{'='*60}")
    if all_pass:
        print("  ALL TESTS PASSED")
    else:
        print("  SOME TESTS FAILED")
    print(f"{'='*60}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
