import numpy as np
from sklearn.metrics import f1_score
from src.final_eval import bootstrap, ci, cm_from, macro_f1_from_cm


def test_fast_macro_f1_matches_sklearn():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 500)
    p = np.where(rng.random(500) < 0.7, y, rng.integers(0, 3, 500))
    assert abs(macro_f1_from_cm(cm_from(y, p)) - f1_score(y, p, average="macro")) < 1e-9


def test_bootstrap_interval_contains_point_estimate_and_pairs_models():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 3, 800)
    good = np.where(rng.random(800) < 0.8, y, rng.integers(0, 3, 800))
    f1, fatal = bootstrap(y, {"a": good, "b": good}, n_boot=300)
    lo, hi = ci(f1["a"])
    assert lo < macro_f1_from_cm(cm_from(y, good)) < hi
    assert np.allclose(f1["a"], f1["b"])           # identical models -> identical paired resamples
