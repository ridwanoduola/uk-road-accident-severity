import numpy as np
import pandas as pd
from src.ablation import CORE5, columns_for
from src.explain import shapley_values
from src.features import add_features, core_features


def test_core_features_match_full_feature_pipeline(labeled_df):
    full = add_features(labeled_df)
    core = core_features(labeled_df)
    num, flag, cat = columns_for(CORE5)
    for c in num + flag + cat:
        a, b = full[c], core[c]
        if a.dtype == object:
            assert (a.astype(str) == b.astype(str)).all(), c
        else:
            assert np.allclose(a.astype(float), b.astype(float), equal_nan=True), c


def _toy(df):                     # depends on a, b (interaction) but not on c
    return 1 / (1 + np.exp(-(1.5 * df["a"] + 0.8 * df["b"] + 1.2 * df["a"] * df["b"] - 1)))


def test_shapley_efficiency_and_dummy_variable():
    rng = np.random.default_rng(0)
    bg = pd.DataFrame({"a": rng.integers(0, 2, 200), "b": rng.integers(0, 2, 200), "c": rng.integers(0, 2, 200)})
    x = {"a": 1, "b": 1, "c": 0}
    phi, base, pred = shapley_values(_toy, x, bg, ["a", "b", "c"])
    assert abs(sum(phi.values()) - (pred - base)) < 1e-9          # efficiency
    assert abs(phi["c"]) < 1e-9                                    # unused variable gets zero credit
    assert phi["a"] > 0 and phi["b"] > 0
