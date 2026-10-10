import numpy as np
from src import config as cfg
from src.ablation import ALL_VARS, CORE5, SCENE, columns_for, lookup_predict
from src.features import ALL_FEATURES, add_features, build_preprocessor


def test_full_variable_set_reproduces_all_features():
    num, flag, cat = columns_for(ALL_VARS)
    assert set(num + flag + cat) == set(ALL_FEATURES)


def test_subsets_do_not_leak_excluded_variables():
    num, flag, cat = columns_for(SCENE)
    assert "n_missing_fields" not in num and "n_crash_events" not in num
    assert not set(cfg.POST_CRASH_COLS) & set(cat)
    num, flag, cat = columns_for(CORE5)
    assert "driver_unknown" not in flag and "age_of_oldest_driver" not in num


def test_preprocessor_accepts_subset(labeled_df):
    df = add_features(labeled_df)
    num, flag, cat = columns_for(CORE5)
    Xt = build_preprocessor("tree", num, flag, cat).fit_transform(df)
    assert Xt.shape == (len(df), len(num) + len(flag) + len(cat))


def test_lookup_predict_majority_and_fallback(labeled_df):
    y = labeled_df[cfg.TARGET].map({c: i for i, c in enumerate(cfg.CLASS_ORDER)}).to_numpy()
    pred, counts = lookup_predict(labeled_df, y, labeled_df, ["vehicle_type"])
    assert set(pred) <= {0, 1, 2} and len(counts) == labeled_df["vehicle_type"].nunique()
