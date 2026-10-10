import numpy as np
from src import config as cfg
from src.features import ALL_FEATURES, add_features, build_preprocessor
from src.split import make_split, profile_key


def test_add_features_values(labeled_df):
    df = add_features(labeled_df)
    assert set(df["is_dark"].unique()) <= {0, 1}
    assert (df["n_crash_events"].between(0, 3)).all()
    miss = (df[cfg.CATEGORICAL_COLS] == cfg.MISSING_LABEL).sum(axis=1)
    assert (df["n_missing_fields"] == miss).all()
    assert (df.loc[df["driver_unknown"] == 1, "sex_of_driver"] == cfg.MISSING_LABEL).all()
    assert (df.loc[df["driver_unknown"] == 1, "age_missing"] == 1).all()


def test_preprocessors_output_clean_matrix(labeled_df):
    df = add_features(labeled_df)
    for kind in ["linear", "tree"]:
        Xt = build_preprocessor(kind).fit_transform(df[ALL_FEATURES])
        assert not np.isnan(Xt).any()
        assert Xt.shape[0] == len(df)


def test_unseen_category_does_not_crash(labeled_df):
    df = add_features(labeled_df)
    prep = build_preprocessor("linear").fit(df[ALL_FEATURES])
    new = df[ALL_FEATURES].head(3).copy()
    new["vehicle_type"] = "something never seen"
    assert prep.transform(new).shape[0] == 3


def test_random_split_disjoint_and_stratified(labeled_df):
    dev, test = make_split(labeled_df, mode="random")
    assert len(set(dev) & set(test)) == 0 and len(dev) + len(test) == len(labeled_df)
    full = labeled_df[cfg.TARGET].value_counts(normalize=True)
    part = labeled_df.iloc[test][cfg.TARGET].value_counts(normalize=True)
    assert (full - part).abs().max() < 0.05


def test_grouped_split_shares_no_profile(labeled_df):
    dev, test = make_split(labeled_df, mode="grouped")
    keys = profile_key(labeled_df)
    assert set(keys.iloc[dev]).isdisjoint(set(keys.iloc[test]))
