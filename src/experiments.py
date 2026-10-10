"""Phases 5-6: class-imbalance handling + traditional ML model comparison (development set only).

Run:  python -m src.experiments              stratified random 5-fold CV
      python -m src.experiments --grouped    stricter: a feature profile never appears in two folds
The test set is NOT touched. Results are saved to reports/model_comparison_<mode>.txt / .csv
"""
import argparse
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
from . import config as cfg
from .data import load_raw, clean
from .features import ALL_FEATURES, add_features, build_preprocessor
from .split import make_split, profile_key

RS = cfg.RANDOM_STATE
LABELS = [0, 1, 2]  # slight, serious, fatal (cfg.CLASS_ORDER)


def get_models():
    """name -> (preprocessor kind, estimator, pass sample_weight to fit?)"""
    m = {
        "LogReg (no weights)": ("linear", LogisticRegression(max_iter=3000), False),
        "LogReg (balanced)": ("linear", LogisticRegression(max_iter=3000, class_weight="balanced"), False),
        "LogReg + interactions (balanced)": ("linear_int", LogisticRegression(max_iter=500, C=0.5, class_weight="balanced"), False),
        "Decision tree (balanced)": ("tree", DecisionTreeClassifier(max_depth=12, min_samples_leaf=10, class_weight="balanced", random_state=RS), False),
        "Random forest (no weights)": ("tree", RandomForestClassifier(n_estimators=300, min_samples_leaf=3, n_jobs=-1, random_state=RS), False),
        "Random forest (balanced)": ("tree", RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced_subsample", n_jobs=-1, random_state=RS), False),
        "HistGradientBoosting (balanced)": ("tree", HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, random_state=RS), True),
    }
    try:
        from xgboost import XGBClassifier
        m["XGBoost (balanced)"] = ("tree", XGBClassifier(n_estimators=400, learning_rate=0.08, max_depth=6, subsample=0.9,
                                   colsample_bytree=0.9, tree_method="hist", n_jobs=-1, random_state=RS), True)
    except ImportError:
        print("xgboost not installed -> skipping (pip install xgboost)")
    try:
        from lightgbm import LGBMClassifier
        m["LightGBM (balanced)"] = ("tree", LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63, subsample=0.9,
                                    subsample_freq=1, colsample_bytree=0.9, n_jobs=-1, random_state=RS, verbose=-1), True)
    except ImportError:
        print("lightgbm not installed -> skipping (pip install lightgbm)")
    return m


def make_pipeline(kind, est):
    if kind == "linear_int":
        return Pipeline([("prep", build_preprocessor("linear")),
                         ("poly", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
                         ("clf", clone(est))])
    return Pipeline([("prep", build_preprocessor(kind)), ("clf", clone(est))])


def fold_metrics(y, pred):
    p, r, _, _ = precision_recall_fscore_support(y, pred, labels=LABELS, zero_division=0)
    return {
        "accuracy": accuracy_score(y, pred),
        "macro_F1": f1_score(y, pred, average="macro", zero_division=0),
        "balanced_acc": balanced_accuracy_score(y, pred),
        "recall_slight": r[0], "recall_serious": r[1], "recall_fatal": r[2],
        "precision_fatal": p[2],
    }


def load_dev():
    """Cleaned, feature-engineered DEVELOPMENT set (test set excluded) -> (dev, X, y_int)."""
    labeled, _, _ = clean(load_raw())
    df = add_features(labeled)
    dev_idx, _ = make_split(df, mode="random")
    dev = df.iloc[dev_idx].reset_index(drop=True)
    X = dev[ALL_FEATURES]
    y = dev[cfg.TARGET].map({c: i for i, c in enumerate(cfg.CLASS_ORDER)}).to_numpy()
    return dev, X, y


def get_splits(dev, X, y, grouped=False):
    if grouped:
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RS)
        return list(cv.split(X, y, groups=profile_key(dev)))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RS)
    return list(cv.split(X, y))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grouped", action="store_true", help="group folds by feature profile (stricter)")
    ap.add_argument("--only", default=None, help="run only models whose name contains this text")
    args = ap.parse_args()
    mode = "grouped" if args.grouped else "random"

    dev, X, y = load_dev()
    splits = get_splits(dev, X, y, grouped=args.grouped)
    print(f"Development rows: {len(dev):,} | CV mode: {mode} | folds: {len(splits)}\n")

    rows = []
    for name, (kind, est, use_sw) in get_models().items():
        if args.only and args.only.lower() not in name.lower():
            continue
        t0 = time.time()
        for fold, (tr, va) in enumerate(splits):
            pipe = make_pipeline(kind, est)
            fit_kw = {}
            if use_sw:
                fit_kw["clf__sample_weight"] = compute_sample_weight("balanced", y[tr])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pipe.fit(X.iloc[tr], y[tr], **fit_kw)
                pred = pipe.predict(X.iloc[va])
            rows.append({"model": name, "fold": fold, **fold_metrics(y[va], pred)})
        print(f"  done: {name:<36} {time.time() - t0:6.1f}s")

    res = pd.DataFrame(rows)
    metrics = ["accuracy", "macro_F1", "balanced_acc", "recall_slight", "recall_serious", "recall_fatal", "precision_fatal"]
    g = res.groupby("model")[metrics]
    mean, std = g.mean(), g.std()
    table = pd.DataFrame({c: [f"{mean.loc[i, c]:.3f} ± {std.loc[i, c]:.3f}" for i in mean.index] for c in metrics}, index=mean.index)
    table = table.loc[mean.sort_values("macro_F1", ascending=False).index]

    text = (f"Model comparison | development set {len(dev):,} rows | 5-fold CV ({mode}) | mean ± std over folds\n\n"
            + table.to_string())
    print("\n" + text)
    (cfg.ROOT / "reports").mkdir(exist_ok=True)
    (cfg.ROOT / "reports" / f"model_comparison_{mode}.txt").write_text(text, encoding="utf-8")
    res.to_csv(cfg.ROOT / "reports" / f"model_comparison_{mode}.csv", index=False)


if __name__ == "__main__":
    main()
