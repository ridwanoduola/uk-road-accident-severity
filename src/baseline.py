"""Phase 4 sanity baselines (development set only; the test set is NOT touched).
Run:  python -m src.baseline      -> prints and saves reports/baseline_results.txt"""
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from . import config as cfg
from .data import load_raw, clean
from .features import ALL_FEATURES, add_features, build_preprocessor
from .split import make_split


def main():
    labeled, _, _ = clean(load_raw())
    df = add_features(labeled)
    dev_idx, _test_idx = make_split(df, mode="random")
    dev = df.iloc[dev_idx]
    X, y = dev[ALL_FEATURES], dev[cfg.TARGET]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.RANDOM_STATE)

    models = {
        "Majority class": DummyClassifier(strategy="most_frequent"),
        "Logistic regression (class_weight=balanced)": Pipeline([
            ("prep", build_preprocessor("linear")),
            ("clf", LogisticRegression(max_iter=3000, class_weight="balanced")),
        ]),
    }
    lines = [f"Development set: {len(dev):,} rows | 5-fold stratified CV | pooled out-of-fold predictions", ""]
    for name, model in models.items():
        pred = cross_val_predict(model, X, y, cv=cv, n_jobs=-1)
        lines += [
            "=" * 78, name, "=" * 78,
            f"accuracy={accuracy_score(y, pred):.3f}  macro-F1={f1_score(y, pred, average='macro', zero_division=0):.3f}  "
            f"balanced-accuracy={balanced_accuracy_score(y, pred):.3f}",
            classification_report(y, pred, labels=cfg.CLASS_ORDER, digits=3, zero_division=0),
            "confusion matrix (rows=true, cols=predicted; order: " + ", ".join(cfg.CLASS_ORDER) + ")",
            str(confusion_matrix(y, pred, labels=cfg.CLASS_ORDER)), "",
        ]
    text = "\n".join(lines)
    print(text)
    (cfg.ROOT / "reports").mkdir(exist_ok=True)
    (cfg.ROOT / "reports" / "baseline_results.txt").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
