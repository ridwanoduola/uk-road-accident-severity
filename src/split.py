"""Phase 4: locked test set + development set.

Design: 85% development / 15% test. All model selection, tuning and cross-validation happen on the
development set only; the test set is touched once, at the very end.
  mode='random'  : stratified random split (primary; realistic because new accidents often resemble past ones)
  mode='grouped' : whole feature-profiles go to one side only (stricter robustness check)
Run:  python -m src.split
"""
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from . import config as cfg
from .data import load_raw, clean

PROFILE_COLS = cfg.CATEGORICAL_COLS + cfg.NUMERIC_COLS


def profile_key(df: pd.DataFrame) -> pd.Series:
    return df[PROFILE_COLS].astype("string").fillna("NA").agg("|".join, axis=1)


def make_split(df: pd.DataFrame, test_size: float = 0.15, mode: str = "random", seed: int = cfg.RANDOM_STATE):
    idx = np.arange(len(df))
    y = df[cfg.TARGET]
    if mode == "random":
        dev, test = train_test_split(idx, test_size=test_size, stratify=y, random_state=seed)
    elif mode == "grouped":
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        dev, test = next(gss.split(idx, y, groups=profile_key(df)))
    else:
        raise ValueError("mode must be 'random' or 'grouped'")
    return np.sort(dev), np.sort(test)


def main():
    labeled, _, _ = clean(load_raw())
    summary = {}
    for mode in ["random", "grouped"]:
        dev, test = make_split(labeled, mode=mode)
        shared = len(set(profile_key(labeled.iloc[dev])) & set(profile_key(labeled.iloc[test])))
        print(f"\n[{mode}] dev={len(dev):,}  test={len(test):,}  profiles shared between dev and test: {shared:,}")
        for name, ix in [("dev", dev), ("test", test)]:
            print(f"   {name} class mix %:", (labeled.iloc[ix][cfg.TARGET].value_counts(normalize=True) * 100).round(1).to_dict())
        summary[mode] = {"seed": cfg.RANDOM_STATE, "n_rows": len(labeled), "dev": dev.tolist(), "test": test.tolist()}
    cfg.CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    (cfg.CLEAN_PATH.parent / "splits.json").write_text(json.dumps(summary), encoding="utf-8")
    print("\nSaved row indices to data/processed/splits.json (indices refer to the cleaned labeled data).")


if __name__ == "__main__":
    main()
