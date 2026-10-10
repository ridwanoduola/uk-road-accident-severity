"""Loading, auditing and cleaning the raw accident data.

Cleaning rules (derived from reports/value_audit.txt):
  * accident_index is dropped (corrupted by Excel scientific notation).
  * Text columns are stripped and lowercased  -> recovers 'Serious'/'Slight'/'Fatal' variants.
  * Blank accident_severity (target unknown) -> set aside as `unlabeled`, NOT silently deleted.
  * speed_limit == -1 is a missing-value code -> NaN + flag (row kept).
  * age blank -> NaN + flag (row kept).
  * 'data missing or out of range' stays as its own category (informative, row kept).
  * Exact duplicate RECORDS (identical on every column including accident_index) are removed.
    The audit showed rows sharing an ID are always fully identical, so the ID behaves as a
    genuine unique key. Rows that merely share the same FEATURE values are different accidents
    and are kept (see extra_checks for how this affects evaluation).
  * Age < 16 combined with a motor-only vehicle_type ('only cars', 'at least one van') is
    implausible -> NaN. Ages < 16 involving bipeds (e.g. child cyclists) are kept.
Imputation happens later, inside the model pipeline, to avoid leakage.
"""
import pandas as pd
from . import config as cfg


def load_raw(path=cfg.RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={cfg.ID_COL: "string"})
    df.columns = df.columns.str.strip().str.lower()
    return df


def audit(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in df.columns:
        s = df[c]
        rows.append({
            "column": c,
            "dtype": str(s.dtype),
            "n_null": int(s.isna().sum()),
            "n_missing_label": int((s.astype("string") == cfg.MISSING_LABEL).sum()),
            "n_unique": int(s.nunique(dropna=True)),
        })
    return pd.DataFrame(rows)


def clean(df: pd.DataFrame):
    """Return (labeled, unlabeled, log). Nothing is dropped without being counted in `log`."""
    out = df.copy()
    log = {"n_input": len(out)}

    # --- exact duplicate records (must happen BEFORE the ID is dropped)
    if cfg.ID_COL in out.columns:
        dup = out.duplicated(keep="first")
        id_shared = out[cfg.ID_COL].duplicated(keep=False).sum()
        log["id_shared_but_not_identical_rows"] = int(id_shared - out.duplicated(keep=False).sum())
        log["exact_duplicate_records_removed"] = int(dup.sum())
        tgt = out[cfg.TARGET].astype("string").str.strip()
        log["  of which had blank target"] = int((dup & (tgt.isna() | (tgt == ""))).sum())
        out = out[~dup].drop(columns=[cfg.ID_COL])

    # --- target: standardise text, separate rows with no label
    raw_target = out[cfg.TARGET].astype("string")
    stripped = raw_target.str.strip()
    log["target_case_variants_fixed"] = int(
        ((stripped != stripped.str.lower()) & stripped.notna() & (stripped != "")).sum()
    )
    out[cfg.TARGET] = stripped.str.lower().replace("", pd.NA)

    # --- categorical features
    for c in cfg.CATEGORICAL_COLS:
        out[c] = out[c].astype("string").str.strip().str.lower()

    # --- numeric features
    out["speed_limit"] = pd.to_numeric(out["speed_limit"], errors="coerce")
    log["speed_limit_minus1_to_nan"] = int((out["speed_limit"] == -1).sum())
    out.loc[out["speed_limit"] == -1, "speed_limit"] = pd.NA
    out["speed_missing"] = out["speed_limit"].isna().astype(int)

    out["age_of_oldest_driver"] = pd.to_numeric(out["age_of_oldest_driver"], errors="coerce")
    age = out["age_of_oldest_driver"]
    log["age_blank"] = int(age.isna().sum())
    out["age_unusual"] = ((age < 16) | (age > 100)).astype(int)  # flag kept for transparency
    log["age_unusual_flagged"] = int(out["age_unusual"].sum())
    implausible = (age < 16) & out["vehicle_type"].isin(["only cars", "at least one van"])
    log["age_under16_motor_only_set_to_nan"] = int(implausible.sum())
    out.loc[implausible, "age_of_oldest_driver"] = pd.NA
    out["age_missing"] = out["age_of_oldest_driver"].isna().astype(int)

    # --- split on target
    bad_label = out[cfg.TARGET].notna() & ~out[cfg.TARGET].isin(cfg.CLASS_ORDER)
    log["target_unrecognised_label"] = int(bad_label.sum())
    unlabeled = out[out[cfg.TARGET].isna() | bad_label].reset_index(drop=True)
    labeled = out[out[cfg.TARGET].isin(cfg.CLASS_ORDER)].reset_index(drop=True)
    log["n_labeled"] = len(labeled)
    log["n_unlabeled_set_aside"] = len(unlabeled)
    return labeled, unlabeled, log


if __name__ == "__main__":
    raw = load_raw()
    labeled, unlabeled, log = clean(raw)
    print("CLEANING LOG")
    for k, v in log.items():
        print(f"  {k}: {v:,}")
    cfg.CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    labeled.to_csv(cfg.CLEAN_PATH, index=False)
    unlabeled.to_csv(cfg.CLEAN_PATH.with_name("accidents_unlabeled.csv"), index=False)
    print("\nClass distribution (labeled):")
    print(labeled[cfg.TARGET].value_counts().to_frame("n").assign(pct=lambda d: (d.n / d.n.sum() * 100).round(1)))
