"""Follow-up checks raised by the value audit.  Run: python -m src.extra_checks
Saves reports/extra_checks.txt. Nothing is modified."""
import pandas as pd
from . import config as cfg
from .data import load_raw, clean


def main():
    raw = load_raw()
    labeled, unlabeled, log = clean(raw)
    feats = cfg.CATEGORICAL_COLS + ["speed_limit", "age_of_oldest_driver"]
    out = []
    p = out.append

    p("1) CLEANING LOG")
    for k, v in log.items():
        p(f"   {k}: {v:,}")

    p("\n2) CLASS DISTRIBUTION (labeled rows)")
    vc = labeled[cfg.TARGET].value_counts()
    p((vc.to_frame("n").assign(pct=(vc / vc.sum() * 100).round(1))).to_string())

    p("\n3) DUPLICATES (computed on the raw file)")
    p(f"   accident_index: distinct={raw[cfg.ID_COL].nunique():,} of {len(raw):,} rows; "
      f"rows sharing an ID={raw[cfg.ID_COL].duplicated(keep=False).sum():,}")
    p(f"   rows identical on ALL columns incl. ID: {raw.duplicated(keep=False).sum():,}")
    p("   -> if the two numbers match, shared IDs are always exact copies (ID is a reliable key)")

    p("\n4) FEATURE-PROFILE REPEATS AND LABEL NOISE (after removing exact duplicate records)")
    feats_df = labeled[feats].astype("string").fillna("NA")
    profile = feats_df.agg("|".join, axis=1)
    size = profile.map(profile.value_counts())
    rep = size >= 2
    p(f"   labeled rows: {len(labeled):,} | distinct feature profiles: {profile.nunique():,}")
    p(f"   rows whose profile appears at least twice: {int(rep.sum()):,} ({rep.mean():.1%}); "
      f"singleton-profile rows: {int((~rep).sum()):,}")
    onehot = pd.get_dummies(labeled[cfg.TARGET])
    order = labeled[cfg.TARGET].value_counts().index.tolist()   # ties go to the more frequent class
    onehot = onehot[order].astype(int)
    group_counts = onehot.groupby(profile).transform("sum")
    n_conf = (group_counts.gt(0).sum(axis=1) > 1)
    p(f"   rows in repeated profiles that contain conflicting labels: {int((n_conf & rep).sum()):,} "
      f"({(n_conf & rep).sum() / max(rep.sum(), 1):.1%} of repeated-profile rows)")
    if rep.sum() == 0:
        p("   no repeated profiles; leave-one-out check not applicable")
        rep = rep.copy()
    loo = (group_counts - onehot)[rep]                              # leave-one-out: other rows only
    pred = loo.values.argmax(axis=1)
    truth = onehot[rep].values.argmax(axis=1)
    acc = (pred == truth).mean() if len(truth) else float("nan")
    base = labeled[cfg.TARGET].value_counts(normalize=True).iloc[0]
    p(f"   leave-one-out majority-label accuracy on repeated-profile rows: {acc:.1%} "
      f"(always-predict-'{order[0]}' baseline on all rows: {base:.1%})")
    p("   NOTE: this is an optimistic-but-honest guide to how predictable severity is from these")
    p("   features. Singleton profiles cannot be assessed this way, so it is NOT a global ceiling.")

    p("\n5) FLAGGED UNUSUAL AGES (<16 or >100) by vehicle_type")
    odd = labeled[labeled["age_unusual"] == 1]
    p(pd.crosstab(odd["vehicle_type"], odd[cfg.TARGET]).to_string() if len(odd) else "   none")

    p("\n6) AGE BLANK vs SEX_OF_DRIVER")
    p(pd.crosstab(labeled["age_missing"].map({0: "age present", 1: "age blank"}), labeled["sex_of_driver"]).to_string())

    p("\n7) AGE BLANK vs VEHICLE_TYPE")
    p(pd.crosstab(labeled["vehicle_type"], labeled["age_missing"].map({0: "age present", 1: "age blank"})).to_string())

    p("\n8) SEVERITY BY MISSINGNESS (is 'missing' informative?)")
    for c in ["age_missing", "speed_missing"]:
        p(f"   {c}:")
        p((pd.crosstab(labeled[c], labeled[cfg.TARGET], normalize="index") * 100).round(1).to_string())
    n_missing_labels = (labeled[cfg.CATEGORICAL_COLS] == cfg.MISSING_LABEL).sum(axis=1)
    p("   number of 'data missing' labels per row vs severity (%):")
    p((pd.crosstab(n_missing_labels.clip(upper=4), labeled[cfg.TARGET], normalize="index") * 100).round(1).to_string())

    p("\n9) UNLABELED ROWS (set aside): do they look different?")
    if len(unlabeled):
        p(f"   speed_limit mean labeled={labeled['speed_limit'].mean():.1f} vs unlabeled={unlabeled['speed_limit'].mean():.1f}")
        p(f"   light_conditions darkness share labeled={(labeled['light_conditions']=='darkness').mean():.1%} "
          f"vs unlabeled={(unlabeled['light_conditions']=='darkness').mean():.1%}")

    text = "\n".join(out)
    print(text)
    (cfg.ROOT / "reports").mkdir(exist_ok=True)
    (cfg.ROOT / "reports" / "extra_checks.txt").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
