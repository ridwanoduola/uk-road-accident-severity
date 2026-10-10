"""Phase 9a: interpretation + error analysis of the FINAL model. No refitting, no model selection.

Run:  python -m src.error_analysis        (needs models/hgb_final.joblib, created by src.final_eval)
Outputs: reports/error_analysis.txt, reports/figures/50_permutation_importance.png,
         reports/figures/51_accuracy_by_profile_type.png
"""
import joblib
import numpy as np
import pandas as pd
from . import config as cfg
from .data import load_raw, clean
from .features import ALL_FEATURES, add_features
from .final_eval import cm_from, macro_f1_from_cm
from .split import make_split, profile_key

CORE5 = ["speed_limit", "light_conditions", "vehicle_type", "weather_conditions", "road_surface_conditions"]


def _key(df, cols):
    return df[cols].astype("string").fillna("NA").agg("|".join, axis=1)

GROUPS = {"speed_limit": ["speed_limit", "speed_missing"],
          "age_of_oldest_driver": ["age_of_oldest_driver", "age_missing", "age_unusual"],
          **{c: [c] for c in cfg.CATEGORICAL_COLS}}
PROFILE_ORDER = ["profile never seen in dev", "seen once in dev", "seen >=2x, labels all agree", "seen >=2x, labels MIXED"]


def permutation_importance(pipe, test_df, y, n_repeats=10, seed=cfg.RANDOM_STATE):
    """Shuffle each ORIGINAL variable (with its missing-flags) among test rows, recompute the engineered
    features from the shuffled data, and measure how much macro-F1 / fatal recall drop."""
    rng = np.random.default_rng(seed)
    base = cm_from(y, pipe.predict(test_df[ALL_FEATURES]))
    base_f1, base_rec = macro_f1_from_cm(base), base[2, 2] / base[2].sum()
    rows = []
    for name, cols in GROUPS.items():
        d_f1, d_rec = [], []
        for _ in range(n_repeats):
            idx = rng.permutation(len(test_df))
            d = test_df.copy()
            for c in cols:
                d[c] = test_df[c].to_numpy()[idx]
            cm = cm_from(y, pipe.predict(add_features(d)[ALL_FEATURES]))
            d_f1.append(base_f1 - macro_f1_from_cm(cm))
            d_rec.append(base_rec - cm[2, 2] / cm[2].sum())
        rows.append({"variable": name, "macroF1_drop": np.mean(d_f1), "sd_F1": np.std(d_f1, ddof=1),
                     "fatal_recall_drop": np.mean(d_rec)})
    return pd.DataFrame(rows).sort_values("macroF1_drop", ascending=False).reset_index(drop=True), base_f1


def profile_analysis(dev, test, yd, yt, pred, cols=None):
    """Is the model's error concentrated in feature profiles whose labels are inconsistent in the dev data?"""
    cnt = {}
    kd = profile_key(dev) if cols is None else _key(dev, cols)
    kt_ = profile_key(test) if cols is None else _key(test, cols)
    for k, y in zip(kd.to_numpy(), yd):
        cnt.setdefault(k, np.zeros(3, dtype=int))[y] += 1
    cat, maj = [], []
    for k in kt_.to_numpy():
        a = cnt.get(k)
        if a is None:
            cat.append(PROFILE_ORDER[0]); maj.append(-1)
        else:
            n = a.sum()
            cat.append(PROFILE_ORDER[1] if n == 1 else (PROFILE_ORDER[2] if a.max() == n else PROFILE_ORDER[3]))
            maj.append(int(a.argmax()))
    t = pd.DataFrame({"type": cat, "correct": pred == yt, "maj": maj, "y": yt})
    out = []
    total_err = (~t["correct"]).sum()
    for g in PROFILE_ORDER:
        s = t[t["type"] == g]
        if len(s) == 0:
            continue
        seen = s[s["maj"] >= 0]
        out.append({"profile type": g, "n": len(s), "share of test": len(s) / len(t),
                    "model accuracy": s["correct"].mean(),
                    "share of ALL errors": (~s["correct"]).sum() / total_err,
                    "dev-majority-label accuracy": (seen["maj"] == seen["y"]).mean() if len(seen) else np.nan})
    return pd.DataFrame(out)


def subgroup_table(test, y, pred, col):
    d = pd.DataFrame({"g": test[col].astype(str).to_numpy(), "y": y, "p": pred})
    rows = []
    for g, s in d.groupby("g"):
        nf, flagged = (s["y"] == 2).sum(), (s["p"] == 2).sum()
        rows.append({col: g, "n": len(s), "accuracy": (s["y"] == s["p"]).mean(), "n_fatal": nf,
                     "fatal recall": (s.loc[s["y"] == 2, "p"] == 2).mean() if nf >= 20 else np.nan,
                     "fatal precision": (s.loc[s["p"] == 2, "y"] == 2).mean() if flagged >= 20 else np.nan})
    return pd.DataFrame(rows).sort_values("n", ascending=False)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bundle = joblib.load(cfg.MODEL_DIR / "hgb_final.joblib")
    pipe = bundle["pipeline"]
    labeled, _, _ = clean(load_raw())
    df = add_features(labeled)
    dev_idx, test_idx = make_split(df, mode="random")
    dev, test = df.iloc[dev_idx], df.iloc[test_idx]
    m = {c: i for i, c in enumerate(cfg.CLASS_ORDER)}
    yd, yt = dev[cfg.TARGET].map(m).to_numpy(), test[cfg.TARGET].map(m).to_numpy()
    pred = pipe.predict(test[ALL_FEATURES])
    out = [f"ERROR ANALYSIS of the final HistGradientBoosting | test rows {len(test):,} | "
           f"accuracy {np.mean(pred == yt):.3f}", ""]

    imp, base_f1 = permutation_importance(pipe, test, yt)
    out += [f"1) PERMUTATION IMPORTANCE (drop in test macro-F1 when a variable is shuffled; baseline {base_f1:.3f}; 10 repeats)",
            imp.round(4).to_string(index=False), ""]
    fig, ax = plt.subplots(figsize=(8, 0.4 * len(imp) + 1.5))
    ax.barh(imp["variable"][::-1], imp["macroF1_drop"][::-1], xerr=imp["sd_F1"][::-1], color="#34495e")
    ax.set_xlabel("drop in macro-F1 when shuffled"); ax.set_title("Permutation importance (final model, test set)")
    cfg.FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(cfg.FIG_DIR / "50_permutation_importance.png", bbox_inches="tight"); plt.close(fig)

    pa = profile_analysis(dev, test, yd, yt, pred)
    out += ["2) WHERE DO THE ERRORS COME FROM? (feature profile = exact combination of all input values)",
            pa.round(3).to_string(index=False),
            "   If most errors sit in 'MIXED' profiles, they are largely irreducible label noise, not model weakness.", ""]
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.barh(pa["profile type"][::-1], pa["model accuracy"][::-1], color="#2c7fb8")
    for i, (a, s) in enumerate(zip(pa["model accuracy"][::-1], pa["share of ALL errors"][::-1])):
        ax.text(a + 0.01, i, f"acc {a:.0%} | {s:.0%} of all errors", va="center", fontsize=8)
    ax.set_xlim(0, 1.35); ax.set_xlabel("accuracy"); ax.set_title("Model accuracy by profile type (test set)")
    fig.savefig(cfg.FIG_DIR / "51_accuracy_by_profile_type.png", bbox_inches="tight"); plt.close(fig)

    pc = profile_analysis(dev, test, yd, yt, pred, cols=CORE5)
    out += ["2b) THE SAME ANALYSIS USING ONLY THE FIVE CORE VARIABLES AS THE 'PROFILE' (speed, light, vehicle, weather, road surface)",
            "    (the profile in 2) includes age, which the model barely uses, so most profiles look 'never seen' for a trivial reason)",
            pc.round(3).to_string(index=False), ""]
    test = test.copy()
    test["speed_limit_cat"] = test["speed_limit"].map(lambda v: "missing" if pd.isna(v) else str(int(v)))
    test["age_status"] = test["age_missing"].map({0: "age present", 1: "age blank"})
    out.append("3) PERFORMANCE BY SUBGROUP (fatal recall/precision shown only when n_fatal / n_flagged >= 20)")
    for col in ["speed_limit_cat", "light_conditions", "vehicle_type", "sex_of_driver", "weather_conditions", "age_status"]:
        out += ["", subgroup_table(test, yt, pred, col).round(3).to_string(index=False)]
    text = "\n".join(out)
    print(text)
    (cfg.ROOT / "reports" / "error_analysis.txt").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
