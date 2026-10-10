"""Phase 3: exploratory data analysis.   Run: python -m src.eda
Saves figures to reports/figures/ and a text summary to reports/eda_summary.txt."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, kruskal
from . import config as cfg
from .data import load_raw, clean

COLORS = {"slight": "#4C9F70", "serious": "#E0A030", "fatal": "#C0392B"}
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})
T = cfg.TARGET


def short(s, n=38):
    s = str(s)
    return s if len(s) <= n else s[: n - 1] + "…"


def save(fig, name):
    cfg.FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(cfg.FIG_DIR / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def cramers_v(x, y):
    ct = pd.crosstab(x, y)
    chi2, p, _, _ = chi2_contingency(ct, correction=False)
    n = ct.values.sum(); r, k = ct.shape
    phi2 = max(0.0, chi2 / n - (k - 1) * (r - 1) / (n - 1))
    rc = r - (r - 1) ** 2 / (n - 1); kc = k - (k - 1) ** 2 / (n - 1)
    return float(np.sqrt(phi2 / max(1e-12, min(kc - 1, rc - 1)))), float(p)


def fig_target(df):
    vc = df[T].value_counts().reindex(cfg.CLASS_ORDER)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(vc.index, vc.values, color=[COLORS[k] for k in vc.index])
    for i, v in enumerate(vc.values):
        ax.text(i, v, f"{v:,}\n{v / vc.sum():.1%}", ha="center", va="bottom")
    ax.set_title("Accident severity (labeled records)"); ax.set_ylabel("count")
    ax.set_ylim(0, vc.max() * 1.18)
    save(fig, "00_target_distribution")


def fig_stacked(df, col, idx, sort_by_fatal=True):
    ct = pd.crosstab(df[col], df[T]).reindex(columns=cfg.CLASS_ORDER, fill_value=0)
    n = ct.sum(axis=1)
    pct = ct.div(n, axis=0) * 100
    pct = pct.sort_values("fatal") if sort_by_fatal else pct.sort_index()
    n = n.loc[pct.index]
    labels = [f"{short(i)}  (n={n[i]:,})" for i in pct.index]
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(pct) + 1.8))
    left = np.zeros(len(pct))
    for k in cfg.CLASS_ORDER:
        ax.barh(labels, pct[k].values, left=left, color=COLORS[k], label=k)
        for y, (l, w) in enumerate(zip(left, pct[k].values)):
            if w >= 6:
                ax.text(l + w / 2, y, f"{w:.0f}%", ha="center", va="center", color="white", fontsize=8)
        left += pct[k].values
    ax.set_xlim(0, 100); ax.set_xlabel("% of accidents"); ax.set_title(f"Severity mix by {col}")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18), frameon=False)
    save(fig, f"{idx:02d}_severity_by_{col}")


def fig_age(df):
    d = df.dropna(subset=["age_of_oldest_driver"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for k in cfg.CLASS_ORDER:
        axes[0].hist(d.loc[d[T] == k, "age_of_oldest_driver"], bins=range(10, 105, 5),
                     density=True, alpha=0.5, color=COLORS[k], label=k)
    axes[0].set_xlabel("age of oldest driver"); axes[0].set_ylabel("density"); axes[0].legend()
    axes[0].set_title("Age distribution by severity")
    axes[1].boxplot([d.loc[d[T] == k, "age_of_oldest_driver"] for k in cfg.CLASS_ORDER], showfliers=False)
    axes[1].set_xticklabels(cfg.CLASS_ORDER)
    axes[1].set_title("Age by severity (no outliers)")
    save(fig, "20_age_by_severity")


def fig_heat(df, a, b, idx, min_n=30):
    d = df.assign(_f=(df[T] == "fatal").astype(int))
    g = d.groupby([a, b])["_f"].agg(["mean", "size"]).reset_index()
    rate = g.pivot(index=a, columns=b, values="mean")
    cnt = g.pivot(index=a, columns=b, values="size")
    rate = rate.where(cnt >= min_n)
    fig, ax = plt.subplots(figsize=(1.9 * rate.shape[1] + 2.5, 0.7 * rate.shape[0] + 1.8))
    im = ax.imshow(rate.values.astype(float), cmap="Reds", aspect="auto")
    ax.set_xticks(range(rate.shape[1])); ax.set_xticklabels([short(c, 22) for c in rate.columns], rotation=20, ha="right")
    ax.set_yticks(range(rate.shape[0])); ax.set_yticklabels([short(i, 22) for i in rate.index])
    for i in range(rate.shape[0]):
        for j in range(rate.shape[1]):
            v = rate.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0%}\nn={int(cnt.values[i, j]):,}", ha="center", va="center", fontsize=8)
    ax.set_title(f"Fatal share: {a} x {b}  (cells with n<{min_n} hidden)")
    fig.colorbar(im, ax=ax, label="fatal share")
    save(fig, f"{idx:02d}_fatal_heat_{a}_x_{b}")


def main():
    raw = load_raw()
    df, _, _ = clean(raw)
    df = df.copy()
    df["speed_limit_cat"] = df["speed_limit"].map(lambda v: "missing" if pd.isna(v) else str(int(v)))
    df["age_status"] = df["age_missing"].map({0: "age present", 1: "age blank"})
    df["n_missing_fields"] = (df[cfg.CATEGORICAL_COLS] == cfg.MISSING_LABEL).sum(axis=1).clip(upper=4).astype(str)
    df["n_missing_fields"] = df["n_missing_fields"].replace({"4": "4+"})

    out = [f"EDA on {len(df):,} labeled, de-duplicated records", ""]
    fig_target(df)

    cat_cols = ["speed_limit_cat"] + cfg.CATEGORICAL_COLS
    for i, c in enumerate(cat_cols, start=1):
        fig_stacked(df, c, i)
    fig_stacked(df, "age_status", 12)
    fig_stacked(df, "n_missing_fields", 13, sort_by_fatal=False)
    fig_age(df)

    heat_pairs = [("speed_limit_cat", "light_conditions"), ("speed_limit_cat", "vehicle_type"),
                  ("speed_limit_cat", "road_surface_conditions")]
    for j, (a, b) in enumerate(heat_pairs, start=30):
        fig_heat(df, a, b, j)

    # association with the target
    rows = []
    for c in cat_cols + ["age_status", "n_missing_fields"]:
        v, p = cramers_v(df[c], df[T])
        rows.append((c, v, p))
    assoc = pd.DataFrame(rows, columns=["feature", "cramers_v", "chi2_p"]).sort_values("cramers_v", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 0.4 * len(assoc) + 1.5))
    ax.barh(assoc["feature"][::-1], assoc["cramers_v"][::-1], color="#34495e")
    ax.set_xlabel("bias-corrected Cramér's V with accident_severity"); ax.set_title("Strength of association with severity")
    save(fig, "40_association_with_target")
    out += ["ASSOCIATION WITH SEVERITY (Cramér's V; ~0.1 weak, ~0.3 moderate)", assoc.round(4).to_string(index=False), ""]

    groups = [df.loc[df[T] == k, "age_of_oldest_driver"].dropna() for k in cfg.CLASS_ORDER]
    H, p = kruskal(*groups)
    n = sum(len(g) for g in groups)
    out += ["AGE vs SEVERITY (Kruskal-Wallis)", f"   H={H:.1f}, p={p:.2e}, eta^2={(H - 2) / (n - 3):.4f}",
            df.groupby(T)["age_of_oldest_driver"].describe().round(1).to_string(), ""]

    # highest / lowest fatal share (n>=100)
    recs = []
    for c in cat_cols:
        g = df.groupby(c)[T].agg(n="size", fatal=lambda s: (s == "fatal").mean()).reset_index()
        g.insert(0, "feature", c); g = g.rename(columns={c: "category"})
        recs.append(g)
    allc = pd.concat(recs); allc = allc[allc["n"] >= 100].sort_values("fatal", ascending=False)
    out += ["HIGHEST FATAL SHARE (n>=100)", allc.head(10).round(3).to_string(index=False), "",
            "LOWEST FATAL SHARE (n>=100)", allc.tail(10).round(3).to_string(index=False)]

    text = "\n".join(out)
    print(text)
    (cfg.ROOT / "reports").mkdir(exist_ok=True)
    (cfg.ROOT / "reports" / "eda_summary.txt").write_text(text, encoding="utf-8")
    print(f"\nFigures saved to {cfg.FIG_DIR}")


if __name__ == "__main__":
    main()
