"""Value-level audit of every column (except accident_index).

Run:  python -m src.inspect_values
Prints a report and saves it to reports/value_audit.txt. Nothing is modified or dropped.
Paste the report back so cleaning rules can be written from what is actually in the data.
"""
import re
import pandas as pd
from . import config as cfg

CATEGORICAL_LIKE = cfg.CATEGORICAL_COLS + [cfg.TARGET]
NUMERIC_LIKE = cfg.NUMERIC_COLS
VALID_SPEEDS = {20, 30, 40, 50, 60, 70}


def norm(x: str) -> str:
    return re.sub(r"\s+", " ", str(x).strip().lower())


def main():
    # Read everything as strings so nothing is silently coerced or hidden.
    df = pd.read_csv(cfg.RAW_PATH, dtype=str, keep_default_na=False)
    df.columns = [c.strip().lower() for c in df.columns]
    cols = [c for c in df.columns if c != cfg.ID_COL]

    out = []
    p = out.append
    p(f"Rows: {len(df):,} | Columns audited: {len(cols)} (excluding {cfg.ID_COL})")
    p(f"Fully duplicated rows (ignoring {cfg.ID_COL}): {df[cols].duplicated().sum():,}")
    unexpected = set(cols) - set(CATEGORICAL_LIKE + NUMERIC_LIKE)
    if unexpected:
        p(f"Columns not in config (check names): {sorted(unexpected)}")

    for c in cols:
        s = df[c]
        p("\n" + "=" * 90)
        p(f"COLUMN: {c}")
        p("=" * 90)
        blank = (s.str.strip() == "").sum()
        p(f"blank/empty entries: {blank:,} ({blank / len(df):.2%}) | unique raw values: {s.nunique():,}")

        if c in NUMERIC_LIKE:
            num = pd.to_numeric(s.replace("", pd.NA), errors="coerce")
            bad = s[(num.isna()) & (s.str.strip() != "")]
            p(f"non-numeric, non-blank entries: {len(bad):,}")
            if len(bad):
                p("  offending values (top 20):")
                p(bad.value_counts().head(20).to_string())
            p("numeric summary:")
            p(num.describe().round(2).to_string())
            if c == "speed_limit":
                odd = num.dropna()[~num.dropna().isin(VALID_SPEEDS)]
                p(f"speed limits outside {sorted(VALID_SPEEDS)}: {len(odd):,}")
                if len(odd):
                    p(odd.value_counts().to_string())
            if c == "age_of_oldest_driver":
                odd = num.dropna()[(num.dropna() < 16) | (num.dropna() > 100)]
                p(f"ages <16 or >100: {len(odd):,}")
                if len(odd):
                    p(odd.value_counts().sort_index().to_string())
            p("all distinct values (sorted):")
            p(", ".join(sorted(s.unique(), key=lambda v: (not v.replace('.', '', 1).isdigit(), v))[:80]))
        else:
            vc = s.value_counts(dropna=False)
            p("value counts (repr shows stray spaces/case):")
            for val, n in vc.head(60).items():
                p(f"  {n:>8,}  {val!r}")
            if len(vc) > 60:
                p(f"  ... {len(vc) - 60} more distinct values")
            # Variants that collapse to the same text after trimming/lowercasing
            groups = {}
            for v in vc.index:
                groups.setdefault(norm(v), []).append(v)
            clashes = {k: v for k, v in groups.items() if len(v) > 1}
            if clashes:
                p("INCONSISTENT VARIANTS (same text after strip/lowercase):")
                for k, v in clashes.items():
                    p(f"  {k!r}: {v}")
            else:
                p("no whitespace/case variants found")

    p("\n" + "=" * 90)
    p("MISSINGNESS CO-OCCURRENCE (rows where the 'missing' label or blank appears)")
    p("=" * 90)
    miss = pd.DataFrame({c: (df[c].map(norm).isin([cfg.MISSING_LABEL, ""])) for c in cols})
    p(miss.sum().sort_values(ascending=False).to_string())
    p(f"rows with 0 missing: {(miss.sum(axis=1) == 0).sum():,} | with >=1: {(miss.sum(axis=1) >= 1).sum():,} "
      f"| with >=4: {(miss.sum(axis=1) >= 4).sum():,}")

    text = "\n".join(out)
    print(text)
    cfg.ROOT.joinpath("reports").mkdir(exist_ok=True)
    (cfg.ROOT / "reports" / "value_audit.txt").write_text(text, encoding="utf-8")
    print("\nSaved to reports/value_audit.txt")


if __name__ == "__main__":
    main()
