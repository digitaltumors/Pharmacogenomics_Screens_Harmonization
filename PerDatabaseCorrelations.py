import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Literal

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# ============================================================
# Config
# ============================================================

METRIC_WHITELIST = [
    "fit_auc", "fit_ic50", "fit_ec50",
    "auc_emp",
    "fit_auc_win", "auc_emp_win",
]

AUC_METRICS = {"fit_auc", "auc_emp", "fit_auc_win", "auc_emp_win"}
IC_METRICS = {"fit_ic50", "fit_ec50"}
IC_EC_COLS = ["fit_ic50", "fit_ec50"]

DRUG_KEYS = ["canonical_drug", "DrugID", "drug", "improve_drug_id"]
CELL_KEYS = ["canonical_cell", "cell", "CellID", "cell_line", "improve_sample_id", "sample"]

# IC/EC values are assumed to be in log10(µM) from the fitter.
# 1 nM  = 1e-3 µM → log10(µM) = -3
# 100 µM       → log10(µM) =  2
LOG10UM_MIN, LOG10UM_MAX = -3.0, 2.0  # 1 nM .. 100 µM log10(µM)

# ============================================================
# IO helpers
# ============================================================

def _is_resource_fork(path: Path) -> bool:
    return path.name.startswith("._")


def _read_csv(path: Path) -> pd.DataFrame:
    for enc in (None, "utf-8", "utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(path, dtype=str, low_memory=False, encoding=enc)
            break
        except Exception:
            continue
    else:
        df = pd.read_csv(path, dtype=str, low_memory=False, engine="python")

    drop_cols = [c for c in df.columns if re.match(r"^Unnamed:\s*0+$", str(c), flags=re.I)]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df


def _safe_write(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


# ============================================================
# Column picking / normalization
# ============================================================

def _pick_key(cols: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    lc = {c.lower(): c for c in cols}
    for want in candidates:
        if want.lower() in lc:
            return lc[want.lower()]
    # permissive contains (unique)
    for want in candidates:
        hits = [c for c in cols if want.lower() in c.lower()]
        if len(hits) == 1:
            return hits[0]
    return None


def _norm_like_overlap(s: pd.Series) -> pd.Series:
    # strip quotes, collapse whitespace, uppercase
    s = s.astype(str).str.strip().str.replace(r'^[\'"]|[\'"]$', '', regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.upper()
    return s


def _to_numeric_clean(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


# ============================================================
# AUC transforms (optional; defaults to 'none')
# ============================================================

_AUC_RE = re.compile(r'(^|[^a-zA-Z0-9])auc([^a-zA-Z0-9]|$)', re.IGNORECASE)


def _is_auc_col(col: str) -> bool:
    return bool(_AUC_RE.search(str(col)))


def transform_auc_for_dict(
    db_dict: Dict[str, pd.DataFrame],
    mode: Literal["none", "logit", "logit_on_1_minus"] = "none",
    eps: float = 1e-6,
) -> Dict[str, pd.DataFrame]:
    if mode == "none":
        return db_dict
    out: Dict[str, pd.DataFrame] = {}
    for key, df in db_dict.items():
        tmp = df.copy()
        for col in tmp.columns:
            if not _is_auc_col(col):
                continue
            x = _to_numeric_clean(tmp[col])
            finite = x[np.isfinite(x)]
            if finite.empty:
                continue
            p = x.copy()
            # If looks like 0–100, rescale to 0–1
            if float(np.nanmax(finite)) > 1.5:
                p = p / 100.0
            if mode == "logit_on_1_minus":
                p = 1.0 - p
            p = p.clip(lower=eps, upper=1.0 - eps)
            with np.errstate(divide="ignore", invalid="ignore"):
                tmp[col] = np.log(p / (1.0 - p))
        out[key] = tmp
    return out


# ============================================================
# IC/EC filtering (log10 µM)
# ============================================================

def apply_ic_ec_filter_for_dict(
    db_dict: Dict[str, pd.DataFrame],
    mode: Literal["window", "upper", "off"] = "off",
    lo_log10uM: float = LOG10UM_MIN,
    hi_log10uM: float = LOG10UM_MAX,
) -> Dict[str, pd.DataFrame]:
    """
    Filter IC/EC values (fit_ic50 / fit_ec50), assumed to be log10(µM):

      - mode="window":
          * values < lo_log10uM (e.g. < 1 nM) and > hi_log10uM (e.g. > 100 µM)
            are set to NaN (ignored in correlations).
      - mode="upper":
          * values > hi_log10uM (e.g. > 100 µM) are set to NaN;
            no lower bound.
      - mode="off":
          * no filtering is applied (values left as-is).
    """
    if mode == "off":
        return db_dict

    out: Dict[str, pd.DataFrame] = {}
    for key, df in db_dict.items():
        tmp = df.copy()
        for col in IC_EC_COLS:
            if col not in tmp.columns:
                continue
            s = _to_numeric_clean(tmp[col])
            if mode == "window":
                mask = (s >= lo_log10uM) & (s <= hi_log10uM)
            elif mode == "upper":
                mask = s <= hi_log10uM
            else:
                mask = np.ones(len(s), dtype=bool)
            tmp[col] = s.where(mask, np.nan)
        out[key] = tmp
    return out


# ============================================================
# Prepare per-DB and build long table
# ============================================================

def _prepare_df_alias_aware(
    df: pd.DataFrame,
    rep_agg: Literal["median", "mean"],
) -> Tuple[pd.DataFrame, List[str]]:
    drug_key = "canonical_drug" if "canonical_drug" in df.columns else _pick_key(df.columns, DRUG_KEYS)
    cell_key = "canonical_cell" if "canonical_cell" in df.columns else _pick_key(df.columns, CELL_KEYS)
    if not drug_key or not cell_key:
        raise KeyError(f"Could not resolve drug/cell columns. Seen: {df.columns.tolist()}")

    drug = _norm_like_overlap(df[drug_key])
    cell = _norm_like_overlap(df[cell_key])

    keep_metrics = [c for c in df.columns if c in METRIC_WHITELIST]
    if not keep_metrics:
        raise ValueError("No whitelisted metric columns present.")

    clean = pd.DataFrame({"drug": drug, "cell": cell})
    for m in keep_metrics:
        clean[m] = _to_numeric_clean(df[m])

    group_keys = ["drug", "cell"]
    if "replicate" in df.columns:
        agg = "median" if rep_agg == "median" else "mean"
        clean = clean.groupby(group_keys, as_index=False)[keep_metrics].agg(agg)
    else:
        clean = clean.drop_duplicates(group_keys, keep="first")

    return clean, keep_metrics


def build_long_from_db_dict(
    db_dict: Dict[str, pd.DataFrame],
    rep_agg: Literal["median", "mean"],
) -> pd.DataFrame:
    canon: Dict[str, Tuple[pd.DataFrame, List[str]]] = {}
    for db_name, df in db_dict.items():
        canon[str(db_name)] = _prepare_df_alias_aware(df, rep_agg)

    rows = []
    dbs = list(canon.keys())
    for i in range(len(dbs)):
        for j in range(i + 1, len(dbs)):
            db1, db2 = dbs[i], dbs[j]
            df1, mets1 = canon[db1]
            df2, mets2 = canon[db2]
            shared_metrics = sorted(set(mets1) & set(mets2))
            if not shared_metrics or df1.empty or df2.empty:
                continue

            merged = df1.merge(df2, on=["drug", "cell"], suffixes=("_db1", "_db2"))
            if merged.empty:
                continue

            for m in shared_metrics:
                sub = merged[["drug", "cell", f"{m}_db1", f"{m}_db2"]].rename(
                    columns={f"{m}_db1": "val_db1", f"{m}_db2": "val_db2"}
                )
                sub["val_db1"] = _to_numeric_clean(sub["val_db1"])
                sub["val_db2"] = _to_numeric_clean(sub["val_db2"])
                sub = sub.dropna(subset=["val_db1", "val_db2"])
                if sub.empty:
                    continue
                sub["metric"] = m
                sub["db1"] = db1
                sub["db2"] = db2
                sub["db_pair"] = db1 + "|" + db2
                rows.append(
                    sub[["metric", "db_pair", "drug", "cell", "val_db1", "val_db2", "db1", "db2"]]
                )

    if not rows:
        return pd.DataFrame(
            columns=["metric", "db_pair", "drug", "cell", "val_db1", "val_db2", "db1", "db2", "_level"]
        )

    out = pd.concat(rows, ignore_index=True)
    out["_level"] = out[["val_db1", "val_db2"]].astype(float).mean(axis=1)
    return out


# ============================================================
# Biologically motivated bins
# ============================================================

def add_bio_bins(
    df_long: pd.DataFrame,
    bin_mode: str,
    auc_thresh: float,
    auc_quantile: float,
    ic_thresh: float,
) -> pd.DataFrame:
    """
    Adds a 'bio_bin' column to df_long according to bin_mode.

    bin_mode:
      - 'none'               : all rows in bin 'all'
      - 'auc_lt_fixed'       : for AUC metrics, _level < auc_thresh → 'AUC_lt', else 'AUC_ge'
      - 'auc_bottom_quantile': for AUC metrics, bottom auc_quantile of _level per (metric,db_pair)
                               in 'AUC_bottom', rest 'AUC_rest'
      - 'ic_lt_fixed'        : for IC metrics, _level < ic_thresh → 'IC_lt', else 'IC_ge'
    """
    df = df_long.copy()
    df["bio_bin"] = "all"

    if bin_mode == "none":
        return df

    if bin_mode == "auc_lt_fixed":
        mask_auc = df["metric"].isin(AUC_METRICS)
        df.loc[mask_auc & (df["_level"] < auc_thresh), "bio_bin"] = "AUC_lt"
        df.loc[mask_auc & (df["_level"] >= auc_thresh), "bio_bin"] = "AUC_ge"
        return df

    if bin_mode == "ic_lt_fixed":
        mask_ic = df["metric"].isin(IC_METRICS)
        df.loc[mask_ic & (df["_level"] < ic_thresh), "bio_bin"] = "IC_lt"
        df.loc[mask_ic & (df["_level"] >= ic_thresh), "bio_bin"] = "IC_ge"
        return df

    if bin_mode == "auc_bottom_quantile":
        out_rows = []
        for (metric, db_pair), g in df.groupby(["metric", "db_pair"], sort=False):
            if metric not in AUC_METRICS:
                g = g.copy()
                g["bio_bin"] = "all"
                out_rows.append(g)
                continue
            levels = g["_level"].to_numpy(float)
            if levels.size == 0 or not np.isfinite(levels).any():
                g = g.copy()
                g["bio_bin"] = "all"
                out_rows.append(g)
                continue
            thresh = float(np.nanquantile(levels, auc_quantile))
            g = g.copy()
            mask_bottom = g["_level"] <= thresh
            g.loc[mask_bottom, "bio_bin"] = "AUC_bottom"
            g.loc[~mask_bottom, "bio_bin"] = "AUC_rest"
            out_rows.append(g)
        return pd.concat(out_rows, ignore_index=True)

    raise ValueError(f"Unknown bin_mode: {bin_mode!r}")


# ============================================================
# Per-bin stats (Spearman only)
# ============================================================

def compute_per_bin_stats(
    df_long: pd.DataFrame,
    auc_thresh: float,   # kept for signature compatibility (unused here)
    ic_thresh: float,    # kept for signature compatibility (unused here)
    cohort: str,
) -> pd.DataFrame:
    """
    For each metric × db_pair × bio_bin, compute:
      - Spearman r
      - n_pairs          : number of (drug, cell) pairs
      - n_unique_drugs   : unique drugs in this bin
      - n_unique_cells   : unique cells in this bin

    Adds a 'cohort' column.
    """
    rows = []

    for (metric, db_pair, bio_bin), g in df_long.groupby(["metric", "db_pair", "bio_bin"]):
        x = g["val_db1"].to_numpy(float)
        y = g["val_db2"].to_numpy(float)
        mask = np.isfinite(x) & np.isfinite(y)

        x = x[mask]
        y = y[mask]

        g_valid = g.loc[mask].copy()

        n_pairs = x.size
        n_drugs = g_valid["drug"].nunique() if "drug" in g_valid.columns else np.nan
        n_cells = g_valid["cell"].nunique() if "cell" in g_valid.columns else np.nan

        if n_pairs < 3:
            rows.append(
                {
                    "cohort": cohort,
                    "metric": metric,
                    "db_pair": db_pair,
                    "bio_bin": bio_bin,
                    "n": n_pairs,
                    "n_pairs": n_pairs,
                    "n_unique_drugs": n_drugs,
                    "n_unique_cells": n_cells,
                    "spearman_r": np.nan,
                }
            )
            continue

        r_s, _ = spearmanr(x, y)
        rows.append(
            {
                "cohort": cohort,
                "metric": metric,
                "db_pair": db_pair,
                "bio_bin": bio_bin,
                "n": n_pairs,
                "n_pairs": n_pairs,
                "n_unique_drugs": n_drugs,
                "n_unique_cells": n_cells,
                "spearman_r": float(r_s),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# Extreme-bin summaries for example selection
# ============================================================

def summarize_extreme_correlations(
    stats_df: pd.DataFrame,
    *,
    top_k: int = 5,
    min_n: int = 10,
) -> pd.DataFrame:
    """
    Pick bins with very high and very low Spearman correlations, per metric.

    Returns a small table with columns:
      cohort, metric, db_pair, bio_bin, n, spearman_r, extreme_label
    """
    if stats_df.empty or "spearman_r" not in stats_df.columns:
        return pd.DataFrame(
            columns=[
                "cohort", "metric", "db_pair", "bio_bin",
                "n", "spearman_r", "extreme_label",
            ]
        )

    df_valid = stats_df.copy()
    df_valid = df_valid[(df_valid["n"] >= min_n) & np.isfinite(df_valid["spearman_r"])]

    if df_valid.empty:
        return pd.DataFrame(
            columns=[
                "cohort", "metric", "db_pair", "bio_bin",
                "n", "spearman_r", "extreme_label",
            ]
        )

    rows: list[pd.DataFrame] = []
    for metric, g in df_valid.groupby("metric"):
        top = g.sort_values("spearman_r", ascending=False).head(top_k).copy()
        top["extreme_label"] = "high_corr"

        bottom = g.sort_values("spearman_r", ascending=True).head(top_k).copy()
        bottom["extreme_label"] = "low_corr"

        rows.append(top)
        rows.append(bottom)

    out = pd.concat(rows, ignore_index=True)

    keep_cols = [
        "cohort", "metric", "db_pair", "bio_bin",
        "n", "spearman_r", "extreme_label",
    ]
    existing = [c for c in keep_cols if c in out.columns]
    others = [c for c in out.columns if c not in existing]
    return out[existing + others]


def build_per_pair_extreme_examples(
    df_binned: pd.DataFrame,
    extreme_bins: pd.DataFrame,
    *,
    max_pairs_per_bin: int = 100,
) -> pd.DataFrame:
    """
    Pair-level table restricted to rows from extreme bins.

    Columns include:
      cohort, metric, db_pair, bio_bin, extreme_label,
      n_bin, spearman_r_bin,
      drug, cell, val_db1, val_db2, db1, db2
    """
    if df_binned.empty or extreme_bins.empty:
        return pd.DataFrame(
            columns=[
                "cohort", "metric", "db_pair", "bio_bin", "extreme_label",
                "n_bin", "spearman_r_bin",
                "drug", "cell", "val_db1", "val_db2", "db1", "db2",
            ]
        )

    if "cohort" not in df_binned.columns:
        raise KeyError("df_binned must contain a 'cohort' column for merging extreme pairs.")

    bin_cols = [
        "cohort", "metric", "db_pair", "bio_bin",
        "n", "spearman_r", "extreme_label",
    ]
    bin_cols = [c for c in bin_cols if c in extreme_bins.columns]
    eb = extreme_bins[bin_cols].copy()

    merged = df_binned.merge(
        eb,
        on=["cohort", "metric", "db_pair", "bio_bin"],
        how="inner",
        suffixes=("", "_bin"),
    )
    if merged.empty:
        return pd.DataFrame(
            columns=[
                "cohort", "metric", "db_pair", "bio_bin", "extreme_label",
                "n_bin", "spearman_r_bin",
                "drug", "cell", "val_db1", "val_db2", "db1", "db2",
            ]
        )

    rows: list[pd.DataFrame] = []
    group_keys = ["cohort", "metric", "db_pair", "bio_bin", "extreme_label"]
    for _, g in merged.groupby(group_keys):
        if len(g) > max_pairs_per_bin:
            g = g.sample(n=max_pairs_per_bin, random_state=0)
        rows.append(g)

    out = pd.concat(rows, ignore_index=True)

    out = out.rename(
        columns={
            "n": "n_bin",
            "spearman_r": "spearman_r_bin",
        }
    )

    col_order = [
        "cohort", "metric", "db_pair", "bio_bin", "extreme_label",
        "n_bin", "spearman_r_bin",
        "drug", "cell", "val_db1", "val_db2", "db1", "db2",
    ]
    existing = [c for c in col_order if c in out.columns]
    others = [c for c in out.columns if c not in existing]
    return out[existing + others]


# ============================================================
# FULL + FILTERED collection (no GDSC2 compile)
# ============================================================

def _prefer_first_existing(input_dir: Path, names: List[str]) -> Optional[Path]:
    for nm in names:
        p = input_dir / nm
        if p.exists() and p.is_file() and not _is_resource_fork(p) and p.name.endswith("_harmonized.csv"):
            return p
    return None


def _collect_four_dbs(input_dir: Path, work_root: Path) -> Tuple[Dict[str, Path], Dict[str, Path]]:
    # NOTE: work_root is kept in signature for minimal downstream changes; unused now.

    ctrp_full = _prefer_first_existing(
        input_dir,
        [
            "CTRP_replicates_AUC_harmonized.csv",
            "CTRP_harmonized.csv",
            "CTRPv2_harmonized.csv",
        ],
    )
    ctrp_filt = _prefer_first_existing(
        input_dir,
        [
            "CTRP_replicates_filtered_AUC_harmonized.csv",
            "CTRP_replicates_filtered_harmonized.csv",
        ],
    )

    gdsc1_full = _prefer_first_existing(input_dir, ["GDSC1_replicates_AUC_harmonized.csv"])
    gdsc1_filt = _prefer_first_existing(input_dir, ["GDSC1_replicates_Filtered_AUC_harmonized.csv"])

    # ---- GDSC2: NO COMPILE ----
    # We simply pick a single existing harmonized file for each cohort if present.
    gdsc2_full = _prefer_first_existing(
        input_dir,
        [
            "GDSC2_Compile_replicates_AUC_harmonized.csv",
            "GDSC2_replicates_AUC_harmonized.csv",
            "GDSC2_AUC_harmonized.csv",
        ],
    )
    gdsc2_filt = _prefer_first_existing(
        input_dir,
        [
            "GDSC2_Compile_replicates_filtered_AUC_harmonized.csv",
            "GDSC2_replicates_filtered_AUC_harmonized.csv",
            "GDSC2_filtered_AUC_harmonized.csv",
        ],
    )

    prism_full = _prefer_first_existing(input_dir, ["PRISM_AUC_harmonized.csv"])
    prism_filt = _prefer_first_existing(input_dir, ["PRISM_replicates_Filtered_AUC_harmonized.csv"])

    full: Dict[str, Path] = {}
    filt: Dict[str, Path] = {}

    if ctrp_full:
        full["CTRP"] = ctrp_full
    if gdsc1_full:
        full["GDSC1"] = gdsc1_full
    if gdsc2_full:
        full["GDSC2"] = gdsc2_full
    if prism_full:
        full["PRISM"] = prism_full

    if ctrp_filt:
        filt["CTRP"] = ctrp_filt
    if gdsc1_filt:
        filt["GDSC1"] = gdsc1_filt
    if gdsc2_filt:
        filt["GDSC2"] = gdsc2_filt
    if prism_filt:
        filt["PRISM"] = prism_filt

    return full, filt


# ============================================================
# Run one cohort (full OR filtered)
# ============================================================

def _run_one_cohort_biobins(
    paths: Dict[str, Path],
    cohort_label: str,
    out_dir: Path,
    rep_agg: str,
    auc_mode: str,
    auc_eps: float,
    bin_mode: str,
    auc_thresh: float,
    auc_quantile: float,
    ic_thresh: float,
    ic_clip_mode: str,
):
    if not paths:
        print(f"[info] Cohort '{cohort_label}' has no usable tables; skipping.")
        return

    db_dict: Dict[str, pd.DataFrame] = {}
    for label, path in sorted(paths.items()):
        try:
            df = _read_csv(path)
            db_dict[label] = df
        except Exception as e:
            print(f"[warn] Skipping {label} ({path.name}) in cohort '{cohort_label}': {e}")

    if len(db_dict) < 2:
        print(f"[info] Cohort '{cohort_label}': only {len(db_dict)} usable tables; need ≥2. Skipping.")
        return

    # Optional AUC transform
    db_dict = transform_auc_for_dict(db_dict, mode=auc_mode, eps=auc_eps)

    # IC/EC clipping (log10 µM)
    db_dict = apply_ic_ec_filter_for_dict(
        db_dict,
        mode=ic_clip_mode,
        lo_log10uM=LOG10UM_MIN,
        hi_log10uM=LOG10UM_MAX,
    )

    # Build long table
    df_long = build_long_from_db_dict(db_dict, rep_agg=rep_agg)
    if df_long.empty:
        print(f"[info] Cohort '{cohort_label}': no overlaps across DB pairs; skipping.")
        return

    # Biologically motivated bins
    df_binned = add_bio_bins(
        df_long,
        bin_mode=bin_mode,
        auc_thresh=auc_thresh,
        auc_quantile=auc_quantile,
        ic_thresh=ic_thresh,
    )
    df_binned["cohort"] = cohort_label

    # Per-bin stats (Spearman only)
    stats_df = compute_per_bin_stats(
        df_binned,
        auc_thresh=auc_thresh,
        ic_thresh=ic_thresh,
        cohort=cohort_label,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    _safe_write(stats_df, out_dir / "per_bin_stats.csv")

    # --------------------------------------------------------
    # Extreme bins (high/low correlation) for quick inspection
    # --------------------------------------------------------
    extreme_bins = summarize_extreme_correlations(
        stats_df,
        top_k=5,
        min_n=10,
    )
    if not extreme_bins.empty:
        _safe_write(extreme_bins, out_dir / "per_bin_extreme_examples.csv")

        extreme_pairs = build_per_pair_extreme_examples(
            df_binned,
            extreme_bins,
            max_pairs_per_bin=100,
        )
        if not extreme_pairs.empty:
            _safe_write(extreme_pairs, out_dir / "per_pair_extreme_examples.csv")

    # provenance
    run_params = {
        "cohort": cohort_label,
        "tables_used": {k: str(v) for k, v in paths.items()},
        "replicate_aggregation": rep_agg,
        "auc_transform": auc_mode,
        "auc_eps": auc_eps,
        "bin_mode": bin_mode,
        "auc_threshold": auc_thresh,
        "auc_quantile": auc_quantile,
        "ic_threshold_log10uM": ic_thresh,
        "ic_clip_mode": ic_clip_mode,
        "ic_window_log10uM": [LOG10UM_MIN, LOG10UM_MAX],
        "metrics_whitelist": METRIC_WHITELIST,
        "auc_metrics": sorted(AUC_METRICS),
        "ic_metrics": sorted(IC_METRICS),
    }
    with open(out_dir / "run_params.json", "w") as f:
        json.dump(run_params, f, indent=2)

    print(f"[info] Cohort '{cohort_label}': wrote {out_dir / 'per_bin_stats.csv'}")


# ============================================================
# CLI
# ============================================================

def main():
    ap = argparse.ArgumentParser(
        description=(
            "Correlate CTRP, GDSC1, GDSC2, PRISM on *_harmonized.csv; "
            "full & filtered cohorts; IC/EC filters in log10(µM); "
            "biologically motivated bins; Spearman correlation per-dbpair per-bin."
        )
    )
    ap.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing *_harmonized.csv for CTRP, GDSC1, GDSC2, PRISM (full & filtered).",
    )
    ap.add_argument(
        "--out-root",
        required=True,
        help="Output root directory; script writes into <out-root>/full/<bin_tag>/ and filtered/.",
    )
    ap.add_argument(
        "--work-root",
        default=None,
        help="(Unused) kept for backwards compatibility; previously used for GDSC2 compilation.",
    )

    ap.add_argument(
        "--rep-agg",
        choices=["median", "mean"],
        default="median",
        help="How to aggregate replicates within a DB before cross-DB correlation.",
    )

    ap.add_argument(
        "--ic-clip-mode",
        choices=["window", "upper", "off"],
        default="window",
        help=(
            "How to handle IC/EC values (fit_ic50/fit_ec50), assumed log10(µM): "
            "'window' = drop values <1 nM or >100 µM; "
            "'upper' = drop values >100 µM only; "
            "'off' = no IC/EC filtering."
        ),
    )

    ap.add_argument(
        "--auc-transform",
        choices=["none", "logit", "logit_on_1_minus"],
        default="none",
        help=(
            "Transform to apply to AUC-like columns before correlation: "
            "'none', 'logit', or 'logit_on_1_minus'."
        ),
    )
    ap.add_argument(
        "--auc-eps",
        type=float,
        default=1e-6,
        help="Epsilon used to clamp AUCs away from 0 and 1 for the logit transform.",
    )

    ap.add_argument(
        "--bin-mode",
        choices=["none", "auc_lt_fixed", "auc_bottom_quantile", "ic_lt_fixed"],
        default="none",
        help=(
            "Binning strategy:\n"
            "  none               : single bin 'all'\n"
            "  auc_lt_fixed       : AUC metrics: _level < auc-threshold vs >=\n"
            "  auc_bottom_quantile: AUC metrics: bottom auc-quantile vs rest\n"
            "  ic_lt_fixed        : IC metrics: _level < ic-threshold vs >=\n"
        ),
    )
    ap.add_argument(
        "--auc-threshold",
        type=float,
        default=0.2,
        help="Threshold for AUC-based bins (e.g., 0.2).",
    )
    ap.add_argument(
        "--auc-quantile",
        type=float,
        default=0.2,
        help="Quantile (0–1) for 'auc_bottom_quantile' bin mode (e.g., 0.2).",
    )
    ap.add_argument(
        "--ic-threshold",
        type=float,
        default=0.0,
        help="Threshold for IC-based bins, in log10(µM) (0.0 = 1 µM).",
    )

    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    out_root = Path(args.out_root)
    work_root = Path(args.work_root) if args.work_root else out_root

    full_paths, filt_paths = _collect_four_dbs(input_dir, work_root)
    print("[info] FULL cohort selections:", {k: str(v.name) for k, v in full_paths.items()})
    print("[info] FILTERED cohort selections:", {k: str(v.name) for k, v in filt_paths.items()})

    if args.bin_mode == "none":
        bin_tag = "bin_all"
    elif args.bin_mode == "auc_lt_fixed":
        bin_tag = f"bin_auc_lt_{args.auc_threshold:g}"
    elif args.bin_mode == "auc_bottom_quantile":
        bin_tag = f"bin_auc_bottom_q{args.auc_quantile:g}"
    elif args.bin_mode == "ic_lt_fixed":
        bin_tag = f"bin_ic_lt_log10_{args.ic_threshold:g}"
    else:
        bin_tag = "bin_custom"

    _run_one_cohort_biobins(
        paths=full_paths,
        cohort_label="full",
        out_dir=out_root / "full" / bin_tag,
        rep_agg=args.rep_agg,
        auc_mode=args.auc_transform,
        auc_eps=args.auc_eps,
        bin_mode=args.bin_mode,
        auc_thresh=args.auc_threshold,
        auc_quantile=args.auc_quantile,
        ic_thresh=args.ic_threshold,
        ic_clip_mode=args.ic_clip_mode,
    )

    _run_one_cohort_biobins(
        paths=filt_paths,
        cohort_label="filtered",
        out_dir=out_root / "filtered" / bin_tag,
        rep_agg=args.rep_agg,
        auc_mode=args.auc_transform,
        auc_eps=args.auc_eps,
        bin_mode=args.bin_mode,
        auc_thresh=args.auc_threshold,
        auc_quantile=args.auc_quantile,
        ic_thresh=args.ic_threshold,
        ic_clip_mode=args.ic_clip_mode,
    )


if __name__ == "__main__":
    main()
