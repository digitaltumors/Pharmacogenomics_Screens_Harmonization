#!/usr/bin/env python3
import os
import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional

import gzip
import pickle  # kept only in case you later want backward-compat helpers
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

# ============================================================
# ID / metric helpers
# ============================================================
_ID_EXCLUDE = {
    "drug_canonical", "cell_canonical", "chem_name", "common_name",
    "improve_drug_id", "improve_sample_id",
    "canonical_drug", "canonical_cell"
}

def _get_cols(df: pd.DataFrame, drug_col: str | None = None, cell_col: str | None = None):
    """
    Pick drug / cell columns, preferring canonical_drug / canonical_cell,
    mirroring the harmonized schema.
    """
    cols = list(df.columns)

    # Drug
    if drug_col and drug_col in cols:
        dcol = drug_col
    elif "canonical_drug" in cols:
        dcol = "canonical_drug"
    elif "drug_canonical" in cols:
        dcol = "drug_canonical"
    else:
        dcol = cols[0]

    # Cell
    if cell_col and cell_col in cols:
        ccol = cell_col
    elif "canonical_cell" in cols:
        ccol = "canonical_cell"
    elif "cell_canonical" in cols:
        ccol = "cell_canonical"
    else:
        ccol = cols[1] if len(cols) > 1 else cols[0]

    return dcol, ccol

def _numeric_metric_cols(df: pd.DataFrame):
    num = [c for c in df.columns
           if c not in _ID_EXCLUDE and pd.api.types.is_numeric_dtype(df[c])]
    # also exclude the first two columns (ID-like)
    return [c for c in num if c not in {df.columns[0], df.columns[1]}]

def _metric_pairs(df1: pd.DataFrame, df2: pd.DataFrame):
    m1 = {c.lower(): c for c in _numeric_metric_cols(df1)}
    m2 = {c.lower(): c for c in _numeric_metric_cols(df2)}
    common = sorted(set(m1).intersection(m2))
    return [(m1[k], m2[k]) for k in common]

def _prep(
    df: pd.DataFrame,
    metric_col: str,
    agg: str = "mean",
    drug_col: str | None = None,
    cell_col: str | None = None,
):
    dcol, ccol = _get_cols(df, drug_col, cell_col)
    sub = df[[dcol, ccol, metric_col]].copy()
    sub[metric_col] = pd.to_numeric(sub[metric_col], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    sub = (
        sub.dropna(subset=[dcol, ccol, metric_col])
           .groupby([dcol, ccol], as_index=False)[metric_col]
           .agg(agg)
           .rename(columns={dcol: "drug", ccol: "cell", metric_col: "metric"})
    )
    return sub

def _corr_safe(x: np.ndarray, y: np.ndarray, corr: str = "pearson") -> float:
    """Return Pearson or Spearman r with basic safety checks; NaN if degenerate."""
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2:
        return np.nan
    xx, yy = x[m], y[m]
    # constant vector check (applies to both Pearson and Spearman)
    if np.isclose(np.std(xx), 0) or np.isclose(np.std(yy), 0):
        return np.nan
    if corr == "spearman":
        r, _ = spearmanr(xx, yy)
    else:
        r, _ = pearsonr(xx, yy)
    return float(r)

# ============================================================
# IC50/EC50 clipping (biological)
# ============================================================
_IC50_RE = re.compile(r'(^|[^a-zA-Z0-9])ic50([^a-zA-Z0-9]|$)', re.IGNORECASE)
_EC50_RE = re.compile(r'(^|[^a-zA-Z0-9])ec50([^a-zA-Z0-9]|$)', re.IGNORECASE)

def _is_ic50_col(col: str) -> bool:
    return bool(_IC50_RE.search(str(col)))

def _is_ec50_col(col: str) -> bool:
    return bool(_EC50_RE.search(str(col)))

def apply_biological_clipping_for_dict(
    db_dict: dict[str, pd.DataFrame],
    dict_label: str,
) -> dict[str, pd.DataFrame]:
    """
    Returns a NEW dict with IC50/EC50 clipped per rule:

      - if dict_label == 'GX_sets': IC50/EC50 -> [0.001, 100] (µM)
      - else:                        IC50/EC50 -> [-6, 6] (log10(M))

    For harmonized *_harmonized.csv use the second branch (log10(M) clipping).
    """
    is_gx = (str(dict_label).strip() == "GX_sets")
    lo, hi = (0.001, 100.0) if is_gx else (-6.0, 6.0)

    out: dict[str, pd.DataFrame] = {}
    for name, df in db_dict.items():
        tmp = df.copy()
        for col in tmp.columns:
            if _is_ic50_col(col) or _is_ec50_col(col):
                s = pd.to_numeric(tmp[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
                tmp[col] = s.clip(lower=lo, upper=hi)
        out[name] = tmp
    return out

# ============================================================
# Permutation routine for one DB pair
# ============================================================
def _pair_to_plot_dfs(
    db1: pd.DataFrame,
    db2: pd.DataFrame,
    *,
    db1_name: str,
    db2_name: str,
    corr: str = "pearson",
    permute_db: int = 1,
    n_perm: int = 10000,
    random_state: int = 0,
    min_pairs: int = 3,
    agg: str = "mean",
    drug_col: str | None = None,
    cell_col: str | None = None,
):
    rng = np.random.default_rng(random_state)
    obs_rows, null_rows = [], []

    for m1, m2 in _metric_pairs(db1, db2):
        A = _prep(db1, m1, agg=agg, drug_col=drug_col, cell_col=cell_col)
        B = _prep(db2, m2, agg=agg, drug_col=drug_col, cell_col=cell_col)
        metric_label = m1 if m1.lower() == m2.lower() else f"{m1} ~ {m2}"

        for drug in np.intersect1d(A["drug"].unique(), B["drug"].unique()):
            a = A[A["drug"] == drug].set_index("cell")["metric"]
            b = B[B["drug"] == drug].set_index("cell")["metric"]
            cells = a.index.intersection(b.index)
            if len(cells) < min_pairs:
                continue

            x = a.loc[cells].to_numpy(float)
            y = b.loc[cells].to_numpy(float)

            r_obs = _corr_safe(x, y, corr=corr)

            # permutations under the chosen correlation
            r_perm = np.empty(n_perm, dtype=float)
            for i in range(n_perm):
                if permute_db == 1:
                    r_perm[i] = _corr_safe(rng.permutation(x), y, corr=corr)
                else:
                    r_perm[i] = _corr_safe(x, rng.permutation(y), corr=corr)

            # summarize permutation null for a z-score (optional)
            r_perm_fin = r_perm[np.isfinite(r_perm)]
            null_mean = float(np.nan) if r_perm_fin.size == 0 else float(r_perm_fin.mean())
            null_std  = float(np.nan) if r_perm_fin.size <= 1 else float(r_perm_fin.std(ddof=1))
            if (
                not np.isfinite(r_obs)
                or not np.isfinite(null_mean)
                or not np.isfinite(null_std)
                or np.isclose(null_std, 0)
            ):
                z_score = np.nan
            else:
                z_score = (r_obs - null_mean) / null_std

            # observed row
            obs_rows.append({
                "db1": db1_name, "db2": db2_name,
                "permute_db": "db1" if permute_db == 1 else "db2",
                "metric_label": metric_label, "drug": drug,
                "corr": corr, "r": r_obs, "z_score": z_score, "n_pairs": int(len(cells)),
            })

            # null rows (skip NaNs)
            for i, v in enumerate(r_perm):
                if not np.isfinite(v):
                    continue
                null_rows.append({
                    "db1": db1_name,
                    "db2": db2_name,
                    "permute_db": "db1" if permute_db == 1 else "db2",
                    "perm_id": i,
                    "metric_label": metric_label,
                    "drug": drug,
                    "corr": corr,
                    "r_perm": float(v),
                })

    return pd.DataFrame(obs_rows), pd.DataFrame(null_rows)

# ============================================================
# Run over ALL DB pairs in a dictionary
# ============================================================
def make_plot_dfs_for_dictionary(
    data_dict: dict[str, pd.DataFrame],
    *,
    corr: str = "spearman",
    permute_target: str = "first",
    n_perm: int = 10000,
    random_state: int = 42,
    min_pairs: int = 3,
    agg: str = "mean",
    drug_col: str | None = None,
    cell_col: str | None = None,
    keep_pair_labels: bool = True,
):
    keys = list(data_dict.keys())
    obs_all, null_all = [], []
    permute_db = 1 if permute_target == "first" else 2

    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            k1, k2 = keys[i], keys[j]
            obs_df, null_df = _pair_to_plot_dfs(
                data_dict[k1], data_dict[k2],
                db1_name=k1, db2_name=k2,
                corr=corr,
                permute_db=permute_db, n_perm=n_perm, random_state=random_state,
                min_pairs=min_pairs, agg=agg, drug_col=drug_col, cell_col=cell_col,
            )
            if not obs_df.empty:
                obs_all.append(obs_df)
            if not null_df.empty:
                null_all.append(null_df)

    obs_out = (
        pd.concat(obs_all, ignore_index=True)
        if obs_all
        else pd.DataFrame(
            columns=[
                "db1", "db2", "permute_db", "metric_label",
                "drug", "corr", "r", "z_score", "n_pairs",
            ]
        )
    )

    null_out = (
        pd.concat(null_all, ignore_index=True)
        if null_all
        else pd.DataFrame(
            columns=[
                "db1", "db2", "permute_db", "perm_id", "metric_label",
                "drug", "corr", "r_perm",
            ]
        )
    )

    if not keep_pair_labels:
        obs_out = obs_out.drop(columns=["db1", "db2", "permute_db"], errors="ignore")
        null_out = null_out.drop(columns=["db1", "db2", "permute_db", "perm_id"], errors="ignore")

    return obs_out, null_out

# ============================================================
# Harmonized CSV loading + GDSC2 compilation (matches first script)
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
    # drop common index columns
    drop_cols = [c for c in df.columns if re.match(r"^Unnamed:\s*0+$", str(c), flags=re.I)]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    # try to coerce numeric columns where possible (for metrics)
    for c in df.columns:
        # best-effort numeric conversion; leave as string if many NaNs
        try:
            s = pd.to_numeric(df[c], errors="coerce")
            # if we got "reasonable" number of non-nan, keep numeric
            if s.notna().sum() > 0 and s.notna().sum() >= 0.1 * len(s):
                df[c] = s
        except Exception:
            continue
    return df

def _prefer_first_existing(input_dir: Path, names: List[str]) -> Optional[Path]:
    for nm in names:
        p = input_dir / nm
        if p.exists() and p.is_file() and not _is_resource_fork(p) and p.name.endswith("_harmonized.csv"):
            return p
    return None

def _compile_gdsc2(input_dir: Path, out_root: Path) -> Dict[str, Path]:
    """
    Mirror logic from the correlation script:
    combine GDSC2 new/old releases, preferring new on duplicates.
    """
    specs = [
        (
            "full",
            "GDSC2_NewRelease_replicates_AUC_harmonized.csv",
            "GDSC2_OldRelease_replicates_AUC_harmonized.csv",
            "GDSC2_Compile_replicates_AUC_harmonized.csv",
        ),
        (
            "filtered",
            "GDSC2_NewRelease_replicates_filtered_AUC_harmonized.csv",
            "GDSC2_OldRelease_replicates_filtered_AUC_harmonized.csv",
            "GDSC2_Compile_replicates_filtered_AUC_harmonized.csv",
        ),
    ]
    out: Dict[str, Path] = {}
    out_root.mkdir(parents=True, exist_ok=True)
    for scope, new_name, old_name, out_name in specs:
        new_p, old_p = input_dir / new_name, input_dir / old_name
        if not new_p.exists() and not old_p.exists():
            continue
        frames = []
        if new_p.exists():
            dnew = _read_csv(new_p)
            dnew["__prio__"] = 0
            frames.append(dnew)
        if old_p.exists():
            dold = _read_csv(old_p)
            dold["__prio__"] = 1
            frames.append(dold)
        df = pd.concat(frames, ignore_index=True)

        # prefer "new" on duplicates
        cell_key = "canonical_cell" if "canonical_cell" in df.columns else (
            "cell_canonical" if "cell_canonical" in df.columns else "cell"
        )
        drug_key = "canonical_drug" if "canonical_drug" in df.columns else (
            "drug_canonical" if "drug_canonical" in df.columns else "drug"
        )
        dup_keys = [cell_key, drug_key] + (["replicate"] if "replicate" in df.columns else [])
        df = (
            df.sort_values(["__prio__"] + dup_keys, kind="mergesort")
              .drop_duplicates(dup_keys, keep="first")
              .drop(columns="__prio__")
        )
        out_path = out_root / out_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        out[scope] = out_path
    return out

def _collect_four_dbs(input_dir: Path, work_root: Path) -> tuple[Dict[str, Path], Dict[str, Path]]:
    """
    Return (full_paths, filtered_paths) where each is a dict {DB_NAME: Path}
    for CTRP, GDSC1, compiled GDSC2, PRISM, matching the first script.
    """
    ctrp_full = _prefer_first_existing(
        input_dir,
        [
            "CTRP_postQC_harmonized.csv",
            "CTRP_harmonized.csv",
            "CTRP_replicates_AUC_harmonized.csv",
            "CTRPv2_harmonized.csv",
        ],
    )
    ctrp_filt = _prefer_first_existing(
        input_dir,
        [
            "CTRP_postQC_filtered_harmonized.csv",
            "CTRP_replicates_filtered_harmonized.csv",
            "CTRP_replicates_filtered_AUC_harmonized.csv",
        ],
    )
    gdsc1_full = _prefer_first_existing(input_dir, ["GDSC1_replicates_AUC_harmonized.csv"])
    gdsc1_filt = _prefer_first_existing(input_dir, ["GDSC1_replicates_Filtered_AUC_harmonized.csv"])

    compiled = _compile_gdsc2(input_dir, work_root)
    gdsc2_full = compiled.get("full")
    gdsc2_filt = compiled.get("filtered")

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

def _load_db_dict_from_paths(paths: Dict[str, Path]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for label, p in sorted(paths.items()):
        print(f"[load] {label}: {p.name}")
        out[label] = _read_csv(p)
    return out

# ============================================================
# CLI
# ============================================================
def main():
    p = argparse.ArgumentParser(
        description=(
            "Build per-drug permutation nulls for correlations between "
            "CTRP, GDSC1, GDSC2 (compiled), and PRISM from *_harmonized.csv. "
            "Uses the same file loading and GDSC2 combination as the main "
            "correlation script (full/filtered cohorts)."
        )
    )
    p.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing *_harmonized.csv inputs for CTRP, GDSC1, GDSC2, PRISM.",
    )
    p.add_argument(
        "--out-root",
        required=True,
        help="Output root directory where permutation results will be written.",
    )
    p.add_argument(
        "--work-root",
        default=None,
        help="Scratch dir for compiled GDSC2 (defaults to out-root if not provided).",
    )
    p.add_argument(
        "--scope",
        choices=["full", "filtered", "both"],
        default="both",
        help="Which cohort(s) to run: full, filtered, or both.",
    )
    p.add_argument(
        "--corr",
        choices=["pearson", "spearman"],
        default="pearson",
        help="Correlation coefficient to use for permutations.",
    )
    p.add_argument("--n-perm", type=int, default=1000)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument(
        "--permute-target",
        choices=["first", "second"],
        default="first",
        help="Which DB to shuffle in the null (first vs second).",
    )
    p.add_argument("--min-pairs", type=int, default=3)
    p.add_argument("--agg", default="mean", help="Aggregation across replicates (mean or median).")
    p.add_argument(
        "--keep-pair-labels",
        action="store_true",
        help="If set, keep db1/db2/permute_db columns in outputs.",
    )

    args = p.parse_args()

    input_dir = Path(args.input_dir)
    out_root = Path(args.out_root)
    work_root = Path(args.work_root) if args.work_root else out_root

    out_root.mkdir(parents=True, exist_ok=True)

    # Collect harmonized inputs with compiled GDSC2
    full_paths, filt_paths = _collect_four_dbs(input_dir, work_root)
    print("[info] FULL cohort selections:", {k: v.name for k, v in full_paths.items()})
    print("[info] FILTERED cohort selections:", {k: v.name for k, v in filt_paths.items()})

    scopes: list[tuple[str, Dict[str, Path]]] = []
    if args.scope in ("full", "both") and full_paths:
        scopes.append(("full", full_paths))
    if args.scope in ("filtered", "both") and filt_paths:
        scopes.append(("filtered", filt_paths))

    if not scopes:
        print("[warn] No usable cohorts (full/filtered) found; nothing to do.")
        return

    for scope_label, paths in scopes:
        if len(paths) < 2:
            print(f"[info] {scope_label}: only {len(paths)} DB(s); need ≥2. Skipping.")
            continue

        print(f"[run] scope={scope_label}")
        data_dict = _load_db_dict_from_paths(paths)

        # Biological clipping (for harmonized data -> log10(M) regime)
        data_dict = apply_biological_clipping_for_dict(data_dict, dict_label="harmonized_full")

        obs_df, null_df = make_plot_dfs_for_dictionary(
            data_dict,
            corr=args.corr,
            permute_target=args.permute_target,
            n_perm=args.n_perm,
            random_state=args.random_state,
            min_pairs=args.min_pairs,
            agg=args.agg,
            keep_pair_labels=args.keep_pair_labels,
        )

        out_dir = out_root / scope_label
        out_dir.mkdir(parents=True, exist_ok=True)
        obs_df.to_csv(out_dir / f"{scope_label}_obs.csv", index=False)
        null_df.to_csv(out_dir / f"{scope_label}_null.csv", index=False)
        print(f"[{scope_label}] corr={args.corr} rows: obs={len(obs_df)} null={len(null_df)}")

if __name__ == "__main__":
    main()
