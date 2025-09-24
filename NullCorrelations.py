import os, argparse, re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from itertools import combinations
from typing import Dict, List, Tuple

# ---------------- ID aliases & normalization ----------------
DRUG_ALIASES = ["DrugID", "drug", "improve_drug_id", "drug_canonical", "chem_name", "common_name"]
CELL_ALIASES = ["CellID", "cell_line", "improve_sample_id", "sample", "cell", "cell_canonical"]

def _find_col(df: pd.DataFrame, candidates: List[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of {candidates} found in dataframe columns: {list(df.columns)[:10]}...")

def _norm_id_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()

def _canon_one_df(df: pd.DataFrame, normalize_ids: bool=True) -> Tuple[pd.DataFrame, List[str]]:
    """
    Return df with columns: drug, cell, <numeric metrics...>; and list of metric cols.
    """
    dcol = _find_col(df, DRUG_ALIASES)
    ccol = _find_col(df, CELL_ALIASES)
    out = df.rename(columns={dcol: "drug", ccol: "cell"}).copy()
    if normalize_ids:
        out["drug"] = _norm_id_series(out["drug"])
        out["cell"] = _norm_id_series(out["cell"])
    metric_cols = [c for c in out.columns if c not in ("drug","cell") and pd.api.types.is_numeric_dtype(out[c])]
    if not metric_cols:
        raise ValueError("No numeric metric columns found (expected e.g. AUC, IC50, EC50).")
    return out[["drug","cell"] + metric_cols], metric_cols

# ---------------- Biological clipping (IC50/EC50) ----------------
_IC50_RE = re.compile(r'(^|[^a-zA-Z0-9])ic50([^a-zA-Z0-9]|$)', re.IGNORECASE)
_EC50_RE = re.compile(r'(^|[^a-zA-Z0-9])ec50([^a-zA-Z0-9]|$)', re.IGNORECASE)
def _is_ic50_col(col: str) -> bool: return bool(_IC50_RE.search(str(col)))
def _is_ec50_col(col: str) -> bool: return bool(_EC50_RE.search(str(col)))

def apply_biological_clipping(df: pd.DataFrame, label_for_rules: str) -> pd.DataFrame:
    """
    Clipping rule:
      - if label == 'GX_sets': IC50/EC50 -> [0.001, 100]  (µM)
      - else:                   IC50/EC50 -> [-6, 6]      (log10(M))
    Only columns that look like IC50/EC50 are clipped. IDs untouched.
    """
    is_gx = (str(label_for_rules).strip() == "GX_sets")
    lo, hi = (0.001, 100.0) if is_gx else (-6.0, 6.0)
    out = df.copy()
    for col in out.columns:
        if _is_ic50_col(col) or _is_ec50_col(col):
            s = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            out[col] = s.clip(lower=lo, upper=hi)
    return out

# ---------------- Metric pairing / prep ----------------
def _numeric_metric_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in ("drug","cell") and pd.api.types.is_numeric_dtype(df[c])]

def _metric_pairs(df1: pd.DataFrame, df2: pd.DataFrame) -> List[Tuple[str,str]]:
    m1 = {c.lower(): c for c in _numeric_metric_cols(df1)}
    m2 = {c.lower(): c for c in _numeric_metric_cols(df2)}
    common = sorted(set(m1).intersection(m2))
    return [(m1[k], m2[k]) for k in common]

def _prep(df: pd.DataFrame, metric_col: str, agg="mean") -> pd.DataFrame:
    sub = df[["drug", "cell", metric_col]].copy()
    sub[metric_col] = pd.to_numeric(sub[metric_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    sub = (
        sub.dropna(subset=["drug", "cell", metric_col])
           .groupby(["drug", "cell"], as_index=False)[metric_col]
           .agg(agg)
           .rename(columns={metric_col: "metric"})
    )
    return sub

def _pearson_safe(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2:
        return np.nan
    xx, yy = x[m], y[m]
    if np.isclose(np.std(xx), 0) or np.isclose(np.std(yy), 0):
        return np.nan
    return float(pearsonr(xx, yy)[0])

# ---------------- core per-pair permuter ----------------
def _pair_to_plot_dfs(
    db1: pd.DataFrame,
    db2: pd.DataFrame,
    *,
    db1_name: str,
    db2_name: str,
    permute_db: int = 1,
    n_perm: int = 10000,
    random_state: int = 0,
    min_pairs: int = 3,
    agg: str = "mean",
):
    rng = np.random.default_rng(random_state)
    obs_rows, null_rows = [], []

    for m1, m2 in _metric_pairs(db1, db2):
        A = _prep(db1, m1, agg=agg)
        B = _prep(db2, m2, agg=agg)
        metric_label = m1 if m1.lower() == m2.lower() else f"{m1} ~ {m2}"

        for drug in np.intersect1d(A["drug"].unique(), B["drug"].unique()):
            a = A[A["drug"] == drug].set_index("cell")["metric"]
            b = B[B["drug"] == drug].set_index("cell")["metric"]
            cells = a.index.intersection(b.index)
            if len(cells) < min_pairs:
                continue

            x = a.loc[cells].to_numpy(float)
            y = b.loc[cells].to_numpy(float)

            r_obs = _pearson_safe(x, y)

            # permutations (shuffle x or y)
            r_perm = np.empty(n_perm, dtype=float)
            if permute_db == 1:
                for i in range(n_perm):
                    r_perm[i] = _pearson_safe(rng.permutation(x), y)
            else:
                for i in range(n_perm):
                    r_perm[i] = _pearson_safe(x, rng.permutation(y))

            # summarize for z-score (optional)
            r_fin = r_perm[np.isfinite(r_perm)]
            null_mean = float(np.nan) if r_fin.size == 0 else float(r_fin.mean())
            null_std  = float(np.nan) if r_fin.size <= 1 else float(r_fin.std(ddof=1))
            z_score   = np.nan if (not np.isfinite(r_obs) or not np.isfinite(null_mean)
                                   or not np.isfinite(null_std) or np.isclose(null_std, 0)) \
                               else (r_obs - null_mean) / null_std

            obs_rows.append({
                "db1": db1_name, "db2": db2_name,
                "permute_db": "db1" if permute_db == 1 else "db2",
                "metric_label": metric_label, "drug": drug,
                "pearson_r": r_obs, "z_score": z_score, "n_pairs": int(len(cells)),
            })

            for i, v in enumerate(r_perm):
                if not np.isfinite(v):
                    continue
                null_rows.append({
                    "db1": db1_name, "db2": db2_name,
                    "permute_db": "db1" if permute_db == 1 else "db2",
                    "perm_id": i,
                    "metric_label": metric_label, "drug": drug,
                    "r_perm": float(v),
                })

    return pd.DataFrame(obs_rows), pd.DataFrame(null_rows)

# ---------------- run over ALL CSVs in a directory ----------------
def make_plot_dfs_for_csv_dir(
    csv_dir: Path,
    *,
    label_for_rules: str,
    permute_target: str = "first",
    n_perm: int = 10000,
    random_state: int = 42,
    min_pairs: int = 3,
    agg: str = "mean",
    normalize_ids: bool = True,
):
    # Load all CSVs -> {db_name: df}
    data_dict: Dict[str, pd.DataFrame] = {}
    for path in csv_dir.glob("*.csv"):
        name = path.stem
        df = pd.read_csv(path)
        df = apply_biological_clipping(df, label_for_rules=name if label_for_rules is None else label_for_rules)
        # Canonicalize and record metric columns (raises if none)
        df_canon, _ = _canon_one_df(df, normalize_ids=normalize_ids)
        data_dict[name] = df_canon

    keys = sorted(data_dict.keys())
    if len(keys) < 2:
        raise ValueError(f"Need at least 2 CSVs in {csv_dir} (found {len(keys)}).")

    obs_all, null_all = [], []
    permute_db = 1 if permute_target == "first" else 2

    for i, j in combinations(range(len(keys)), 2):
        k1, k2 = keys[i], keys[j]
        obs_df, null_df = _pair_to_plot_dfs(
            data_dict[k1], data_dict[k2],
            db1_name=k1, db2_name=k2,
            permute_db=permute_db, n_perm=n_perm, random_state=random_state,
            min_pairs=min_pairs, agg=agg
        )
        if not obs_df.empty:  obs_all.append(obs_df)
        if not null_df.empty: null_all.append(null_df)

    obs_out  = (pd.concat(obs_all, ignore_index=True)
                if obs_all else pd.DataFrame(columns=["db1","db2","permute_db","metric_label","drug","pearson_r","z_score","n_pairs"]))
    null_out = (pd.concat(null_all, ignore_index=True)
                if null_all else pd.DataFrame(columns=["db1","db2","permute_db","perm_id","metric_label","drug","r_perm"]))

    return obs_out, null_out

# ---------------- CLI ----------------
def main():
    p = argparse.ArgumentParser(description="Null permutation test from a folder of CSVs (per-drug, per-metric, per-DB pair).")
    p.add_argument("--csv-dir", required=True, help="Folder containing database CSV files.")
    p.add_argument("--out-root", required=True, help="Output root directory.")
    p.add_argument("--label", default=None, help="Label for output subfolder/files (default: csv-dir name).")
    p.add_argument("--n-perm", type=int, default=1000)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--permute-target", choices=["first","second"], default="first", help="Which DB to permute within each pair.")
    p.add_argument("--min-pairs", type=int, default=3, help="Minimum shared cells per drug to compute r.")
    p.add_argument("--agg", default="mean", help="Aggregation over duplicate (drug,cell) rows: e.g., mean, median.")
    p.add_argument("--no-normalize-ids", action="store_true", help="Disable lower/strip normalization of IDs.")
    p.add_argument("--clip-label", default=None, help="Override rule label for IC50/EC50 clipping (e.g., 'GX_sets').")
    args = p.parse_args()

    csv_dir = Path(args.csv_dir)
    if not csv_dir.is_dir():
        raise NotADirectoryError(f"--csv-dir not found: {csv_dir}")

    out_dir = Path(args.out_root) / (args.label if args.label else csv_dir.name)
    out_dir.mkdir(parents=True, exist_ok=True)

    obs_df, null_df = make_plot_dfs_for_csv_dir(
        csv_dir,
        label_for_rules=(args.clip_label if args.clip_label is not None else (args.label if args.label else csv_dir.name)),
        permute_target=args.permute_target,
        n_perm=args.n_perm,
        random_state=args.random_state,
        min_pairs=args.min_pairs,
        agg=args.agg,
        normalize_ids=(not args.no_normalize_ids),
    )

    obs_path  = out_dir / f"{args.label if args.label else csv_dir.name}_obs.csv"
    null_path = out_dir / f"{args.label if args.label else csv_dir.name}_null.csv"
    obs_df.to_csv(obs_path, index=False)
    null_df.to_csv(null_path, index=False)
    print(f"[done] obs_rows={len(obs_df)}  null_rows={len(null_df)}")
    print(f"saved: {obs_path}")
    print(f"saved: {null_path}")

if __name__ == "__main__":
    main()
