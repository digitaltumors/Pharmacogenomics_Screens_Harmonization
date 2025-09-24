import argparse, os, re
from pathlib import Path
from typing import Dict, List, Tuple, Literal
import numpy as np
import pandas as pd
from itertools import combinations
from scipy.stats import pearsonr, spearmanr

# --------- Aliases / normalizers ----------
DRUG_ALIASES = ["DrugID", "drug", "improve_drug_id"]
CELL_ALIASES = ["CellID", "cell_line", "improve_sample_id", "sample", "cell"]

def _find_col(df: pd.DataFrame, candidates: List[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of {candidates} found in dataframe.")

def _norm_id_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()

def _canon_one_df(df: pd.DataFrame, normalize_ids: bool=True) -> Tuple[pd.DataFrame, List[str]]:
    """Return df with: drug, cell, <metrics...>; also list of metric cols."""
    dcol = _find_col(df, DRUG_ALIASES)
    ccol = _find_col(df, CELL_ALIASES)
    out = df.rename(columns={dcol: "drug", ccol: "cell"}).copy()
    if normalize_ids:
        out["drug"] = _norm_id_series(out["drug"])
        out["cell"] = _norm_id_series(out["cell"])
    metric_cols = [c for c in out.columns if c not in ("drug","cell") and pd.api.types.is_numeric_dtype(out[c])]
    if not metric_cols:
        raise ValueError("No numeric metric columns found.")
    return out[["drug","cell"] + metric_cols], metric_cols

# --------- Biological clipping ----------
_IC50_RE = re.compile(r'(^|[^a-zA-Z0-9])ic50([^a-zA-Z0-9]|$)', re.IGNORECASE)
_EC50_RE = re.compile(r'(^|[^a-zA-Z0-9])ec50([^a-zA-Z0-9]|$)', re.IGNORECASE)

def _is_ic50_col(col: str) -> bool: return bool(_IC50_RE.search(str(col)))
def _is_ec50_col(col: str) -> bool: return bool(_EC50_RE.search(str(col)))

def apply_biological_clipping(df: pd.DataFrame, dict_label: str) -> pd.DataFrame:
    is_gx = (str(dict_label).strip() == "GX_sets")
    lo, hi = (0.001, 100.0) if is_gx else (-6.0, 6.0)
    out = df.copy()
    for col in out.columns:
        if _is_ic50_col(col) or _is_ec50_col(col):
            s = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            out[col] = s.clip(lower=lo, upper=hi)
    return out

# --------- Correlation helpers ----------
def _corr_xy(x: np.ndarray, y: np.ndarray, corr: Literal["pearson","spearman"]="pearson") -> float:
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return np.nan
    r, _ = (pearsonr if corr=="pearson" else spearmanr)(x[m], y[m])
    return float(np.clip(r, -0.999999, 0.999999))

def build_long_from_db_dict(db_dict: Dict[str, pd.DataFrame], normalize_ids=True) -> pd.DataFrame:
    canon = {}
    for db_name, df in db_dict.items():
        canon[db_name] = _canon_one_df(df, normalize_ids=normalize_ids)

    rows = []
    dbs = list(canon.keys())
    for i, j in combinations(range(len(dbs)), 2):
        db1, db2 = dbs[i], dbs[j]
        df1, mets1 = canon[db1]
        df2, mets2 = canon[db2]
        shared_metrics = sorted(set(mets1) & set(mets2))
        if not shared_metrics:
            continue
        merged = df1.merge(df2, on=["drug","cell"], suffixes=("_db1","_db2"))
        if merged.empty: continue
        for m in shared_metrics:
            col1, col2 = f"{m}_db1", f"{m}_db2"
            if col1 not in merged or col2 not in merged:
                continue
            sub = merged[["drug","cell", col1, col2]].dropna()
            if sub.empty: continue
            sub = sub.rename(columns={col1:"val_db1", col2:"val_db2"})
            sub["metric"] = m
            sub["db_pair"] = f"{db1}|{db2}"
            rows.append(sub)
    if not rows:
        raise ValueError("No overlaps across DB pairs / metrics.")
    return pd.concat(rows, ignore_index=True)

def evaluate_per_dbpair_correlations(
    db_dict: Dict[str, pd.DataFrame], corr="pearson", normalize_ids=True
) -> pd.DataFrame:
    df_long = build_long_from_db_dict(db_dict, normalize_ids)
    rows = []
    for (metric, db_pair), g in df_long.groupby(["metric","db_pair"]):
        r = _corr_xy(g["val_db1"].to_numpy(float), g["val_db2"].to_numpy(float), corr)
        rows.append({"metric": metric, "db_pair": db_pair, "n": len(g), "r": r})
    return pd.DataFrame(rows)

# --------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Evaluate per-DB correlations from CSV folder.")
    ap.add_argument("--csv-dir", required=True, help="Folder containing database CSV files.")
    ap.add_argument("--out-dir", required=True, help="Output folder.")
    ap.add_argument("--corr", default="pearson", choices=["pearson","spearman"])
    ap.add_argument("--no-normalize-ids", action="store_true")
    args = ap.parse_args()

    in_dir = Path(args.csv_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Load all CSVs in folder
    db_dict = {}
    for path in in_dir.glob("*.csv"):
        name = path.stem
        df = pd.read_csv(path)
        df = apply_biological_clipping(df, name)
        db_dict[name] = df

    res = evaluate_per_dbpair_correlations(
        db_dict, corr=args.corr, normalize_ids=(not args.no_normalize_ids)
    )
    res.to_csv(out_dir / "per_dbpair_correlations.csv", index=False)

if __name__ == "__main__":
    main()
