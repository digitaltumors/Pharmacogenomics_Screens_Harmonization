import os
import argparse
import warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional 

from sklearn.metrics import r2_score
from scipy.optimize import curve_fit
from scipy.optimize import OptimizeWarning

# --------------------------------
# Bounds & window
# --------------------------------
#HS_BOUNDS = ([0, 0, -4], [1, 12, 4])  # einf, ec50, hs
HS_BOUNDS = ([0, -10, -4], [1, 12, 4])  # einf, ec50, hs
WINDOW_LO = np.log10(0.03)            # ~ -1.522879
WINDOW_HI = np.log10(10.0)            # 1.0

# ==============================================
# Overflow-safe model implementations
# ==============================================
_LN10 = np.log(10.0)

def _sigma_neg_z(z):
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    if np.any(pos):
        zp = z[pos]
        out[pos] = np.exp(-zp) / (1.0 + np.exp(-zp))
    if np.any(~pos):
        zn = z[~pos]
        out[~pos] = 1.0 / (1.0 + np.exp(zn))
    return out

def response_curve(x, einf, ec50, hs):
    x = np.asarray(x, dtype=float)
    z = (ec50 - x) * hs * _LN10
    return einf + (1.0 - einf) * _sigma_neg_z(z)

def response_integral(x, einf, ec50, hs):
    x = np.asarray(x, dtype=float)
    t = (ec50 - x) * hs
    u = t * _LN10
    softplus_u = np.maximum(u, 0.0) + np.log1p(np.exp(-np.abs(u)))
    log10_term = softplus_u / _LN10
    if np.isscalar(hs):
        hs_arr = np.array([hs], dtype=float)
    else:
        hs_arr = np.asarray(hs, dtype=float)
    if np.all(np.abs(hs_arr) > 1e-12):
        return (1.0 - einf) * (log10_term / hs) + x
    else:
        xx = np.asarray(x, dtype=float)
        if xx.ndim == 0:
            return float(x)
        yy = response_curve(xx, einf, ec50, hs)
        area = np.cumsum((yy[:-1] + yy[1:]) * np.diff(xx) * 0.5)
        out = np.empty_like(xx, dtype=float)
        out[0] = xx[0]
        out[1:] = xx[0] + area
        return out

# --------------------------------
# Helpers
# --------------------------------
def hill_ic_at_y(einf, ec50, hs, y_target=0.5):
    if hs == 0:
        return np.nan
    denom = (y_target - einf)
    if denom <= 0 or (1 - einf) <= 0:
        return np.nan
    rhs = (1 - einf) / denom - 1.0
    if rhs <= 0:
        return np.nan
    return ec50 - (np.log10(rhs) / hs)

def _safe_fit_auc(x1, x2, params):
    einf_, ec50_, hs_ = params
    if np.isfinite(hs_) and abs(hs_) > 1e-12:
        return (response_integral(x2, *params) - response_integral(x1, *params)) / (x2 - x1)
    xx = np.linspace(x1, x2, 200)
    yy = response_curve(xx, *params)
    return np.trapz(yy, xx) / (x2 - x1)

def _to_numeric_array(a):
    if isinstance(a, pd.Series):
        return pd.to_numeric(a, errors="coerce").to_numpy(dtype=float)
    try:
        return np.asarray(a, dtype=float)
    except Exception:
        return pd.to_numeric(pd.Series(a), errors="coerce").to_numpy(dtype=float)

def _clean_xy(x, y):
    x = _to_numeric_array(x)
    y = _to_numeric_array(y)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]; y = y[mask]
    if y.size and np.nanmax(y) > 2:
        y = y / 100.0
    y = np.clip(y, 0.0, 1.0)
    return x, y

def empirical_auc_basic(xs, ys):
    xs = _to_numeric_array(xs)
    ys = _to_numeric_array(ys)
    m = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[m], ys[m]
    if ys.size and np.nanmax(ys) > 2:
        ys = ys / 100.0
    ys = np.clip(ys, 0.0, 1.0)
    if xs.size < 2:
        return np.nan
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    ux, inv = np.unique(xs, return_inverse=True)
    if ux.size < 2:
        return np.nan
    y_med = np.empty_like(ux, dtype=float)
    for k in range(ux.size):
        y_med[k] = np.median(ys[inv == k])
    return np.trapz(y_med, ux) / (ux[-1] - ux[0])

# --------------------------------
# Fitting
# --------------------------------
def response_curve_fit(
    xdata,
    ydata,
    bounds=HS_BOUNDS,
    use_lowdose_anchor=False,
    anchor_offset_log10=0.5,
    anchor_sigma=0.15,
    anchor_x_fixed: Optional[float] = None,   # NEW: fixed anchor x in log10(µM)
):
    xdata = np.asarray(xdata, dtype=float)
    ydata = np.asarray(ydata, dtype=float)
    finite = np.isfinite(xdata) & np.isfinite(ydata)
    xdata = xdata[finite]
    ydata = np.clip(ydata[finite], 0.0, 1.0)

    if xdata.size < 2:
        return None, None

    # -------- dynamic ec50 bounds from data --------
    xmin = float(np.nanmin(xdata))
    xmax = float(np.nanmax(xdata))

    if (not np.isfinite(xmin)) or (not np.isfinite(xmax)) or xmax <= xmin:
        # bail out if dose axis is degenerate
        return None, None

    pad = 0.25
    ec50_lo = xmin - pad
    ec50_hi = min(xmax + pad, np.log10(12.0))

    # ensure ec50_lo < ec50_hi
    if ec50_hi <= ec50_lo:
        # widen artificially around the middle if something weird happens
        mid = 0.5 * (xmin + xmax)
        ec50_lo = mid - 1.0
        ec50_hi = mid + 1.0

    # base bounds from HS_BOUNDS (for einf and hs)
    base_lb = np.array(bounds[0], dtype=float)
    base_ub = np.array(bounds[1], dtype=float)

    # build per-curve bounds: einf fixed by HS_BOUNDS, ec50 from data, hs from HS_BOUNDS
    lb = np.array([base_lb[0], ec50_lo, base_lb[2]], dtype=float)
    ub = np.array([base_ub[0], ec50_hi, base_ub[2]], dtype=float)

    # final safety: enforce lb < ub with a tiny margin
    eps = 1e-8
    if np.any(lb + eps >= ub):
        return None, None

    nfev = 300
    initial_guess = [0.5, float(np.mean(xdata)), -1.0]
    initial_guess = np.minimum(np.maximum(initial_guess, lb + eps), ub - eps)

    # ---- anchor logic ----
    if use_lowdose_anchor and xdata.size >= 1 and np.isfinite(np.nanmin(xdata)):
        if anchor_x_fixed is not None:
            # FIXED anchor in log10(µM), same units as xdata
            x_anchor = float(anchor_x_fixed)
        else:
            # OLD behavior: place anchor 0.5 log10 units below lowest dose
            x_anchor = float(np.nanmin(xdata)) - float(anchor_offset_log10)

        x_aug = np.r_[xdata, x_anchor]
        y_aug = np.r_[ydata, 1.0]
        sigma = np.r_[np.ones_like(xdata, dtype=float), float(anchor_sigma)]
    else:
        x_aug = xdata
        y_aug = ydata
        sigma = None

    popt, pcov = None, None
    while popt is None and nfev <= 10000:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", OptimizeWarning)
                popt, pcov = curve_fit(
                    response_curve, x_aug, y_aug,
                    p0=initial_guess,
                    bounds=(lb, ub),        # <<< IMPORTANT: use our lb/ub, not HS_BOUNDS
                    max_nfev=nfev,
                    sigma=sigma,
                    absolute_sigma=False,
                )
        except RuntimeError:
            pass
        nfev *= 2

    return popt, pcov

# --------------------------------
# Metrics
# --------------------------------
FIT_COLS = [
    'fit_auc','fit_ic50','fit_ec50','fit_ec50se','fit_r2','fit_rmse','fit_einf','fit_hs',
    'aac','auc','dss','auc_emp','fit_auc_win','fit_ic50_win','fit_ec50_win'
]

def compute_fit_metrics(xdata, ydata, popt, pcov):
    def _na_series(auc_emp_val=np.nan):
        d = {c: np.nan for c in FIT_COLS}
        if np.isfinite(auc_emp_val):
            d['auc_emp'] = np.round(auc_emp_val, 4)
        return pd.Series(d, index=FIT_COLS)

    auc_emp = empirical_auc_basic(xdata, ydata)

    if popt is None or pcov is None or (np.ndim(pcov) == 2 and np.any(np.isnan(np.diag(pcov)))):
        return _na_series(auc_emp)

    einf, ec50, hs = popt
    perr = np.sqrt(np.diag(pcov)) if np.ndim(pcov) == 2 else np.array([np.nan, np.nan, np.nan])
    ec50se = perr[1] if perr.size > 1 else np.nan

    xs = np.asarray(xdata, dtype=float); ys = np.asarray(ydata, dtype=float)
    if xs.size < 2 or not np.all(np.isfinite(xs)):
        return _na_series(auc_emp)
    xmin = float(np.nanmin(xs)); xmax = float(np.nanmax(xs))
    if not np.isfinite(xmin) or not np.isfinite(xmax) or xmax == xmin:
        return _na_series(auc_emp)

    ypred = response_curve(xs, *popt)
    r2 = np.nan; rmse = np.nan
    finite_mask = np.isfinite(ys) & np.isfinite(ypred)
    if np.any(finite_mask):
        ys_f = ys[finite_mask]; ypred_f = ypred[finite_mask]
        y_span = float(np.nanmax(ys_f) - np.nanmin(ys_f))
        if y_span >= 1e-6:
            r2 = r2_score(ys_f, ypred_f)
        rmse = float(np.sqrt(np.mean((ys_f - ypred_f) ** 2)))

    auc_fit_span = _safe_fit_auc(xmin, xmax, popt)
    xx = np.linspace(xmin, xmax, 200); yy = response_curve(xx, *popt)
    auc_trapz_fit = np.trapz(yy, xx) / (xmax - xmin)
    aac1 = 1 - auc_trapz_fit

    ic50 = hill_ic_at_y(einf, ec50, hs, 0.5) if einf < 0.5 else np.nan
    ic10 = hill_ic_at_y(einf, ec50, hs, 0.1) if einf < 0.9 else np.nan
    ic10x = np.nanmin([ic10, xmax]) if np.isfinite(ic10) else xmax
    dss1 = (auc_trapz_fit - 0.1*(ic10x - xmin)) / (0.9 * (xmax - xmin)) if xmax > ec50 else 0
    dss2 = dss1 / (1 - einf) if (1 - einf) != 0 else np.nan

    lo = WINDOW_LO; hi = WINDOW_HI
    fit_auc_win = _safe_fit_auc(lo, hi, popt) if hi > lo else np.nan
    ic50_win = hill_ic_at_y(einf, ec50, hs, 0.5)
    if (not np.isfinite(ic50_win)) or (ic50_win < lo) or (ic50_win > hi):
        ic50_win = np.nan
    ec50_win = ec50 if (lo <= ec50 <= hi) else np.nan

    return pd.Series({
        'fit_auc': auc_fit_span,
        'fit_ic50': ic50,
        'fit_ec50': ec50,
        'fit_ec50se': ec50se,
        'fit_r2': r2,
        'fit_rmse': rmse,
        'fit_einf': einf,
        'fit_hs': hs,
        'aac': aac1,
        'auc': auc_trapz_fit,
        'dss': dss2,
        'auc_emp': auc_emp,
        'fit_auc_win': fit_auc_win,
        'fit_ic50_win': ic50_win,
        'fit_ec50_win': ec50_win,
    }).round(4)

# --------------------------------
# Config & logic (no globals)
# --------------------------------
@dataclass
class FlowConfig:
    hs_thresh: float = -0.3
    r2_low: float = 0.2
    r2_high: float = 0.8
    rmse_pctl: float = 75.0
    rmse_material: float = 0.01
    anchor_offset: float = 0.5
    anchor_sigma: float = 0.15
    # NEW: fixed anchor location in log10(µM); if None, use old "min − offset" logic
    anchor_log10uM: Optional[float] = None

def _should_try_anchor(b: pd.Series, cfg: FlowConfig) -> bool:
    return (np.isfinite(b['fit_hs']) and (b['fit_hs'] < cfg.hs_thresh)
            and np.isfinite(b['fit_r2']) and (cfg.r2_low <= b['fit_r2'] <= cfg.r2_high)
            and bool(b.get('_rmse_is_high', False)))

def _accept_anchor(b: pd.Series, a: pd.Series, cfg: FlowConfig) -> bool:
    d_rmse = a['fit_rmse'] - b['fit_rmse']
    d_r2   = a['fit_r2']   - b['fit_r2']
    win_ic50_gain = (not np.isfinite(b['fit_ic50_win'])) and np.isfinite(a['fit_ic50_win'])
    win_ec50_gain = (not np.isfinite(b['fit_ec50_win'])) and np.isfinite(a['fit_ec50_win'])
    return (np.isfinite(d_rmse) and d_rmse <= -cfg.rmse_material and (not np.isfinite(d_r2) or d_r2 >= 0)) \
           or win_ic50_gain or win_ec50_gain

# --------------------------------
# Batch processor with auto-anchor
# --------------------------------
def process_df_auto(df, cfg: FlowConfig, group_cols=('cell_line', 'drug', 'replicate')):
    effective_group_cols = [c for c in group_cols if c in df.columns]
    groups = df.groupby(effective_group_cols, dropna=False) if effective_group_cols else [((), df)]

    # Pass 1: baseline fits
    baseline_rows = []
    xys_cache = {}
    for name, g in groups:
        x, y = _clean_xy(g["dose"], g["response"])
        xys_cache[name] = (x, y)
        if x.size < 1 or y.size < 1:
            b = pd.Series({c: np.nan for c in FIT_COLS}, index=FIT_COLS)
        elif x.size < 3 or np.nanstd(y) == 0:
            einf = float(np.nanmean(y)) if y.size else 0.5
            ec50 = float(np.nanmean(x)) if x.size else 0.0
            hs = 0.0
            b = compute_fit_metrics(x, y, (einf, ec50, hs), None)
        else:
            popt, pcov = response_curve_fit(x, y, use_lowdose_anchor=False)
            if popt is None:
                b = pd.Series({c: np.nan for c in FIT_COLS}, index=FIT_COLS)
            else:
                b = compute_fit_metrics(x, y, popt, pcov)
        baseline_rows.append((name, b))

    # Determine high-RMSE threshold from baseline
    bdf = pd.DataFrame([dict(b) for _, b in baseline_rows])
    rmse_thresh = np.nanpercentile(bdf["fit_rmse"], cfg.rmse_pctl) if "fit_rmse" in bdf else np.nan

    # Collect rows as dicts (safe)
    rows = []

    for (name, b) in baseline_rows:
        b = b.copy()
        b['_rmse_is_high'] = (np.isfinite(b['fit_rmse']) and np.isfinite(rmse_thresh) and (b['fit_rmse'] >= rmse_thresh))
        try_anchor = _should_try_anchor(b, cfg)

        x, y = xys_cache[name]
        a = None
        decision = "anchor_off"
        reason = "baseline_kept"

        if try_anchor and x.size >= 3 and np.nanstd(y) > 0:
            poptA, pcovA = response_curve_fit(
                x, y,
                use_lowdose_anchor=True,
                anchor_offset_log10=cfg.anchor_offset,
                anchor_sigma=cfg.anchor_sigma,
                anchor_x_fixed=cfg.anchor_log10uM,   # NEW: fixed anchor x if provided
            )
            if poptA is not None:
                a = compute_fit_metrics(x, y, poptA, pcovA)
                if _accept_anchor(b, a, cfg):
                    decision = "anchor_on"
                    reason = "improved"
                else:
                    reason = "no_improvement"
            else:
                reason = "anchor_fit_failed"
        elif try_anchor:
            reason = "insufficient_points_or_variance"

        final = a if (decision == "anchor_on" and a is not None) else b

        # ---- Build row as a dict (no length mismatch) ----
        row = {}

        # keys
        if effective_group_cols:
            if isinstance(name, tuple):
                for k, v in zip(effective_group_cols, name):
                    row[k] = v
            else:
                row[effective_group_cols[0]] = name

        # final metrics (FIT_COLS order)
        for k in FIT_COLS:
            row[k] = final.get(k, np.nan)

        # diagnostics
        row.update({
            'anchor_decision': decision,
            'anchor_reason': reason,
            'rmse_percentile_threshold': float(rmse_thresh) if np.isfinite(rmse_thresh) else np.nan,
            'baseline_rmse_is_high': bool(b.get('_rmse_is_high', False)),
            'baseline_fit_anchor': 0,
            'anchor_soft_offset': cfg.anchor_offset,
            'anchor_soft_sigma': cfg.anchor_sigma,
            'anchor_fixed_log10uM': cfg.anchor_log10uM,  # NEW: record fixed anchor
            'delta_fit_rmse': (a['fit_rmse'] - b['fit_rmse']) if (a is not None and np.isfinite(b['fit_rmse'])) else np.nan,
            'delta_fit_r2':   (a['fit_r2']   - b['fit_r2'])   if (a is not None and np.isfinite(b['fit_r2']))   else np.nan,
        })

        # full baseline / anchor prefixed metrics (for reproducibility)
        for k in FIT_COLS:
            row[f"baseline_{k}"] = b.get(k, np.nan)
        for k in FIT_COLS:
            row[f"anchor_{k}"] = (a.get(k, np.nan) if a is not None else np.nan)

        # final choice flag
        row['fit_anchor'] = 1 if decision == "anchor_on" else 0

        rows.append(row)

    # Build DataFrame; enforce a stable column order
    # 1) keys
    col_order = list(effective_group_cols)
    # 2) final metrics
    col_order += FIT_COLS
    # 3) diagnostics
    diag_cols = [
        'anchor_decision','anchor_reason','rmse_percentile_threshold',
        'baseline_rmse_is_high','baseline_fit_anchor',
        'anchor_soft_offset','anchor_soft_sigma','anchor_fixed_log10uM',
        'delta_fit_rmse','delta_fit_r2'
    ]
    col_order += diag_cols
    # 4) baseline_*, anchor_* blocks
    col_order += [f"baseline_{k}" for k in FIT_COLS]
    col_order += [f"anchor_{k}"   for k in FIT_COLS]
    # 5) final flag
    col_order += ['fit_anchor']

    out_df = pd.DataFrame(rows)
    # Reindex to the desired order; include any accidental extras at the end
    extras = [c for c in out_df.columns if c not in col_order]
    out_df = out_df.reindex(columns=col_order + extras)

    return out_df

# --------------------------------
# CLI / main
# --------------------------------
def main():
    p = argparse.ArgumentParser(description="Dose–response metrics with auto-anchoring (flow-chart, no globals).")
    # Single-file or manifest
    p.add_argument("-i", "--input", help="Single input CSV")
    p.add_argument("-o", "--output", help="Single output CSV")

    p.add_argument("--manifest", help="TSV/CSV/TXT manifest with: input_path, output_path[, group_cols]")
    p.add_argument("--line-index", type=int, help="0-based index into the manifest")

    # Grouping
    p.add_argument("--group-cols", nargs="+", default=["cell_line","drug","replicate"])

    # Flow/anchor knobs (defaults match prior discussion)
    p.add_argument("--hs-thresh", type=float, default=-0.3)
    p.add_argument("--r2-low", type=float, default=0.2)
    p.add_argument("--r2-high", type=float, default=0.8)
    p.add_argument("--rmse-pctl", type=float, default=75.0)
    p.add_argument("--rmse-material", type=float, default=0.01)
    p.add_argument("--anchor-offset", type=float, default=0.5)
    p.add_argument("--anchor-sigma", type=float, default=0.15)
    # NEW: fixed anchor dose in log10(µM)
    p.add_argument(
        "--anchor-log10uM", type=float, default=None,
        help="Fixed low-dose anchor location in log10(µM). "
             "If set, overrides dynamic minDose-0.5 behaviour for anchored fits."
    )

    args = p.parse_args()

    cfg = FlowConfig(
        hs_thresh=args.hs_thresh,
        r2_low=args.r2_low, r2_high=args.r2_high,
        rmse_pctl=args.rmse_pctl,
        rmse_material=args.rmse_material,
        anchor_offset=args.anchor_offset,
        anchor_sigma=args.anchor_sigma,
        anchor_log10uM=args.anchor_log10uM,  # NEW
    )

    def _ensure_parent_dir(path):
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)

    def _run_one(in_path, out_path, group_cols):
        if not os.path.exists(in_path):
            raise FileNotFoundError(f"Input not found: {in_path}")
        df = pd.read_csv(in_path)
        missing = [c for c in ["dose","response"] if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required column(s) {missing} in {in_path}")

        out = process_df_auto(df, cfg=cfg, group_cols=tuple(group_cols))
        _ensure_parent_dir(out_path)
        out.to_csv(out_path, index=False)
        print(
            f"[done] auto-anchor | hs<{cfg.hs_thresh} | r2 in [{cfg.r2_low},{cfg.r2_high}] | "
            f"rmse≥p{cfg.rmse_pctl} | keep if ΔRMSE≤-{cfg.rmse_material} & ΔR2≥0 | "
            f"offset={cfg.anchor_offset} sigma={cfg.anchor_sigma} | "
            f"anchor_log10uM={cfg.anchor_log10uM} | "
            f"groups={len(out)} | {in_path} -> {out_path}"
        )

    # Single-file mode
    if args.input and args.output and not args.manifest:
        _run_one(args.input, args.output, args.group_cols)
        return

    # Manifest mode
    if not args.manifest or args.line_index is None:
        raise SystemExit("Provide --manifest and --line-index (or use -i/-o for single-file mode).")

    def _is_header_row(row: str) -> bool:
        sep = "\t" if ("\t" in row) else ","
        parts = [p.strip() for p in row.split(sep)]
        return (len(parts) >= 2) and (parts[0].lower() == "input_path")

    lines = []
    with open(args.manifest, "r") as fh:
        for raw in fh:
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            if _is_header_row(s):
                continue
            lines.append(s)

    if args.line_index < 0 or args.line_index >= len(lines):
        raise SystemExit(f"--line-index {args.line_index} out of range (0..{len(lines)-1})")

    row = lines[args.line_index]
    sep = "\t" if ("\t" in row) else ","
    parts = [p.strip() for p in row.split(sep)]
    if len(parts) < 2:
        raise SystemExit("Manifest row must have at least input_path and output_path")

    in_path = parts[0]; out_path = parts[1]
    if len(parts) >= 3 and parts[2]:
        group_cols = [c.strip() for c in parts[2].split(",") if c.strip()]
    else:
        group_cols = args.group_cols

    _run_one(in_path, out_path, group_cols)

if __name__ == "__main__":
    main()
