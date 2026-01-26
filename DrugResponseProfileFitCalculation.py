import pandas as pd
import numpy as np
from scipy.optimize import least_squares
from scipy.integrate import quad
from joblib import Parallel, delayed
import argparse
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, required=True)
parser.add_argument('--drug_col', type=str, default='drug_name')
parser.add_argument('--cell_line_col', type=str, default='cell_line')
parser.add_argument('--dose_col', type=str, default='dose_uM')
parser.add_argument('--viability_col', type=str, default='pct_viability')
parser.add_argument('--truncate', action='store_true', help='Truncate dose range to 0.03-10uM')
parser.add_argument('--force_ic50', action='store_true', help='Fix bottom asymptote to 0 to extrapolate IC50')
parser.add_argument('--output_path', type=str, required=True)

args = parser.parse_args()

# NOTE: PharmacoGx AUC calculation fits log-transformed concentrations to Hill slope equation with top asymptote fixed at 1

def hill_eq_vectorized(x, params):
    """Vectorized Hill equation."""
    bottom, ec50, nH = params
    return bottom + (1 - bottom) /  (1 + ((10**x) / (10**ec50))**nH)

def hill_eq_scalar(x, bottom, ec50, nH):
    """Scalar Hill equation for integration."""
    return bottom + (1 - bottom) /  (1 + ((10**x) / (10**ec50))**nH)
def residuals_vectorized(params, x, y):
    """Residuals for least squares."""
    return hill_eq_vectorized(x, params) - y

def fit_batch_of_pairs(pair_batch, data, cols):
    """Fit multiple pairs in one worker - reduces overhead."""
    results = []
    error_details = []
    
    for drug_id, cell_line in pair_batch:
        try:
            drug_data = data[(data[cols['drug']] == drug_id) & 
                            (data[cols['cell']] == cell_line)]

            if args.truncate:
                print("Truncating dose range!", flush=True)
                drug_data = drug_data[(drug_data[cols['dose']] >= 0.03) & (drug_data[cols['dose']] <= 10)]
                # Check if dose range is truncated
                if drug_data[cols['dose']].min() < 0.03 or drug_data[cols['dose']].max() > 10:
                    error_details.append('Truncate argument did not actually truncate dose range!')
                    continue
            
            x = np.log10(drug_data[cols['dose']].values)
            y = drug_data[cols['viability']].values
            
            # Sort
            sort_idx = np.argsort(x)
            x, y = x[sort_idx], y[sort_idx]
            
            # Initial guess
            # p0 = [y.min(), np.median(x), 1.0]
            ec50_guess = np.median(x)
            if ec50_guess < -6 or ec50_guess > 6:
                ec50_guess = 0  # Use middle of [-6, 6] range

            p0 = [
                np.clip(y.min(), 0, 1),
                ec50_guess,
                1.0
            ]
            
            # Bounds
            y_range = abs(y.max() - y.min())
            y_buffer = max(0.5 * y_range, 0.1)  # At least 0.1 buffer

            # NOTE: [bottom, log(ec50), nH] bounds taken from PharmacoGx git repo (2022)
            bounds_lower = [0,-6, 0]
            bounds_upper = [1, 6, 4]
            
            # Use least_squares (faster than curve_fit for this)
            result = least_squares(
                residuals_vectorized, p0, 
                args=(x, y),
                bounds=(bounds_lower, bounds_upper)
                # max_nfev=500,  # Reduced iterations
                # ftol=1e-6,
                # xtol=1e-6
            )
            
            if not result.success:
                error_details.append('optimization_failed')
                continue
            
            bottom, ec50, nH = result.x
            
            # Calculate IC50
            if (0.5 - bottom > 0):
                ic50 = ec50 + (1/nH) * np.log10(0.5/(0.5 - bottom))
                if ic50 > x.max() and not args.force_ic50:
                    ic50 = np.nan
            else:
                if args.force_ic50:
                    # Assume bottom would reach 0 at very high concentrations
                    projected_bottom = 0
                    ic50 = ec50 + (1/nH) * np.log10(0.5 / (0.5 - projected_bottom))
                else:
                    ic50 = np.nan
            
            # ic50 = ec50 * (((top - 0.5)/(0.5 - bottom))**(1/nH)) if (top - 0.5 > 0 and 0.5 - bottom > 0) else np.nan
            
            # Calculate metrics
            y_pred = hill_eq_vectorized(x, result.x)
            ss_res = np.sum((y - y_pred)**2)
            ss_tot = np.sum((y - y.mean())**2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            rmse = np.sqrt(ss_res / len(y))
            
            integrate_lower = float(x.min())
            integrate_upper = float(x.max())
            auc_trapz = np.trapz(y, x) / (integrate_upper - integrate_lower)

            # r2 = r2_score(ydata_sorted, y_pred)
            if bottom == 1:
                auc_sig=0
            elif nH == 0:
                auc_sig = (1-bottom) / 2
            else:
                auc_sig = quad(
                    lambda dose: hill_eq_scalar(dose, bottom, ec50, nH),
                    integrate_lower,
                    integrate_upper
                    # limit=50
                )[0] / (integrate_upper - integrate_lower)

            # auc_trapz = np.trapz(ydata_sorted, xdata_sorted)
            auc_sig_norm = quad(
                lambda dose: hill_eq_scalar(dose, bottom, ec50, nH),
                np.log10(0.03),
                np.log10(10)
                # limit=50
            )[0] / (np.log10(10) - np.log10(0.03))
            
            results.append({
                'Drug ID': drug_id,
                'Cell Line': cell_line,
                'Bottom': bottom,
                'log_EC50': ec50,
                'nH': nH,
                'log_IC50': ic50,
                'AUC_trapz': auc_trapz,
                'AUC_sig': auc_sig,
                'AUC_sig_norm': auc_sig_norm,
                'R2': r2,
                'RMSE': rmse
            })
        except Exception as e:
            error_details.append(f"{type(e).__name__}: {str(e)}")
            continue
    
    return results, error_details

# Main execution
print(f"Starting at {time.strftime('%H:%M:%S')}", flush=True)

# Read only what's needed for pairs
data_df = pd.read_csv(args.data)
pairs = data_df[[args.drug_col, args.cell_line_col]].drop_duplicates().values

print(f"Found {len(pairs)} pairs", flush=True)
print(f"Found {len(data_df[args.drug_col].unique())} drugs", flush=True)
print(f"Found {len(data_df[args.cell_line_col].unique())} cell lines", flush=True)

cols = {
    'drug': args.drug_col,
    'cell': args.cell_line_col,
    'dose': args.dose_col,
    'viability': args.viability_col
}

# Batch pairs for efficiency (each worker processes 100 pairs)
batch_size = 100
pair_batches = [pairs[i:i+batch_size] for i in range(0, len(pairs), batch_size)]

print(f"Processing {len(pair_batches)} batches with 64 cores", flush=True)
print(f"Started fitting at {time.strftime('%H:%M:%S')}", flush=True)
sys.stdout.flush()

all_results = Parallel(n_jobs=64, verbose=50)(
    delayed(fit_batch_of_pairs)(batch, data_df, cols) 
    for batch in pair_batches
)

# Separate results and errors
final_results = []
all_errors = []
for batch_results, batch_errors in all_results:
    final_results.extend(batch_results)
    all_errors.extend(batch_errors)

final_df = pd.DataFrame(final_results)

print(f"\nComplete at {time.strftime('%H:%M:%S')}", flush=True)
print(f"Successful fits: {len(final_df)}/{len(pairs)}", flush=True)

# Show first 10 unique error messages
print("\nFirst 10 unique errors:", flush=True)
unique_errors = list(set(all_errors))[:10]
for error in unique_errors:
    count = all_errors.count(error)
    print(f"  {error}: {count} occurrences", flush=True)

if len(final_df) > 0:
    final_df.to_csv(args.output_path, index=False)
else:
    print("WARNING: No successful fits - not writing output file", flush=True)
