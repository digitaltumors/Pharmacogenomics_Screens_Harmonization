from joblib import Parallel, delayed
import scipy as sp
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--prism_public_full', type=str, required=True)
parser.add_argument('--mts021_full', type=str, required=True)
parser.add_argument('--mts022_full', type=str, required=True)
parser.add_argument('--mts023_full', type=str, required=True)
parser.add_argument('--mts024_full', type=str, required=True)
parser.add_argument('--mts025_full', type=str, required=True)
parser.add_argument('--mts026_full', type=str, required=True)
parser.add_argument('--prism_public_trunc', type=str, required=True)
parser.add_argument('--mts021_trunc', type=str, required=True)
parser.add_argument('--mts022_trunc', type=str, required=True)
parser.add_argument('--mts023_trunc', type=str, required=True)
parser.add_argument('--mts024_trunc', type=str, required=True)
parser.add_argument('--mts025_trunc', type=str, required=True)
parser.add_argument('--mts026_trunc', type=str, required=True)
parser.add_argument('--output_path', type=str, required=True)

args = parser.parse_args()

prism_public_full = pd.read_csv(args.prism_public_full)
mts021_full = pd.read_csv(args.mts021_full)
mts022_full = pd.read_csv(args.mts022_full)
mts023_full = pd.read_csv(args.mts023_full)
mts024_full = pd.read_csv(args.mts024_full)
mts025_full = pd.read_csv(args.mts025_full)
mts026_full = pd.read_csv(args.mts026_full)
prism_public_trunc = pd.read_csv(args.prism_public_trunc)
mts021_trunc = pd.read_csv(args.mts021_trunc)
mts022_trunc = pd.read_csv(args.mts022_trunc)
mts023_trunc = pd.read_csv(args.mts023_trunc)
mts024_trunc = pd.read_csv(args.mts024_trunc)
mts025_trunc = pd.read_csv(args.mts025_trunc)
mts026_trunc = pd.read_csv(args.mts026_trunc)


def bootstrap_spearman_pair(df, col1, col2, n_resamples=500):
    """Efficient bootstrap with reduced resamples"""
    corr = sp.stats.spearmanr(df[col1], df[col2], alternative='greater', nan_policy='raise')
    bs = sp.stats.bootstrap(
        (df[col1].values, df[col2].values),  # Convert to numpy arrays
        lambda x, y: sp.stats.spearmanr(x, y)[0], 
        paired=True, 
        n_resamples=n_resamples,
        method='percentile',  # Faster than BCa
        random_state=42
    )
    return corr[0], bs.confidence_interval.low, bs.confidence_interval.high

def process_one_comparison(db_pair, type_drug, type_dose, response, df, col1, col2, n_resamples=500):
    corr, ci_low, ci_high = bootstrap_spearman_pair(df, col1, col2, n_resamples)
    return {
        'db_pair': db_pair,
        'type_drug': type_drug,
        'type_dose': type_dose,
        'response': response,
        'spearman_corr': corr,
        '95ci_low': ci_low,
        '95ci_high': ci_high,
        'n_unique_drugs': df['Drug ID'].nunique(),
        'n_unique_cells': df['Cell Line'].nunique(),
        'pairs': len(df)
    }

prism_mts021_full = pd.merge(prism_public_full, mts021_full, on = ['Drug ID', 'Cell Line'], suffixes=('_prism', '_mts021'))
prism_mts022_full = pd.merge(prism_public_full, mts022_full, on = ['Drug ID', 'Cell Line'], suffixes=('_prism', '_mts022'))
prism_mts023_full = pd.merge(prism_public_full, mts023_full, on = ['Drug ID', 'Cell Line'], suffixes=('_prism', '_mts023'))
prism_mts024_full = pd.merge(prism_public_full, mts024_full, on = ['Drug ID', 'Cell Line'], suffixes=('_prism', '_mts024'))
prism_mts025_full = pd.merge(prism_public_full, mts025_full, on = ['Drug ID', 'Cell Line'], suffixes=('_prism', '_mts025'))
prism_mts026_full = pd.merge(prism_public_full, mts026_full, on = ['Drug ID', 'Cell Line'], suffixes=('_prism', '_mts026'))
mts021_mts022_full = pd.merge(mts021_full, mts022_full, on = ['Drug ID', 'Cell Line'], suffixes=('_mts021', '_mts022'))
mts021_mts023_full = pd.merge(mts021_full, mts023_full, on = ['Drug ID', 'Cell Line'], suffixes=('_mts021', '_mts023'))
mts021_mts024_full = pd.merge(mts021_full, mts024_full, on = ['Drug ID', 'Cell Line'], suffixes=('_mts021', '_mts024'))
mts021_mts025_full = pd.merge(mts021_full, mts025_full, on = ['Drug ID', 'Cell Line'], suffixes=('_mts021', '_mts025'))
mts021_mts026_full = pd.merge(mts021_full, mts026_full, on = ['Drug ID', 'Cell Line'], suffixes=('_mts021', '_mts026'))
mts022_mts023_full = pd.merge(mts022_full, mts023_full, on = ['Drug ID', 'Cell Line'], suffixes=('_mts022', '_mts023'))
mts022_mts024_full = pd.merge(mts022_full, mts024_full, on = ['Drug ID', 'Cell Line'], suffixes=('_mts022', '_mts024'))
mts022_mts025_full = pd.merge(mts022_full, mts025_full, on = ['Drug ID', 'Cell Line'], suffixes=('_mts022', '_mts025'))
mts022_mts026_full = pd.merge(mts022_full, mts026_full, on = ['Drug ID', 'Cell Line'], suffixes=('_mts022', '_mts026'))
mts023_mts024_full = pd.merge(mts023_full, mts024_full, on = ['Drug ID', 'Cell Line'], suffixes=('_mts023', '_mts024'))
mts023_mts025_full = pd.merge(mts023_full, mts025_full, on = ['Drug ID', 'Cell Line'], suffixes=('_mts023', '_mts025'))
mts023_mts026_full = pd.merge(mts023_full, mts026_full, on = ['Drug ID', 'Cell Line'], suffixes=('_mts023', '_mts026'))
mts024_mts025_full = pd.merge(mts024_full, mts025_full, on = ['Drug ID', 'Cell Line'], suffixes=('_mts024', '_mts025'))
mts024_mts026_full = pd.merge(mts024_full, mts026_full, on = ['Drug ID', 'Cell Line'], suffixes=('_mts024', '_mts026'))
mts025_mts026_full = pd.merge(mts025_full, mts026_full, on = ['Drug ID', 'Cell Line'], suffixes=('_mts025', '_mts026'))

prism_mts021_trunc = pd.merge(prism_public_trunc, mts021_trunc, on = ['Drug ID', 'Cell Line'], suffixes=('_prism', '_mts021'))
prism_mts022_trunc = pd.merge(prism_public_trunc, mts022_trunc, on = ['Drug ID', 'Cell Line'], suffixes=('_prism', '_mts022'))
prism_mts023_trunc = pd.merge(prism_public_trunc, mts023_trunc, on = ['Drug ID', 'Cell Line'], suffixes=('_prism', '_mts023'))
prism_mts024_trunc = pd.merge(prism_public_trunc, mts024_trunc, on = ['Drug ID', 'Cell Line'], suffixes=('_prism', '_mts024'))
prism_mts025_trunc = pd.merge(prism_public_full, mts025_trunc, on = ['Drug ID', 'Cell Line'], suffixes=('_prism', '_mts025'))
prism_mts026_trunc = pd.merge(prism_public_trunc, mts026_trunc, on = ['Drug ID', 'Cell Line'], suffixes=('_prism', '_mts026'))
mts021_mts022_trunc = pd.merge(mts021_trunc, mts022_trunc, on = ['Drug ID', 'Cell Line'], suffixes=('_mts021', '_mts022'))
mts021_mts023_trunc = pd.merge(mts021_trunc, mts023_trunc, on = ['Drug ID', 'Cell Line'], suffixes=('_mts021', '_mts023'))
mts021_mts024_trunc = pd.merge(mts021_trunc, mts024_trunc, on = ['Drug ID', 'Cell Line'], suffixes=('_mts021', '_mts024'))
mts021_mts025_trunc = pd.merge(mts021_trunc, mts025_trunc, on = ['Drug ID', 'Cell Line'], suffixes=('_mts021', '_mts025'))
mts021_mts026_trunc = pd.merge(mts021_trunc, mts026_trunc, on = ['Drug ID', 'Cell Line'], suffixes=('_mts021', '_mts026'))
mts022_mts023_trunc = pd.merge(mts022_trunc, mts023_trunc, on = ['Drug ID', 'Cell Line'], suffixes=('_mts022', '_mts023'))
mts022_mts024_trunc = pd.merge(mts022_trunc, mts024_trunc, on = ['Drug ID', 'Cell Line'], suffixes=('_mts022', '_mts024'))
mts022_mts025_trunc = pd.merge(mts022_trunc, mts025_trunc, on = ['Drug ID', 'Cell Line'], suffixes=('_mts022', '_mts025'))
mts022_mts026_trunc = pd.merge(mts022_trunc, mts026_trunc, on = ['Drug ID', 'Cell Line'], suffixes=('_mts022', '_mts026'))
mts023_mts024_trunc = pd.merge(mts023_trunc, mts024_trunc, on = ['Drug ID', 'Cell Line'], suffixes=('_mts023', '_mts024'))
mts023_mts025_trunc = pd.merge(mts023_trunc, mts025_trunc, on = ['Drug ID', 'Cell Line'], suffixes=('_mts023', '_mts025'))
mts023_mts026_trunc = pd.merge(mts023_trunc, mts026_trunc, on = ['Drug ID', 'Cell Line'], suffixes=('_mts023', '_mts026'))
mts024_mts025_trunc = pd.merge(mts024_trunc, mts025_trunc, on = ['Drug ID', 'Cell Line'], suffixes=('_mts024', '_mts025'))
mts024_mts026_trunc = pd.merge(mts024_trunc, mts026_trunc, on = ['Drug ID', 'Cell Line'], suffixes=('_mts024', '_mts026'))
mts025_mts026_trunc = pd.merge(mts025_trunc, mts026_trunc, on = ['Drug ID', 'Cell Line'], suffixes=('_mts025', '_mts026'))

# Define all comparisons as tuples
all_comparisons = [
    # PRISM vs MTS021 - All, full-range
    ('PRISM_21', 'All', 'Full', 'EC50', prism_mts021_full, 'log_EC50_prism', 'log_EC50_mts021'),
    ('PRISM_21', 'All', 'Full', 'IC50', prism_mts021_full, 'log_IC50_prism', 'log_IC50_mts021'),
    ('PRISM_21', 'All', 'Full', 'AUC_sig', prism_mts021_full, 'AUC_sig_prism', 'AUC_sig_mts021'),
    ('PRISM_21', 'All', 'Full', 'AUC_trapz', prism_mts021_full, 'AUC_trapz_prism', 'AUC_trapz_mts021'),
    ('PRISM_21', 'All', 'Full', 'AUC_sig_trunc', prism_mts021_full, 'AUC_sig_norm_prism', 'AUC_sig_norm_mts021'),
    # PRISM vs MTS021 - All, truncated-range    
    ('PRISM_21', 'All', 'Truncated', 'EC50', prism_mts021_trunc, 'log_EC50_prism', 'log_EC50_mts021'),
    ('PRISM_21', 'All', 'Truncated', 'IC50', prism_mts021_trunc, 'log_IC50_prism', 'log_IC50_mts021'),
    ('PRISM_21', 'All', 'Truncated', 'AUC_sig', prism_mts021_trunc, 'AUC_sig_prism', 'AUC_sig_mts021'),
    ('PRISM_21', 'All', 'Truncated', 'AUC_trapz', prism_mts021_trunc, 'AUC_trapz_prism', 'AUC_trapz_mts021'),
    ('PRISM_21', 'All', 'Truncated', 'AUC_sig_trunc', prism_mts021_trunc, 'AUC_sig_norm_prism', 'AUC_sig_norm_mts021'),
    # PRISM vs MTS022 - All, full-range
    ('PRISM_22', 'All', 'Full', 'EC50', prism_mts022_full, 'log_EC50_prism', 'log_EC50_mts022'),
    ('PRISM_22', 'All', 'Full', 'IC50', prism_mts022_full, 'log_IC50_prism', 'log_IC50_mts022'),
    ('PRISM_22', 'All', 'Full', 'AUC_sig', prism_mts022_full, 'AUC_sig_prism', 'AUC_sig_mts022'),
    ('PRISM_22', 'All', 'Full', 'AUC_trapz', prism_mts022_full, 'AUC_trapz_prism', 'AUC_trapz_mts022'),
    ('PRISM_22', 'All', 'Full', 'AUC_sig_trunc', prism_mts022_full, 'AUC_sig_norm_prism', 'AUC_sig_norm_mts022'),
    # PRISM vs MTS022 - All, truncated-range    
    ('PRISM_22', 'All', 'Truncated', 'EC50', prism_mts022_trunc, 'log_EC50_prism', 'log_EC50_mts022'),
    ('PRISM_22', 'All', 'Truncated', 'IC50', prism_mts022_trunc, 'log_IC50_prism', 'log_IC50_mts022'),
    ('PRISM_22', 'All', 'Truncated', 'AUC_sig', prism_mts022_trunc, 'AUC_sig_prism', 'AUC_sig_mts022'),
    ('PRISM_22', 'All', 'Truncated', 'AUC_trapz', prism_mts022_trunc, 'AUC_trapz_prism', 'AUC_trapz_mts022'),
    ('PRISM_22', 'All', 'Truncated', 'AUC_sig_trunc', prism_mts022_trunc, 'AUC_sig_norm_prism', 'AUC_sig_norm_mts022'),
    # PRISM vs MTS023 - All, full-range
    ('PRISM_23', 'All', 'Full', 'EC50', prism_mts023_full, 'log_EC50_prism', 'log_EC50_mts023'),
    ('PRISM_23', 'All', 'Full', 'IC50', prism_mts023_full, 'log_IC50_prism', 'log_IC50_mts023'),
    ('PRISM_23', 'All', 'Full', 'AUC_sig', prism_mts023_full, 'AUC_sig_prism', 'AUC_sig_mts023'),
    ('PRISM_23', 'All', 'Full', 'AUC_trapz', prism_mts023_full, 'AUC_trapz_prism', 'AUC_trapz_mts023'),
    ('PRISM_23', 'All', 'Full', 'AUC_sig_trunc', prism_mts023_full, 'AUC_sig_norm_prism', 'AUC_sig_norm_mts023'),
    # PRISM vs MTS023 - All, truncated-range
    ('PRISM_23', 'All', 'Truncated', 'EC50', prism_mts023_trunc, 'log_EC50_prism', 'log_EC50_mts023'),
    ('PRISM_23', 'All', 'Truncated', 'IC50', prism_mts023_trunc, 'log_IC50_prism', 'log_IC50_mts023'),
    ('PRISM_23', 'All', 'Truncated', 'AUC_sig', prism_mts023_trunc, 'AUC_sig_prism', 'AUC_sig_mts023'),
    ('PRISM_23', 'All', 'Truncated', 'AUC_trapz', prism_mts023_trunc, 'AUC_trapz_prism', 'AUC_trapz_mts023'),
    ('PRISM_23', 'All', 'Truncated', 'AUC_sig_trunc', prism_mts023_trunc, 'AUC_sig_norm_prism', 'AUC_sig_norm_mts023'),
     # PRISM vs MTS024 - All, full-range    
    ('PRISM_24', 'All', 'Full', 'EC50', prism_mts024_full, 'log_EC50_prism', 'log_EC50_mts024'),
    ('PRISM_24', 'All', 'Full', 'IC50', prism_mts024_full, 'log_IC50_prism', 'log_IC50_mts024'),
    ('PRISM_24', 'All', 'Full', 'AUC_sig', prism_mts024_full, 'AUC_sig_prism', 'AUC_sig_mts024'),
    ('PRISM_24', 'All', 'Full', 'AUC_trapz', prism_mts024_full, 'AUC_trapz_prism', 'AUC_trapz_mts024'),
    ('PRISM_24', 'All', 'Full', 'AUC_sig_trunc', prism_mts024_full, 'AUC_sig_norm_prism', 'AUC_sig_norm_mts024'),
    # PRISM vs MTS024 - All, truncated-range    
    ('PRISM_24', 'All', 'Truncated', 'EC50', prism_mts024_trunc, 'log_EC50_prism', 'log_EC50_mts024'),
    ('PRISM_24', 'All', 'Truncated', 'IC50', prism_mts024_trunc, 'log_IC50_prism', 'log_IC50_mts024'),
    ('PRISM_24', 'All', 'Truncated', 'AUC_sig', prism_mts024_trunc, 'AUC_sig_prism', 'AUC_sig_mts024'),
    ('PRISM_24', 'All', 'Truncated', 'AUC_trapz', prism_mts024_trunc, 'AUC_trapz_prism', 'AUC_trapz_mts024'),
    ('PRISM_24', 'All', 'Truncated', 'AUC_sig_trunc', prism_mts024_trunc, 'AUC_sig_norm_prism', 'AUC_sig_norm_mts024'),
    # PRISM vs MTS025 - All, full-range    
    ('PRISM_25', 'All', 'Full', 'EC50', prism_mts025_full, 'log_EC50_prism', 'log_EC50_mts025'),
    ('PRISM_25', 'All', 'Full', 'IC50', prism_mts025_full, 'log_IC50_prism', 'log_IC50_mts025'),
    ('PRISM_25', 'All', 'Full', 'AUC_sig', prism_mts025_full, 'AUC_sig_prism', 'AUC_sig_mts025'),
    ('PRISM_25', 'All', 'Full', 'AUC_trapz', prism_mts025_full, 'AUC_trapz_prism', 'AUC_trapz_mts025'),
    ('PRISM_25', 'All', 'Full', 'AUC_sig_trunc', prism_mts025_full, 'AUC_sig_norm_prism', 'AUC_sig_norm_mts025'),
    # PRISM vs MTS025 - All, truncated-range    
    ('PRISM_25', 'All', 'Truncated', 'EC50', prism_mts025_trunc, 'log_EC50_prism', 'log_EC50_mts025'),
    ('PRISM_25', 'All', 'Truncated', 'IC50', prism_mts025_trunc, 'log_IC50_prism', 'log_IC50_mts025'),
    ('PRISM_25', 'All', 'Truncated', 'AUC_sig', prism_mts025_trunc, 'AUC_sig_prism', 'AUC_sig_mts025'),
    ('PRISM_25', 'All', 'Truncated', 'AUC_trapz', prism_mts025_trunc, 'AUC_trapz_prism', 'AUC_trapz_mts025'),
    ('PRISM_25', 'All', 'Truncated', 'AUC_sig_trunc', prism_mts025_trunc, 'AUC_sig_norm_prism', 'AUC_sig_norm_mts025'),
    # PRISM vs MTS026 - All, full-range    
    ('PRISM_26', 'All', 'Full', 'EC50', prism_mts026_full, 'log_EC50_prism', 'log_EC50_mts026'),
    ('PRISM_26', 'All', 'Full', 'IC50', prism_mts026_full, 'log_IC50_prism', 'log_IC50_mts026'),
    ('PRISM_26', 'All', 'Full', 'AUC_sig', prism_mts026_full, 'AUC_sig_prism', 'AUC_sig_mts026'),
    ('PRISM_26', 'All', 'Full', 'AUC_trapz', prism_mts026_full, 'AUC_trapz_prism', 'AUC_trapz_mts026'),
    ('PRISM_26', 'All', 'Full', 'AUC_sig_trunc', prism_mts026_full, 'AUC_sig_norm_prism', 'AUC_sig_norm_mts026'),
    # PRISM vs MTS026 - All, truncated-range    
    ('PRISM_26', 'All', 'Truncated', 'EC50', prism_mts026_trunc, 'log_EC50_prism', 'log_EC50_mts026'),
    ('PRISM_26', 'All', 'Truncated', 'IC50', prism_mts026_trunc, 'log_IC50_prism', 'log_IC50_mts026'),
    ('PRISM_26', 'All', 'Truncated', 'AUC_sig', prism_mts026_trunc, 'AUC_sig_prism', 'AUC_sig_mts026'),
    ('PRISM_26', 'All', 'Truncated', 'AUC_trapz', prism_mts026_trunc, 'AUC_trapz_prism', 'AUC_trapz_mts026'),
    ('PRISM_26', 'All', 'Truncated', 'AUC_sig_trunc', prism_mts026_trunc, 'AUC_sig_norm_prism', 'AUC_sig_norm_mts026'),
    # MTS021 vs MTS022 - All, full-range    
    ('21_22', 'All', 'Full', 'EC50', mts021_mts022_full, 'log_EC50_mts021', 'log_EC50_mts022'),
    ('21_22', 'All', 'Full', 'IC50', mts021_mts022_full, 'log_IC50_mts021', 'log_IC50_mts022'),
    ('21_22', 'All', 'Full', 'AUC_sig', mts021_mts022_full, 'AUC_sig_mts021', 'AUC_sig_mts022'),
    ('21_22', 'All', 'Full', 'AUC_trapz', mts021_mts022_full, 'AUC_trapz_mts021', 'AUC_trapz_mts022'),
    ('21_22', 'All', 'Full', 'AUC_sig_trunc', mts021_mts022_full, 'AUC_sig_norm_mts021', 'AUC_sig_norm_mts022'),
    # MTS021 vs MTS022 - All, truncated-range    
    ('21_22', 'All', 'Truncated', 'EC50', mts021_mts022_trunc, 'log_EC50_mts021', 'log_EC50_mts022'),
    ('21_22', 'All', 'Truncated', 'IC50', mts021_mts022_trunc, 'log_IC50_mts021', 'log_IC50_mts022'),
    ('21_22', 'All', 'Truncated', 'AUC_sig', mts021_mts022_trunc, 'AUC_sig_mts021', 'AUC_sig_mts022'),
    ('21_22', 'All', 'Truncated', 'AUC_trapz', mts021_mts022_trunc, 'AUC_trapz_mts021', 'AUC_trapz_mts022'),
    ('21_22', 'All', 'Truncated', 'AUC_sig_trunc', mts021_mts022_trunc, 'AUC_sig_norm_mts021', 'AUC_sig_norm_mts022'),
    # MTS021 vs MTS023 - All, full-range    
    ('21_23', 'All', 'Full', 'EC50', mts021_mts023_full, 'log_EC50_mts021', 'log_EC50_mts023'),
    ('21_23', 'All', 'Full', 'IC50', mts021_mts023_full, 'log_IC50_mts021', 'log_IC50_mts023'),
    ('21_23', 'All', 'Full', 'AUC_sig', mts021_mts023_full, 'AUC_sig_mts021', 'AUC_sig_mts023'),
    ('21_23', 'All', 'Full', 'AUC_trapz', mts021_mts023_full, 'AUC_trapz_mts021', 'AUC_trapz_mts023'),
    ('21_23', 'All', 'Full', 'AUC_sig_trunc', mts021_mts023_full, 'AUC_sig_norm_mts021', 'AUC_sig_norm_mts023'),
    # MTS021 vs MTS023 - All, truncated-range    
    ('21_23', 'All', 'Truncated', 'EC50', mts021_mts023_trunc, 'log_EC50_mts021', 'log_EC50_mts023'),
    ('21_23', 'All', 'Truncated', 'IC50', mts021_mts023_trunc, 'log_IC50_mts021', 'log_IC50_mts023'),
    ('21_23', 'All', 'Truncated', 'AUC_sig', mts021_mts023_trunc, 'AUC_sig_mts021', 'AUC_sig_mts023'),
    ('21_23', 'All', 'Truncated', 'AUC_trapz', mts021_mts023_trunc, 'AUC_trapz_mts021', 'AUC_trapz_mts023'),
    ('21_23', 'All', 'Truncated', 'AUC_sig_trunc', mts021_mts023_trunc, 'AUC_sig_norm_mts021', 'AUC_sig_norm_mts023'),
    # MTS021 vs MTS024 - All, full-range    
    ('21_24', 'All', 'Full', 'EC50', mts021_mts024_full, 'log_EC50_mts021', 'log_EC50_mts024'),
    ('21_24', 'All', 'Full', 'IC50', mts021_mts024_full, 'log_IC50_mts021', 'log_IC50_mts024'),
    ('21_24', 'All', 'Full', 'AUC_sig', mts021_mts024_full, 'AUC_sig_mts021', 'AUC_sig_mts024'),
    ('21_24', 'All', 'Full', 'AUC_trapz', mts021_mts024_full, 'AUC_trapz_mts021', 'AUC_trapz_mts024'),
    ('21_24', 'All', 'Full', 'AUC_sig_trunc', mts021_mts024_full, 'AUC_sig_norm_mts021', 'AUC_sig_norm_mts024'),
    # MTS021 vs MTS024 - All, truncated-range    
    ('21_24', 'All', 'Truncated', 'EC50', mts021_mts024_trunc, 'log_EC50_mts021', 'log_EC50_mts024'),
    ('21_24', 'All', 'Truncated', 'IC50', mts021_mts024_trunc, 'log_IC50_mts021', 'log_IC50_mts024'),
    ('21_24', 'All', 'Truncated', 'AUC_sig', mts021_mts024_trunc, 'AUC_sig_mts021', 'AUC_sig_mts024'),
    ('21_24', 'All', 'Truncated', 'AUC_trapz', mts021_mts024_trunc, 'AUC_trapz_mts021', 'AUC_trapz_mts024'),
    ('21_24', 'All', 'Truncated', 'AUC_sig_trunc', mts021_mts024_trunc, 'AUC_sig_norm_mts021', 'AUC_sig_norm_mts024'),
    # MTS021 vs MTS025 - All, full-range    
    ('21_25', 'All', 'Full', 'EC50', mts021_mts025_full, 'log_EC50_mts021', 'log_EC50_mts025'),
    ('21_25', 'All', 'Full', 'IC50', mts021_mts025_full, 'log_IC50_mts021', 'log_IC50_mts025'),
    ('21_25', 'All', 'Full', 'AUC_sig', mts021_mts025_full, 'AUC_sig_mts021', 'AUC_sig_mts025'),
    ('21_25', 'All', 'Full', 'AUC_trapz', mts021_mts025_full, 'AUC_trapz_mts021', 'AUC_trapz_mts025'),
    ('21_25', 'All', 'Full', 'AUC_sig_trunc', mts021_mts025_full, 'AUC_sig_norm_mts021', 'AUC_sig_norm_mts025'),
    # MTS021 vs MTS025 - All, truncated-range    
    ('21_25', 'All', 'Truncated', 'EC50', mts021_mts025_trunc, 'log_EC50_mts021', 'log_EC50_mts025'),
    ('21_25', 'All', 'Truncated', 'IC50', mts021_mts025_trunc, 'log_IC50_mts021', 'log_IC50_mts025'),
    ('21_25', 'All', 'Truncated', 'AUC_sig', mts021_mts025_trunc, 'AUC_sig_mts021', 'AUC_sig_mts025'),
    ('21_25', 'All', 'Truncated', 'AUC_trapz', mts021_mts025_trunc, 'AUC_trapz_mts021', 'AUC_trapz_mts025'),
    ('21_25', 'All', 'Truncated', 'AUC_sig_trunc', mts021_mts025_trunc, 'AUC_sig_norm_mts021', 'AUC_sig_norm_mts025'),
    # MTS021 vs MTS026 - All, full-range    
    ('21_26', 'All', 'Full', 'EC50', mts021_mts026_full, 'log_EC50_mts021', 'log_EC50_mts026'),
    ('21_26', 'All', 'Full', 'IC50', mts021_mts026_full, 'log_IC50_mts021', 'log_IC50_mts026'),
    ('21_26', 'All', 'Full', 'AUC_sig', mts021_mts026_full, 'AUC_sig_mts021', 'AUC_sig_mts026'),
    ('21_26', 'All', 'Full', 'AUC_trapz', mts021_mts026_full, 'AUC_trapz_mts021', 'AUC_trapz_mts026'),
    ('21_26', 'All', 'Full', 'AUC_sig_trunc', mts021_mts026_full, 'AUC_sig_norm_mts021', 'AUC_sig_norm_mts026'),
    # MTS021 vs MTS026 - All, truncated-range    
    ('21_26', 'All', 'Truncated', 'EC50', mts021_mts026_trunc, 'log_EC50_mts021', 'log_EC50_mts026'),
    ('21_26', 'All', 'Truncated', 'IC50', mts021_mts026_trunc, 'log_IC50_mts021', 'log_IC50_mts026'),
    ('21_26', 'All', 'Truncated', 'AUC_sig', mts021_mts026_trunc, 'AUC_sig_mts021', 'AUC_sig_mts026'),
    ('21_26', 'All', 'Truncated', 'AUC_trapz', mts021_mts026_trunc, 'AUC_trapz_mts021', 'AUC_trapz_mts026'),
    ('21_26', 'All', 'Truncated', 'AUC_sig_trunc', mts021_mts026_trunc, 'AUC_sig_norm_mts021', 'AUC_sig_norm_mts026'),
    # MTS022 vs MTS023 - All, full-range    
    ('22_23', 'All', 'Full', 'EC50', mts022_mts023_full, 'log_EC50_mts022', 'log_EC50_mts023'),
    ('22_23', 'All', 'Full', 'IC50', mts022_mts023_full, 'log_IC50_mts022', 'log_IC50_mts023'),
    ('22_23', 'All', 'Full', 'AUC_sig', mts022_mts023_full, 'AUC_sig_mts022', 'AUC_sig_mts023'),
    ('22_23', 'All', 'Full', 'AUC_trapz', mts022_mts023_full, 'AUC_trapz_mts022', 'AUC_trapz_mts023'),
    ('22_23', 'All', 'Full', 'AUC_sig_trunc', mts022_mts023_full, 'AUC_sig_norm_mts022', 'AUC_sig_norm_mts023'),
    # MTS022 vs MTS023 - All, truncated-range    
    ('22_23', 'All', 'Truncated', 'EC50', mts022_mts023_trunc, 'log_EC50_mts022', 'log_EC50_mts023'),
    ('22_23', 'All', 'Truncated', 'IC50', mts022_mts023_trunc, 'log_IC50_mts022', 'log_IC50_mts023'),
    ('22_23', 'All', 'Truncated', 'AUC_sig', mts022_mts023_trunc, 'AUC_sig_mts022', 'AUC_sig_mts023'),
    ('22_23', 'All', 'Truncated', 'AUC_trapz', mts022_mts023_trunc, 'AUC_trapz_mts022', 'AUC_trapz_mts023'),
    ('22_23', 'All', 'Truncated', 'AUC_sig_trunc', mts022_mts023_trunc, 'AUC_sig_norm_mts022', 'AUC_sig_norm_mts023'),
    # MTS022 vs MTS024 - All, full-range    
    ('22_24', 'All', 'Full', 'EC50', mts022_mts024_full, 'log_EC50_mts022', 'log_EC50_mts024'),
    ('22_24', 'All', 'Full', 'IC50', mts022_mts024_full, 'log_IC50_mts022', 'log_IC50_mts024'),
    ('22_24', 'All', 'Full', 'AUC_sig', mts022_mts024_full, 'AUC_sig_mts022', 'AUC_sig_mts024'),
    ('22_24', 'All', 'Full', 'AUC_trapz', mts022_mts024_full, 'AUC_trapz_mts022', 'AUC_trapz_mts024'),
    ('22_24', 'All', 'Full', 'AUC_sig_trunc', mts022_mts024_full, 'AUC_sig_norm_mts022', 'AUC_sig_norm_mts024'),
    # MTS022 vs MTS024 - All, truncated-range    
    ('22_24', 'All', 'Truncated', 'EC50', mts022_mts024_trunc, 'log_EC50_mts022', 'log_EC50_mts024'),
    ('22_24', 'All', 'Truncated', 'IC50', mts022_mts024_trunc, 'log_IC50_mts022', 'log_IC50_mts024'),
    ('22_24', 'All', 'Truncated', 'AUC_sig', mts022_mts024_trunc, 'AUC_sig_mts022', 'AUC_sig_mts024'),
    ('22_24', 'All', 'Truncated', 'AUC_trapz', mts022_mts024_trunc, 'AUC_trapz_mts022', 'AUC_trapz_mts024'),
    ('22_24', 'All', 'Truncated', 'AUC_sig_trunc', mts022_mts024_trunc, 'AUC_sig_norm_mts022', 'AUC_sig_norm_mts024'),
    # MTS022 vs MTS025 - All, full-range    
    ('22_25', 'All', 'Full', 'EC50', mts022_mts025_full, 'log_EC50_mts022', 'log_EC50_mts025'),
    ('22_25', 'All', 'Full', 'IC50', mts022_mts025_full, 'log_IC50_mts022', 'log_IC50_mts025'),
    ('22_25', 'All', 'Full', 'AUC_sig', mts022_mts025_full, 'AUC_sig_mts022', 'AUC_sig_mts025'),
    ('22_25', 'All', 'Full', 'AUC_trapz', mts022_mts025_full, 'AUC_trapz_mts022', 'AUC_trapz_mts025'),
    ('22_25', 'All', 'Full', 'AUC_sig_trunc', mts022_mts025_full, 'AUC_sig_norm_mts022', 'AUC_sig_norm_mts025'),
    # MTS022 vs MTS025 - All, truncated-range    
    ('22_25', 'All', 'Truncated', 'EC50', mts022_mts025_trunc, 'log_EC50_mts022', 'log_EC50_mts025'),
    ('22_25', 'All', 'Truncated', 'IC50', mts022_mts025_trunc, 'log_IC50_mts022', 'log_IC50_mts025'),
    ('22_25', 'All', 'Truncated', 'AUC_sig', mts022_mts025_trunc, 'AUC_sig_mts022', 'AUC_sig_mts025'),
    ('22_25', 'All', 'Truncated', 'AUC_trapz', mts022_mts025_trunc, 'AUC_trapz_mts022', 'AUC_trapz_mts025'),
    ('22_25', 'All', 'Truncated', 'AUC_sig_trunc', mts022_mts025_trunc, 'AUC_sig_norm_mts022', 'AUC_sig_norm_mts025'),
    # MTS022 vs MTS026 - All, full-range    
    ('22_26', 'All', 'Full', 'EC50', mts022_mts026_full, 'log_EC50_mts022', 'log_EC50_mts026'),
    ('22_26', 'All', 'Full', 'IC50', mts022_mts026_full, 'log_IC50_mts022', 'log_IC50_mts026'),
    ('22_26', 'All', 'Full', 'AUC_sig', mts022_mts026_full, 'AUC_sig_mts022', 'AUC_sig_mts026'),
    ('22_26', 'All', 'Full', 'AUC_trapz', mts022_mts026_full, 'AUC_trapz_mts022', 'AUC_trapz_mts026'),
    ('22_26', 'All', 'Full', 'AUC_sig_trunc', mts022_mts026_full, 'AUC_sig_norm_mts022', 'AUC_sig_norm_mts026'),
    # MTS022 vs MTS026 - All, truncated-range    
    ('22_26', 'All', 'Truncated', 'EC50', mts022_mts026_trunc, 'log_EC50_mts022', 'log_EC50_mts026'),
    ('22_26', 'All', 'Truncated', 'IC50', mts022_mts026_trunc, 'log_IC50_mts022', 'log_IC50_mts026'),
    ('22_26', 'All', 'Truncated', 'AUC_sig', mts022_mts026_trunc, 'AUC_sig_mts022', 'AUC_sig_mts026'),
    ('22_26', 'All', 'Truncated', 'AUC_trapz', mts022_mts026_trunc, 'AUC_trapz_mts022', 'AUC_trapz_mts026'),
    ('22_26', 'All', 'Truncated', 'AUC_sig_trunc', mts022_mts026_trunc, 'AUC_sig_norm_mts022', 'AUC_sig_norm_mts026'),
    # MTS023 vs MTS024 - All, full-range    
    ('23_24', 'All', 'Full', 'EC50', mts023_mts024_full, 'log_EC50_mts023', 'log_EC50_mts024'),
    ('23_24', 'All', 'Full', 'IC50', mts023_mts024_full, 'log_IC50_mts023', 'log_IC50_mts024'),
    ('23_24', 'All', 'Full', 'AUC_sig', mts023_mts024_full, 'AUC_sig_mts023', 'AUC_sig_mts024'),
    ('23_24', 'All', 'Full', 'AUC_trapz', mts023_mts024_full, 'AUC_trapz_mts023', 'AUC_trapz_mts024'),
    ('23_24', 'All', 'Full', 'AUC_sig_trunc', mts023_mts024_full, 'AUC_sig_norm_mts023', 'AUC_sig_norm_mts024'),
    # MTS023 vs MTS024 - All, truncated-range    
    ('23_24', 'All', 'Truncated', 'EC50', mts023_mts024_trunc, 'log_EC50_mts023', 'log_EC50_mts024'),
    ('23_24', 'All', 'Truncated', 'IC50', mts023_mts024_trunc, 'log_IC50_mts023', 'log_IC50_mts024'),
    ('23_24', 'All', 'Truncated', 'AUC_sig', mts023_mts024_trunc, 'AUC_sig_mts023', 'AUC_sig_mts024'),
    ('23_24', 'All', 'Truncated', 'AUC_trapz', mts023_mts024_trunc, 'AUC_trapz_mts023', 'AUC_trapz_mts024'),
    ('23_24', 'All', 'Truncated', 'AUC_sig_trunc', mts023_mts024_trunc, 'AUC_sig_norm_mts023', 'AUC_sig_norm_mts024'),
    # MTS023 vs MTS025 - All, full-range    
    ('23_25', 'All', 'Full', 'EC50', mts023_mts025_full, 'log_EC50_mts023', 'log_EC50_mts025'),
    ('23_25', 'All', 'Full', 'IC50', mts023_mts025_full, 'log_IC50_mts023', 'log_IC50_mts025'),
    ('23_25', 'All', 'Full', 'AUC_sig', mts023_mts025_full, 'AUC_sig_mts023', 'AUC_sig_mts025'),
    ('23_25', 'All', 'Full', 'AUC_trapz', mts023_mts025_full, 'AUC_trapz_mts023', 'AUC_trapz_mts025'),
    ('23_25', 'All', 'Full', 'AUC_sig_trunc', mts023_mts025_full, 'AUC_sig_norm_mts023', 'AUC_sig_norm_mts025'),
    # MTS023 vs MTS025 - All, truncated-range    
    ('23_25', 'All', 'Truncated', 'EC50', mts023_mts025_trunc, 'log_EC50_mts023', 'log_EC50_mts025'),
    ('23_25', 'All', 'Truncated', 'IC50', mts023_mts025_trunc, 'log_IC50_mts023', 'log_IC50_mts025'),
    ('23_25', 'All', 'Truncated', 'AUC_sig', mts023_mts025_trunc, 'AUC_sig_mts023', 'AUC_sig_mts025'),
    ('23_25', 'All', 'Truncated', 'AUC_trapz', mts023_mts025_trunc, 'AUC_trapz_mts023', 'AUC_trapz_mts025'),
    ('23_25', 'All', 'Truncated', 'AUC_sig_trunc', mts023_mts025_trunc, 'AUC_sig_norm_mts023', 'AUC_sig_norm_mts025'),
    # MTS023 vs MTS026 - All, full-range    
    ('23_26', 'All', 'Full', 'EC50', mts023_mts026_full, 'log_EC50_mts023', 'log_EC50_mts026'),
    ('23_26', 'All', 'Full', 'IC50', mts023_mts026_full, 'log_IC50_mts023', 'log_IC50_mts026'),
    ('23_26', 'All', 'Full', 'AUC_sig', mts023_mts026_full, 'AUC_sig_mts023', 'AUC_sig_mts026'),
    ('23_26', 'All', 'Full', 'AUC_trapz', mts023_mts026_full, 'AUC_trapz_mts023', 'AUC_trapz_mts026'),
    ('23_26', 'All', 'Full', 'AUC_sig_trunc', mts023_mts026_full, 'AUC_sig_norm_mts023', 'AUC_sig_norm_mts026'),
    # MTS023 vs MTS026 - All, truncated-range    
    ('23_26', 'All', 'Truncated', 'EC50', mts023_mts026_trunc, 'log_EC50_mts023', 'log_EC50_mts026'),
    ('23_26', 'All', 'Truncated', 'IC50', mts023_mts026_trunc, 'log_IC50_mts023', 'log_IC50_mts026'),
    ('23_26', 'All', 'Truncated', 'AUC_sig', mts023_mts026_trunc, 'AUC_sig_mts023', 'AUC_sig_mts026'),
    ('23_26', 'All', 'Truncated', 'AUC_trapz', mts023_mts026_trunc, 'AUC_trapz_mts023', 'AUC_trapz_mts026'),
    ('23_26', 'All', 'Truncated', 'AUC_sig_trunc', mts023_mts026_trunc, 'AUC_sig_norm_mts023', 'AUC_sig_norm_mts026'),
    # MTS024 vs MTS025 - All, full-range    
    ('24_25', 'All', 'Full', 'EC50', mts024_mts025_full, 'log_EC50_mts024', 'log_EC50_mts025'),
    ('24_25', 'All', 'Full', 'IC50', mts024_mts025_full, 'log_IC50_mts024', 'log_IC50_mts025'),
    ('24_25', 'All', 'Full', 'AUC_sig', mts024_mts025_full, 'AUC_sig_mts024', 'AUC_sig_mts025'),
    ('24_25', 'All', 'Full', 'AUC_trapz', mts024_mts025_full, 'AUC_trapz_mts024', 'AUC_trapz_mts025'),
    ('24_25', 'All', 'Full', 'AUC_sig_trunc', mts024_mts025_full, 'AUC_sig_norm_mts024', 'AUC_sig_norm_mts025'),
    # MTS024 vs MTS025 - All, truncated-range    
    ('24_25', 'All', 'Truncated', 'EC50', mts024_mts025_trunc, 'log_EC50_mts024', 'log_EC50_mts025'),
    ('24_25', 'All', 'Truncated', 'IC50', mts024_mts025_trunc, 'log_IC50_mts024', 'log_IC50_mts025'),
    ('24_25', 'All', 'Truncated', 'AUC_sig', mts024_mts025_trunc, 'AUC_sig_mts024', 'AUC_sig_mts025'),
    ('24_25', 'All', 'Truncated', 'AUC_trapz', mts024_mts025_trunc, 'AUC_trapz_mts024', 'AUC_trapz_mts025'),
    ('24_25', 'All', 'Truncated', 'AUC_sig_trunc', mts024_mts025_trunc, 'AUC_sig_norm_mts024', 'AUC_sig_norm_mts025'),
    # MTS024 vs MTS026 - All, full-range    
    ('24_26', 'All', 'Full', 'EC50', mts024_mts026_full, 'log_EC50_mts024', 'log_EC50_mts026'),
    ('24_26', 'All', 'Full', 'IC50', mts024_mts026_full, 'log_IC50_mts024', 'log_IC50_mts026'),
    ('24_26', 'All', 'Full', 'AUC_sig', mts024_mts026_full, 'AUC_sig_mts024', 'AUC_sig_mts026'),
    ('24_26', 'All', 'Full', 'AUC_trapz', mts024_mts026_full, 'AUC_trapz_mts024', 'AUC_trapz_mts026'),
    ('24_26', 'All', 'Full', 'AUC_sig_trunc', mts024_mts026_full, 'AUC_sig_norm_mts024', 'AUC_sig_norm_mts026'),
    # MTS024 vs MTS026 - All, truncated-range    
    ('24_26', 'All', 'Truncated', 'EC50', mts024_mts026_trunc, 'log_EC50_mts024', 'log_EC50_mts026'),
    ('24_26', 'All', 'Truncated', 'IC50', mts024_mts026_trunc, 'log_IC50_mts024', 'log_IC50_mts026'),
    ('24_26', 'All', 'Truncated', 'AUC_sig', mts024_mts026_trunc, 'AUC_sig_mts024', 'AUC_sig_mts026'),
    ('24_26', 'All', 'Truncated', 'AUC_trapz', mts024_mts026_trunc, 'AUC_trapz_mts024', 'AUC_trapz_mts026'),
    ('24_26', 'All', 'Truncated', 'AUC_sig_trunc', mts024_mts026_trunc, 'AUC_sig_norm_mts024', 'AUC_sig_norm_mts026'),
    # MTS025 vs MTS026 - All, full-range    
    ('25_26', 'All', 'Full', 'EC50', mts025_mts026_full, 'log_EC50_mts025', 'log_EC50_mts026'),
    ('25_26', 'All', 'Full', 'IC50', mts025_mts026_full, 'log_IC50_mts025', 'log_IC50_mts026'),
    ('25_26', 'All', 'Full', 'AUC_sig', mts025_mts026_full, 'AUC_sig_mts025', 'AUC_sig_mts026'),
    ('25_26', 'All', 'Full', 'AUC_trapz', mts025_mts026_full, 'AUC_trapz_mts025', 'AUC_trapz_mts026'),
    ('25_26', 'All', 'Full', 'AUC_sig_trunc', mts025_mts026_full, 'AUC_sig_norm_mts025', 'AUC_sig_norm_mts026'),
    # MTS025 vs MTS026 - All, truncated-range    
    ('25_26', 'All', 'Truncated', 'EC50', mts025_mts026_trunc, 'log_EC50_mts025', 'log_EC50_mts026'),
    ('25_26', 'All', 'Truncated', 'IC50', mts025_mts026_trunc, 'log_IC50_mts025', 'log_IC50_mts026'),
    ('25_26', 'All', 'Truncated', 'AUC_sig', mts025_mts026_trunc, 'AUC_sig_mts025', 'AUC_sig_mts026'),
    ('25_26', 'All', 'Truncated', 'AUC_trapz', mts025_mts026_trunc, 'AUC_trapz_mts025', 'AUC_trapz_mts026'),
    ('25_26', 'All', 'Truncated', 'AUC_sig_trunc', mts025_mts026_trunc, 'AUC_sig_norm_mts025', 'AUC_sig_norm_mts026'),
]

# Run in parallel - uses all CPU cores
print("Starting bootstrap calculations...")
all_spearman_ci = Parallel(n_jobs=-1, verbose=10)(
    delayed(process_one_comparison)(db_pair, type_drug, type_dose, response, df, col1, col2)
    for db_pair, type_drug, type_dose, response, df, col1, col2 in all_comparisons
)

# Convert to DataFrame
spearman_ci_df = pd.DataFrame(all_spearman_ci)

# Save to CSV
spearman_ci_df.to_csv(args.output_path, index=False)