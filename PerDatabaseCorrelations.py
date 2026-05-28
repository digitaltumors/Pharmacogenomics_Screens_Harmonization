from joblib import Parallel, delayed
import scipy as sp
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gdsc2_full', type=str, required=True)
parser.add_argument('--gdsc2_trunc', type=str, required=True)
parser.add_argument('--prism_full', type=str, required=True)
parser.add_argument('--prism_trunc', type=str, required=True)
parser.add_argument('--gdsc1_full', type=str, required=True)
parser.add_argument('--gdsc1_trunc', type=str, required=True)
parser.add_argument('--ctrp_full', type=str, required=True)
parser.add_argument('--ctrp_trunc', type=str, required=True)
parser.add_argument('--compare_ic50', action='store_true', help='Activate if want to focus on comparing correlations for different IC50 ranges')
parser.add_argument('--compare_all_effective_drugs', action='store_true', help='Activate if want to focus on comparing correlations for all vs. effective drugs')
parser.add_argument('--corr_coeff', type=str, choices=['spearman', 'pearson'], default='spearman', help='Correlation coefficient to use')
parser.add_argument('--output_path', type=str, required=True)

args = parser.parse_args()

gdsc2_full = pd.read_csv(args.gdsc2_full)
gdsc2_trunc = pd.read_csv(args.gdsc2_trunc)
prism_full = pd.read_csv(args.prism_full)
prism_trunc = pd.read_csv(args.prism_trunc)
gdsc1_full = pd.read_csv(args.gdsc1_full)
gdsc1_trunc = pd.read_csv(args.gdsc1_trunc)
ctrpv2_full = pd.read_csv(args.ctrp_full)
ctrpv2_trunc = pd.read_csv(args.ctrp_trunc)
corr_coeff = args.corr_coeff

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

def bootstrap_pearson_pair(df, col1, col2, n_resamples=500):
    """Efficient bootstrap with reduced resamples"""
    corr = sp.stats.pearsonr(df[col1], df[col2], alternative='greater')
    bs = sp.stats.bootstrap(
        (df[col1].values, df[col2].values),  # Convert to numpy arrays
        lambda x, y: sp.stats.pearsonr(x, y)[0], 
        paired=True, 
        n_resamples=n_resamples,
        method='percentile',  # Faster than BCa
        random_state=42
    )
    return corr[0], bs.confidence_interval.low, bs.confidence_interval.high

def process_one_comparison(db_pair, type_dose, response, df, col1, col2, n_resamples=500):
    if corr_coeff == 'pearson':
        corr, ci_low, ci_high = bootstrap_pearson_pair(df, col1, col2, n_resamples)
    else:
        corr, ci_low, ci_high = bootstrap_spearman_pair(df, col1, col2, n_resamples)
    return {
        'db_pair': db_pair,
        'type_dose': type_dose,
        'response': response,
        'corr': corr,
        '95ci_low': ci_low,
        '95ci_high': ci_high,
        'n_unique_drugs': df['Drug ID'].nunique(),
        'n_unique_cells': df['Cell Line'].nunique(),
        'pairs': len(df)
    }

def process_one_comparison_all_effective_drugs(db_pair, type_drug, type_dose, response, df, col1, col2, n_resamples=500):
    if corr_coeff == 'pearson':
        corr, ci_low, ci_high = bootstrap_pearson_pair(df, col1, col2, n_resamples)
    else:
        corr, ci_low, ci_high = bootstrap_spearman_pair(df, col1, col2, n_resamples)
    return {
        'db_pair': db_pair,
        'type_drug': type_drug,
        'type_dose': type_dose,
        'response': response,
        'corr': corr,
        '95ci_low': ci_low,
        '95ci_high': ci_high,
        'n_unique_drugs': df['Drug ID'].nunique(),
        'n_unique_cells': df['Cell Line'].nunique(),
        'pairs': len(df)
    }

gdsc2_prism_full = pd.merge(gdsc2_full, prism_full, on = ['Drug ID', 'Cell Line'], suffixes=('_gdsc2', '_prism'))
gdsc2_prism_trunc = pd.merge(gdsc2_trunc, prism_trunc, on = ['Drug ID', 'Cell Line'], suffixes=('_gdsc2', '_prism'))
gdsc2_gdsc1_full = pd.merge(gdsc2_full, gdsc1_full, on = ['Drug ID', 'Cell Line'], suffixes=('_gdsc2', '_gdsc1'))
gdsc2_gdsc1_trunc = pd.merge(gdsc2_trunc, gdsc1_trunc, on = ['Drug ID', 'Cell Line'], suffixes=('_gdsc2', '_gdsc1'))
gdsc2_ctrpv2_full = pd.merge(gdsc2_full, ctrpv2_full, on = ['Drug ID', 'Cell Line'], suffixes=('_gdsc2', '_ctrpv2'))
gdsc2_ctrpv2_trunc = pd.merge(gdsc2_trunc, ctrpv2_trunc, on = ['Drug ID', 'Cell Line'], suffixes=('_gdsc2', '_ctrpv2'))
gdsc1_prism_full = pd.merge(gdsc1_full, prism_full, on = ['Drug ID', 'Cell Line'], suffixes=('_gdsc1', '_prism'))
gdsc1_prism_trunc = pd.merge(gdsc1_trunc, prism_trunc, on = ['Drug ID', 'Cell Line'], suffixes=('_gdsc1', '_prism'))
gdsc1_ctrpv2_full = pd.merge(gdsc1_full, ctrpv2_full, on = ['Drug ID', 'Cell Line'], suffixes=('_gdsc1', '_ctrpv2'))
gdsc1_ctrpv2_trunc = pd.merge(gdsc1_trunc, ctrpv2_trunc, on = ['Drug ID', 'Cell Line'], suffixes=('_gdsc1', '_ctrpv2'))
prism_ctrpv2_full = pd.merge(prism_full, ctrpv2_full, on = ['Drug ID', 'Cell Line'], suffixes=('_prism', '_ctrpv2'))
prism_ctrpv2_trunc = pd.merge(prism_trunc, ctrpv2_trunc, on = ['Drug ID', 'Cell Line'], suffixes=('_prism', '_ctrpv2'))


if args.compare_ic50:
    gdsc2_full_range = gdsc2_full[(gdsc2_full['log_IC50'] > -3) &  (gdsc2_full['log_IC50'] < 2)]
    prism_full_range = prism_full[(prism_full['log_IC50'] > -3) &  (prism_full['log_IC50'] < 2)]
    gdsc2_trunc_range = gdsc2_trunc[(gdsc2_trunc['log_IC50'] > -3) &  (gdsc2_trunc['log_IC50'] < 2)]
    prism_trunc_range = prism_trunc[(prism_trunc['log_IC50'] > -3) &  (prism_trunc['log_IC50'] < 2)]
    ctrpv2_full_range = ctrpv2_full[(ctrpv2_full['log_IC50'] > -3) &  (ctrpv2_full['log_IC50'] < 2)]
    ctrpv2_trunc_range = ctrpv2_trunc[(ctrpv2_trunc['log_IC50'] > -3) &  (ctrpv2_trunc['log_IC50'] < 2)]
    gdsc1_full_range = gdsc1_full[(gdsc1_full['log_IC50'] > -3) &  (gdsc1_full['log_IC50'] < 2)]
    gdsc1_trunc_range = gdsc1_trunc[(gdsc1_trunc['log_IC50'] > -3) &  (gdsc1_trunc['log_IC50'] < 2)]

    gdsc2_prism_full_range = pd.merge(gdsc2_full_range, prism_full_range, on = ['Drug ID', 'Cell Line'], suffixes=('_gdsc2', '_prism'))
    gdsc2_gdsc1_full_range = pd.merge(gdsc2_full_range, gdsc1_full_range, on = ['Drug ID', 'Cell Line'], suffixes=('_gdsc2', '_gdsc1'))
    gdsc2_ctrpv2_full_range = pd.merge(gdsc2_full_range, ctrpv2_full_range, on = ['Drug ID', 'Cell Line'], suffixes=('_gdsc2', '_ctrpv2'))
    gdsc1_prism_full_range = pd.merge(gdsc1_full_range, prism_full_range, on = ['Drug ID', 'Cell Line'], suffixes=('_gdsc1', '_prism'))
    gdsc1_ctrpv2_full_range = pd.merge(gdsc1_full_range, ctrpv2_full_range, on = ['Drug ID', 'Cell Line'], suffixes=('_gdsc1', '_ctrpv2'))
    prism_ctrpv2_full_range = pd.merge(prism_full_range, ctrpv2_full_range, on = ['Drug ID', 'Cell Line'], suffixes=('_prism', '_ctrpv2'))
    
    all_comparisons = [
        ('GDSCv2_PRISM', 'Full', 'IC50', gdsc2_prism_full, 'log_IC50_gdsc2', 'log_IC50_prism'),
        ('GDSCv2_GDSCv1', 'Full', 'IC50', gdsc2_gdsc1_full, 'log_IC50_gdsc2', 'log_IC50_gdsc1'),
        ('GDSCv2_CTRP', 'Full', 'IC50', gdsc2_ctrpv2_full, 'log_IC50_gdsc2', 'log_IC50_ctrpv2'),
        ('GDSCv1_PRISM', 'Full', 'IC50', gdsc1_prism_full, 'log_IC50_gdsc1', 'log_IC50_prism'),
        ('GDSCv1_CTRP', 'Full', 'IC50', gdsc1_ctrpv2_full, 'log_IC50_gdsc1', 'log_IC50_ctrpv2'),
        ('PRISM_CTRP', 'Full', 'IC50', prism_ctrpv2_full, 'log_IC50_prism', 'log_IC50_ctrpv2'),
        ('GDSCv2_PRISM', 'Range', 'IC50', gdsc2_prism_full_range, 'log_IC50_gdsc2', 'log_IC50_prism'),
        ('GDSCv2_GDSCv1', 'Range', 'IC50', gdsc2_gdsc1_full_range, 'log_IC50_gdsc2', 'log_IC50_gdsc1'),
        ('GDSCv2_CTRP', 'Range', 'IC50', gdsc2_ctrpv2_full_range, 'log_IC50_gdsc2', 'log_IC50_ctrpv2'),
        ('GDSCv1_PRISM', 'Range', 'IC50', gdsc1_prism_full_range, 'log_IC50_gdsc1', 'log_IC50_prism'),
        ('GDSCv1_CTRP', 'Range', 'IC50', gdsc1_ctrpv2_full_range, 'log_IC50_gdsc1', 'log_IC50_ctrpv2'),
        ('PRISM_CTRP', 'Range', 'IC50', prism_ctrpv2_full_range, 'log_IC50_prism', 'log_IC50_ctrpv2'),
    ]
    # Run in parallel - uses all CPU cores
    print("Starting bootstrap calculations...")
    all_corr_ci = Parallel(n_jobs=-1, verbose=10)(
        delayed(process_one_comparison)(db_pair, type_dose, response, df, col1, col2)
        for db_pair, type_dose, response, df, col1, col2 in all_comparisons
    )

elif args.compare_all_effective_drugs:
    # "Effective" by AUC_sig
    gdsc2_full_eff_AUCsig = gdsc2_full[gdsc2_full['AUC_sig'] < gdsc2_full['AUC_sig'].median()]
    prism_full_eff_AUCsig = prism_full[prism_full['AUC_sig'] < prism_full['AUC_sig'].median()]
    gdsc2_trunc_eff_AUCsig = gdsc2_trunc[gdsc2_trunc['AUC_sig'] < gdsc2_trunc['AUC_sig'].median()]
    prism_trunc_eff_AUCsig = prism_trunc[prism_trunc['AUC_sig'] < prism_trunc['AUC_sig'].median()]
    ctrpv2_full_eff_AUCsig = ctrpv2_full[ctrpv2_full['AUC_sig'] < ctrpv2_full['AUC_sig'].median()]
    ctrpv2_trunc_eff_AUCsig = ctrpv2_trunc[ctrpv2_trunc['AUC_sig'] < ctrpv2_trunc['AUC_sig'].median()]
    gdsc1_full_eff_AUCsig = gdsc1_full[gdsc1_full['AUC_sig'] < gdsc1_full['AUC_sig'].median()]
    gdsc1_trunc_eff_AUCsig = gdsc1_trunc[gdsc1_trunc['AUC_sig'] < gdsc1_trunc['AUC_sig'].median()]

    # By AUC_trapz
    gdsc2_full_eff_AUCtrapz = gdsc2_full[gdsc2_full['AUC_trapz'] < gdsc2_full['AUC_trapz'].median()]
    prism_full_eff_AUCtrapz = prism_full[prism_full['AUC_trapz'] < prism_full['AUC_trapz'].median()]
    gdsc2_trunc_eff_AUCtrapz = gdsc2_trunc[gdsc2_trunc['AUC_trapz'] < gdsc2_trunc['AUC_trapz'].median()]
    prism_trunc_eff_AUCtrapz = prism_trunc[prism_trunc['AUC_trapz'] < prism_trunc['AUC_trapz'].median()]
    ctrpv2_full_eff_AUCtrapz = ctrpv2_full[ctrpv2_full['AUC_trapz'] < ctrpv2_full['AUC_trapz'].median()]
    ctrpv2_trunc_eff_AUCtrapz = ctrpv2_trunc[ctrpv2_trunc['AUC_trapz'] < ctrpv2_trunc['AUC_trapz'].median()]
    gdsc1_full_eff_AUCtrapz = gdsc1_full[gdsc1_full['AUC_trapz'] < gdsc1_full['AUC_trapz'].median()]
    gdsc1_trunc_eff_AUCtrapz = gdsc1_trunc[gdsc1_trunc['AUC_trapz'] < gdsc1_trunc['AUC_trapz'].median()]

    # By AUC_sig_norm
    gdsc2_full_eff_AUCsignorm = gdsc2_full[gdsc2_full['AUC_sig_norm'] < gdsc2_full['AUC_sig_norm'].median()]
    prism_full_eff_AUCsignorm = prism_full[prism_full['AUC_sig_norm'] < prism_full['AUC_sig_norm'].median()]
    gdsc2_trunc_eff_AUCsignorm = gdsc2_trunc[gdsc2_trunc['AUC_sig_norm'] < gdsc2_trunc['AUC_sig_norm'].median()]
    prism_trunc_eff_AUCsignorm = prism_trunc[prism_trunc['AUC_sig_norm'] < prism_trunc['AUC_sig_norm'].median()]
    ctrpv2_full_eff_AUCsignorm = ctrpv2_full[ctrpv2_full['AUC_sig_norm'] < ctrpv2_full['AUC_sig_norm'].median()]
    ctrpv2_trunc_eff_AUCsignorm = ctrpv2_trunc[ctrpv2_trunc['AUC_sig_norm'] < ctrpv2_trunc['AUC_sig_norm'].median()]
    gdsc1_full_eff_AUCsignorm = gdsc1_full[gdsc1_full['AUC_sig_norm'] < gdsc1_full['AUC_sig_norm'].median()]
    gdsc1_trunc_eff_AUCsignorm = gdsc1_trunc[gdsc1_trunc['AUC_sig_norm'] < gdsc1_trunc['AUC_sig_norm'].median()]

    # By IC50
    gdsc2_full_eff_IC50 = gdsc2_full[gdsc2_full['log_IC50'] < gdsc2_full['log_IC50'].median()]
    prism_full_eff_IC50 = prism_full[prism_full['log_IC50'] < prism_full['log_IC50'].median()]
    gdsc2_trunc_eff_IC50 = gdsc2_trunc[gdsc2_trunc['log_IC50'] < gdsc2_trunc['log_IC50'].median()]
    prism_trunc_eff_IC50 = prism_trunc[prism_trunc['log_IC50'] < prism_trunc['log_IC50'].median()]
    ctrpv2_full_eff_IC50 = ctrpv2_full[ctrpv2_full['log_IC50'] < ctrpv2_full['log_IC50'].median()]
    ctrpv2_trunc_eff_IC50 = ctrpv2_trunc[ctrpv2_trunc['log_IC50'] < ctrpv2_trunc['log_IC50'].median()]
    gdsc1_full_eff_IC50 = gdsc1_full[gdsc1_full['log_IC50'] < gdsc1_full['log_IC50'].median()]
    gdsc1_trunc_eff_IC50 = gdsc1_trunc[gdsc1_trunc['log_IC50'] < gdsc1_trunc['log_IC50'].median()]

    # By EC50
    gdsc2_full_eff_EC50 = gdsc2_full[gdsc2_full['log_EC50'] < gdsc2_full['log_EC50'].median()]
    prism_full_eff_EC50 = prism_full[prism_full['log_EC50'] < prism_full['log_EC50'].median()]
    gdsc2_trunc_eff_EC50 = gdsc2_trunc[gdsc2_trunc['log_EC50'] < gdsc2_trunc['log_EC50'].median()]
    prism_trunc_eff_EC50 = prism_trunc[prism_trunc['log_EC50'] < prism_trunc['log_EC50'].median()]
    ctrpv2_full_eff_EC50 = ctrpv2_full[ctrpv2_full['log_EC50'] < ctrpv2_full['log_EC50'].median()]
    ctrpv2_trunc_eff_EC50 = ctrpv2_trunc[ctrpv2_trunc['log_EC50'] < ctrpv2_trunc['log_EC50'].median()]
    gdsc1_full_eff_EC50 = gdsc1_full[gdsc1_full['log_EC50'] < gdsc1_full['log_EC50'].median()]
    gdsc1_trunc_eff_EC50 = gdsc1_trunc[gdsc1_trunc['log_EC50'] < gdsc1_trunc['log_EC50'].median()]

    gdsc2_prism_full_eff_AUCsig = pd.merge(gdsc2_full_eff_AUCsig, prism_full_eff_AUCsig, on = ['Drug ID', 'Cell Line'], suffixes=('_gdsc2', '_prism'))
    gdsc2_prism_full_eff_AUCtrapz = pd.merge(gdsc2_full_eff_AUCtrapz, prism_full_eff_AUCtrapz, on = ['Drug ID', 'Cell Line'], suffixes=('_gdsc2', '_prism'))
    gdsc2_prism_full_eff_AUCsignorm = pd.merge(gdsc2_full_eff_AUCsignorm, prism_full_eff_AUCsignorm, on = ['Drug ID', 'Cell Line'], suffixes=('_gdsc2', '_prism'))
    gdsc2_prism_full_eff_IC50 = pd.merge(gdsc2_full_eff_IC50, prism_full_eff_IC50, on = ['Drug ID', 'Cell Line'], suffixes=('_gdsc2', '_prism'))
    gdsc2_prism_full_eff_EC50 = pd.merge(gdsc2_full_eff_EC50, prism_full_eff_EC50, on = ['Drug ID', 'Cell Line'], suffixes=('_gdsc2', '_prism'))
    gdsc2_prism_trunc_eff_AUCsig = pd.merge(gdsc2_trunc_eff_AUCsig, prism_trunc_eff_AUCsig, on = ['Drug ID', 'Cell Line'], suffixes=('_gdsc2', '_prism'))
    gdsc2_prism_trunc_eff_AUCtrapz = pd.merge(gdsc2_trunc_eff_AUCtrapz, prism_trunc_eff_AUCtrapz, on = ['Drug ID', 'Cell Line'], suffixes=('_gdsc2', '_prism'))
    gdsc2_prism_trunc_eff_AUCsignorm = pd.merge(gdsc2_trunc_eff_AUCsignorm, prism_trunc_eff_AUCsignorm, on = ['Drug ID', 'Cell Line'], suffixes=('_gdsc2', '_prism'))
    gdsc2_prism_trunc_eff_IC50 = pd.merge(gdsc2_trunc_eff_IC50, prism_trunc_eff_IC50, on = ['Drug ID', 'Cell Line'], suffixes=('_gdsc2', '_prism'))
    gdsc2_prism_trunc_eff_EC50 = pd.merge(gdsc2_trunc_eff_EC50, prism_trunc_eff_EC50, on = ['Drug ID', 'Cell Line'], suffixes=('_gdsc2', '_prism'))

    gdsc2_gdsc1_full_eff_AUCsig = pd.merge(gdsc2_full_eff_AUCsig, gdsc1_full_eff_AUCsig, on = ['Drug ID', 'Cell Line'], suffixes=('_gdsc2', '_gdsc1'))
    gdsc2_gdsc1_full_eff_AUCtrapz = pd.merge(gdsc2_full_eff_AUCtrapz, gdsc1_full_eff_AUCtrapz, on = ['Drug ID', 'Cell Line'], suffixes=('_gdsc2', '_gdsc1'))
    gdsc2_gdsc1_full_eff_AUCsignorm = pd.merge(gdsc2_full_eff_AUCsignorm, gdsc1_full_eff_AUCsignorm, on = ['Drug ID', 'Cell Line'], suffixes=('_gdsc2', '_gdsc1'))
    gdsc2_gdsc1_full_eff_IC50 = pd.merge(gdsc2_full_eff_IC50, gdsc1_full_eff_IC50, on = ['Drug ID', 'Cell Line'], suffixes=('_gdsc2', '_gdsc1'))
    gdsc2_gdsc1_full_eff_EC50 = pd.merge(gdsc2_full_eff_EC50, gdsc1_full_eff_EC50, on = ['Drug ID', 'Cell Line'], suffixes=('_gdsc2', '_gdsc1'))
    gdsc2_gdsc1_trunc_eff_AUCsig = pd.merge(gdsc2_trunc_eff_AUCsig, gdsc1_trunc_eff_AUCsig, on = ['Drug ID', 'Cell Line'], suffixes=('_gdsc2', '_gdsc1'))
    gdsc2_gdsc1_trunc_eff_AUCtrapz = pd.merge(gdsc2_trunc_eff_AUCtrapz, gdsc1_trunc_eff_AUCtrapz, on = ['Drug ID', 'Cell Line'], suffixes=('_gdsc2', '_gdsc1'))
    gdsc2_gdsc1_trunc_eff_AUCsignorm = pd.merge(gdsc2_trunc_eff_AUCsignorm, gdsc1_trunc_eff_AUCsignorm, on = ['Drug ID', 'Cell Line'], suffixes=('_gdsc2', '_gdsc1'))
    gdsc2_gdsc1_trunc_eff_IC50 = pd.merge(gdsc2_trunc_eff_IC50, gdsc1_trunc_eff_IC50, on = ['Drug ID', 'Cell Line'], suffixes=('_gdsc2', '_gdsc1'))
    gdsc2_gdsc1_trunc_eff_EC50 = pd.merge(gdsc2_trunc_eff_EC50, gdsc1_trunc_eff_EC50, on = ['Drug ID', 'Cell Line'], suffixes=('_gdsc2', '_gdsc1'))

    gdsc2_ctrpv2_full_eff_AUCsig = pd.merge(gdsc2_full_eff_AUCsig, ctrpv2_full_eff_AUCsig, on = ['Drug ID', 'Cell Line'], suffixes=('_gdsc2', '_ctrpv2'))
    gdsc2_ctrpv2_full_eff_AUCtrapz = pd.merge(gdsc2_full_eff_AUCtrapz, ctrpv2_full_eff_AUCtrapz, on = ['Drug ID', 'Cell Line'], suffixes=('_gdsc2', '_ctrpv2'))
    gdsc2_ctrpv2_full_eff_AUCsignorm = pd.merge(gdsc2_full_eff_AUCsignorm, ctrpv2_full_eff_AUCsignorm, on = ['Drug ID', 'Cell Line'], suffixes=('_gdsc2', '_ctrpv2'))
    gdsc2_ctrpv2_full_eff_IC50 = pd.merge(gdsc2_full_eff_IC50, ctrpv2_full_eff_IC50, on = ['Drug ID', 'Cell Line'], suffixes=('_gdsc2', '_ctrpv2'))
    gdsc2_ctrpv2_full_eff_EC50 = pd.merge(gdsc2_full_eff_EC50, ctrpv2_full_eff_EC50, on = ['Drug ID', 'Cell Line'], suffixes=('_gdsc2', '_ctrpv2'))
    gdsc2_ctrpv2_trunc_eff_AUCsig = pd.merge(gdsc2_trunc_eff_AUCsig, ctrpv2_trunc_eff_AUCsig, on = ['Drug ID', 'Cell Line'], suffixes=('_gdsc2', '_ctrpv2'))
    gdsc2_ctrpv2_trunc_eff_AUCtrapz = pd.merge(gdsc2_trunc_eff_AUCtrapz, ctrpv2_trunc_eff_AUCtrapz, on = ['Drug ID', 'Cell Line'], suffixes=('_gdsc2', '_ctrpv2'))
    gdsc2_ctrpv2_trunc_eff_AUCsignorm = pd.merge(gdsc2_trunc_eff_AUCsignorm, ctrpv2_trunc_eff_AUCsignorm, on = ['Drug ID', 'Cell Line'], suffixes=('_gdsc2', '_ctrpv2'))
    gdsc2_ctrpv2_trunc_eff_IC50 = pd.merge(gdsc2_trunc_eff_IC50, ctrpv2_trunc_eff_IC50, on = ['Drug ID', 'Cell Line'], suffixes=('_gdsc2', '_ctrpv2'))
    gdsc2_ctrpv2_trunc_eff_EC50 = pd.merge(gdsc2_trunc_eff_EC50, ctrpv2_trunc_eff_EC50, on = ['Drug ID', 'Cell Line'], suffixes=('_gdsc2', '_ctrpv2'))

    gdsc1_prism_full_eff_AUCsig = pd.merge(gdsc1_full_eff_AUCsig, prism_full_eff_AUCsig, on = ['Drug ID', 'Cell Line'], suffixes=('_gdsc1', '_prism'))
    gdsc1_prism_full_eff_AUCtrapz = pd.merge(gdsc1_full_eff_AUCtrapz, prism_full_eff_AUCtrapz, on = ['Drug ID', 'Cell Line'], suffixes=('_gdsc1', '_prism'))
    gdsc1_prism_full_eff_AUCsignorm = pd.merge(gdsc1_full_eff_AUCsignorm, prism_full_eff_AUCsignorm, on = ['Drug ID', 'Cell Line'], suffixes=('_gdsc1', '_prism'))
    gdsc1_prism_full_eff_IC50 = pd.merge(gdsc1_full_eff_IC50, prism_full_eff_IC50, on = ['Drug ID', 'Cell Line'], suffixes=('_gdsc1', '_prism'))
    gdsc1_prism_full_eff_EC50 = pd.merge(gdsc1_full_eff_EC50, prism_full_eff_EC50, on = ['Drug ID', 'Cell Line'], suffixes=('_gdsc1', '_prism'))
    gdsc1_prism_trunc_eff_AUCsig = pd.merge(gdsc1_trunc_eff_AUCsig, prism_trunc_eff_AUCsig, on = ['Drug ID', 'Cell Line'], suffixes=('_gdsc1', '_prism'))
    gdsc1_prism_trunc_eff_AUCtrapz = pd.merge(gdsc1_trunc_eff_AUCtrapz, prism_trunc_eff_AUCtrapz, on = ['Drug ID', 'Cell Line'], suffixes=('_gdsc1', '_prism'))
    gdsc1_prism_trunc_eff_AUCsignorm = pd.merge(gdsc1_trunc_eff_AUCsignorm, prism_trunc_eff_AUCsignorm, on = ['Drug ID', 'Cell Line'], suffixes=('_gdsc1', '_prism'))
    gdsc1_prism_trunc_eff_IC50 = pd.merge(gdsc1_trunc_eff_IC50, prism_trunc_eff_IC50, on = ['Drug ID', 'Cell Line'], suffixes=('_gdsc1', '_prism'))
    gdsc1_prism_trunc_eff_EC50 = pd.merge(gdsc1_trunc_eff_EC50, prism_trunc_eff_EC50, on = ['Drug ID', 'Cell Line'], suffixes=('_gdsc1', '_prism'))

    gdsc1_ctrpv2_full_eff_AUCsig = pd.merge(gdsc1_full_eff_AUCsig, ctrpv2_full_eff_AUCsig, on = ['Drug ID', 'Cell Line'], suffixes=('_gdsc1', '_ctrpv2'))
    gdsc1_ctrpv2_full_eff_AUCtrapz = pd.merge(gdsc1_full_eff_AUCtrapz, ctrpv2_full_eff_AUCtrapz, on = ['Drug ID', 'Cell Line'], suffixes=('_gdsc1', '_ctrpv2'))
    gdsc1_ctrpv2_full_eff_AUCsignorm = pd.merge(gdsc1_full_eff_AUCsignorm, ctrpv2_full_eff_AUCsignorm, on = ['Drug ID', 'Cell Line'], suffixes=('_gdsc1', '_ctrpv2'))
    gdsc1_ctrpv2_full_eff_IC50 = pd.merge(gdsc1_full_eff_IC50, ctrpv2_full_eff_IC50, on = ['Drug ID', 'Cell Line'], suffixes=('_gdsc1', '_ctrpv2'))
    gdsc1_ctrpv2_full_eff_EC50 = pd.merge(gdsc1_full_eff_EC50, ctrpv2_full_eff_EC50, on = ['Drug ID', 'Cell Line'], suffixes=('_gdsc1', '_ctrpv2'))
    gdsc1_ctrpv2_trunc_eff_AUCsig = pd.merge(gdsc1_trunc_eff_AUCsig, ctrpv2_trunc_eff_AUCsig, on = ['Drug ID', 'Cell Line'], suffixes=('_gdsc1', '_ctrpv2'))
    gdsc1_ctrpv2_trunc_eff_AUCtrapz = pd.merge(gdsc1_trunc_eff_AUCtrapz, ctrpv2_trunc_eff_AUCtrapz, on = ['Drug ID', 'Cell Line'], suffixes=('_gdsc1', '_ctrpv2'))
    gdsc1_ctrpv2_trunc_eff_AUCsignorm = pd.merge(gdsc1_trunc_eff_AUCsignorm, ctrpv2_trunc_eff_AUCsignorm, on = ['Drug ID', 'Cell Line'], suffixes=('_gdsc1', '_ctrpv2'))
    gdsc1_ctrpv2_trunc_eff_IC50 = pd.merge(gdsc1_trunc_eff_IC50, ctrpv2_trunc_eff_IC50, on = ['Drug ID', 'Cell Line'], suffixes=('_gdsc1', '_ctrpv2'))
    gdsc1_ctrpv2_trunc_eff_EC50 = pd.merge(gdsc1_trunc_eff_EC50, ctrpv2_trunc_eff_EC50, on = ['Drug ID', 'Cell Line'], suffixes=('_gdsc1', '_ctrpv2'))

    prism_ctrpv2_full_eff_AUCsig = pd.merge(prism_full_eff_AUCsig, ctrpv2_full_eff_AUCsig, on = ['Drug ID', 'Cell Line'], suffixes=('_prism', '_ctrpv2'))
    prism_ctrpv2_full_eff_AUCtrapz = pd.merge(prism_full_eff_AUCtrapz, ctrpv2_full_eff_AUCtrapz, on = ['Drug ID', 'Cell Line'], suffixes=('_prism', '_ctrpv2'))
    prism_ctrpv2_full_eff_AUCsignorm = pd.merge(prism_full_eff_AUCsignorm, ctrpv2_full_eff_AUCsignorm, on = ['Drug ID', 'Cell Line'], suffixes=('_prism', '_ctrpv2'))
    prism_ctrpv2_full_eff_IC50 = pd.merge(prism_full_eff_IC50, ctrpv2_full_eff_IC50, on = ['Drug ID', 'Cell Line'], suffixes=('_prism', '_ctrpv2'))
    prism_ctrpv2_full_eff_EC50 = pd.merge(prism_full_eff_EC50, ctrpv2_full_eff_EC50, on = ['Drug ID', 'Cell Line'], suffixes=('_prism', '_ctrpv2'))
    prism_ctrpv2_trunc_eff_AUCsig = pd.merge(prism_trunc_eff_AUCsig, ctrpv2_trunc_eff_AUCsig, on = ['Drug ID', 'Cell Line'], suffixes=('_prism', '_ctrpv2'))
    prism_ctrpv2_trunc_eff_AUCtrapz = pd.merge(prism_trunc_eff_AUCtrapz, ctrpv2_trunc_eff_AUCtrapz, on = ['Drug ID', 'Cell Line'], suffixes=('_prism', '_ctrpv2'))
    prism_ctrpv2_trunc_eff_AUCsignorm = pd.merge(prism_trunc_eff_AUCsignorm, ctrpv2_trunc_eff_AUCsignorm, on = ['Drug ID', 'Cell Line'], suffixes=('_prism', '_ctrpv2'))
    prism_ctrpv2_trunc_eff_IC50 = pd.merge(prism_trunc_eff_IC50, ctrpv2_trunc_eff_IC50, on = ['Drug ID', 'Cell Line'], suffixes=('_prism', '_ctrpv2'))
    prism_ctrpv2_trunc_eff_EC50 = pd.merge(prism_trunc_eff_EC50, ctrpv2_trunc_eff_EC50, on = ['Drug ID', 'Cell Line'], suffixes=('_prism', '_ctrpv2'))

    # Define all comparisons as tuples
    all_comparisons = [
        # GDSC2 vs PRISM - All, full-range
        ('GDSCv2_PRISM', 'All', 'Full', 'EC50', gdsc2_prism_full, 'log_EC50_gdsc2', 'log_EC50_prism'),
        ('GDSCv2_PRISM', 'All', 'Full', 'IC50', gdsc2_prism_full, 'log_IC50_gdsc2', 'log_IC50_prism'),
        ('GDSCv2_PRISM', 'All', 'Full', 'AUC_sig', gdsc2_prism_full, 'AUC_sig_gdsc2', 'AUC_sig_prism'),
        ('GDSCv2_PRISM', 'All', 'Full', 'AUC_trapz', gdsc2_prism_full, 'AUC_trapz_gdsc2', 'AUC_trapz_prism'),
        ('GDSCv2_PRISM', 'All', 'Full', 'AUC_sig_trunc', gdsc2_prism_full, 'AUC_sig_norm_gdsc2', 'AUC_sig_norm_prism'),
        # GDSC2 vs PRISM - Effective, full-range 
        ('GDSCv2_PRISM', 'Effective', 'Full', 'EC50', gdsc2_prism_full_eff_EC50, 'log_EC50_gdsc2', 'log_EC50_prism'),
        ('GDSCv2_PRISM', 'Effective', 'Full', 'IC50', gdsc2_prism_full_eff_IC50, 'log_IC50_gdsc2', 'log_IC50_prism'),
        ('GDSCv2_PRISM', 'Effective', 'Full', 'AUC_sig', gdsc2_prism_full_eff_AUCsig, 'AUC_sig_gdsc2', 'AUC_sig_prism'),
        ('GDSCv2_PRISM', 'Effective', 'Full', 'AUC_trapz', gdsc2_prism_full_eff_AUCtrapz, 'AUC_trapz_gdsc2', 'AUC_trapz_prism'),
        ('GDSCv2_PRISM', 'Effective', 'Full', 'AUC_sig_trunc', gdsc2_prism_full_eff_AUCsignorm, 'AUC_sig_norm_gdsc2', 'AUC_sig_norm_prism'),
        # GDSC2 vs PRISM - All, truncated-range    
        ('GDSCv2_PRISM', 'All', 'Truncated', 'EC50', gdsc2_prism_trunc, 'log_EC50_gdsc2', 'log_EC50_prism'),
        ('GDSCv2_PRISM', 'All', 'Truncated', 'IC50', gdsc2_prism_trunc, 'log_IC50_gdsc2', 'log_IC50_prism'),
        ('GDSCv2_PRISM', 'All', 'Truncated', 'AUC_sig', gdsc2_prism_trunc, 'AUC_sig_gdsc2', 'AUC_sig_prism'),
        ('GDSCv2_PRISM', 'All', 'Truncated', 'AUC_trapz', gdsc2_prism_trunc, 'AUC_trapz_gdsc2', 'AUC_trapz_prism'),
        ('GDSCv2_PRISM', 'All', 'Truncated', 'AUC_sig_trunc', gdsc2_prism_trunc, 'AUC_sig_norm_gdsc2', 'AUC_sig_norm_prism'),
        # GDSC2 vs PRISM - Effective, truncated-range    
        ('GDSCv2_PRISM', 'Effective', 'Truncated', 'EC50', gdsc2_prism_trunc_eff_EC50, 'log_EC50_gdsc2', 'log_EC50_prism'),
        ('GDSCv2_PRISM', 'Effective', 'Truncated', 'IC50', gdsc2_prism_trunc_eff_IC50, 'log_IC50_gdsc2', 'log_IC50_prism'),
        ('GDSCv2_PRISM', 'Effective', 'Truncated', 'AUC_sig', gdsc2_prism_trunc_eff_AUCsig, 'AUC_sig_gdsc2', 'AUC_sig_prism'),
        ('GDSCv2_PRISM', 'Effective', 'Truncated', 'AUC_trapz', gdsc2_prism_trunc_eff_AUCtrapz, 'AUC_trapz_gdsc2', 'AUC_trapz_prism'),
        ('GDSCv2_PRISM', 'Effective', 'Truncated', 'AUC_sig_trunc', gdsc2_prism_trunc_eff_AUCsignorm, 'AUC_sig_norm_gdsc2', 'AUC_sig_norm_prism'),
        # GDSC2 vs GDSC1 - All, full-range
        ('GDSCv2_GDSCv1', 'All', 'Full', 'EC50', gdsc2_gdsc1_full, 'log_EC50_gdsc2', 'log_EC50_gdsc1'),
        ('GDSCv2_GDSCv1', 'All', 'Full', 'IC50', gdsc2_gdsc1_full, 'log_IC50_gdsc2', 'log_IC50_gdsc1'),
        ('GDSCv2_GDSCv1', 'All', 'Full', 'AUC_sig', gdsc2_gdsc1_full, 'AUC_sig_gdsc2', 'AUC_sig_gdsc1'),
        ('GDSCv2_GDSCv1', 'All', 'Full', 'AUC_trapz', gdsc2_gdsc1_full, 'AUC_trapz_gdsc2', 'AUC_trapz_gdsc1'),
        ('GDSCv2_GDSCv1', 'All', 'Full', 'AUC_sig_trunc', gdsc2_gdsc1_full, 'AUC_sig_norm_gdsc2', 'AUC_sig_norm_gdsc1'),
        # GDSC2 vs GDSC1 - Effective, full-range
        ('GDSCv2_GDSCv1', 'Effective', 'Full', 'EC50', gdsc2_gdsc1_full_eff_EC50, 'log_EC50_gdsc2', 'log_EC50_gdsc1'),
        ('GDSCv2_GDSCv1', 'Effective', 'Full', 'IC50', gdsc2_gdsc1_full_eff_IC50, 'log_IC50_gdsc2', 'log_IC50_gdsc1'),
        ('GDSCv2_GDSCv1', 'Effective', 'Full', 'AUC_sig', gdsc2_gdsc1_full_eff_AUCsig, 'AUC_sig_gdsc2', 'AUC_sig_gdsc1'),
        ('GDSCv2_GDSCv1', 'Effective', 'Full', 'AUC_trapz', gdsc2_gdsc1_full_eff_AUCtrapz, 'AUC_trapz_gdsc2', 'AUC_trapz_gdsc1'),
        ('GDSCv2_GDSCv1', 'Effective', 'Full', 'AUC_sig_trunc', gdsc2_gdsc1_full_eff_AUCsignorm, 'AUC_sig_norm_gdsc2', 'AUC_sig_norm_gdsc1'),
        # GDSC2 vs GDSC1 -  All, truncated-range
        ('GDSCv2_GDSCv1', 'All', 'Truncated', 'EC50', gdsc2_gdsc1_trunc, 'log_EC50_gdsc2', 'log_EC50_gdsc1'),
        ('GDSCv2_GDSCv1', 'All', 'Truncated', 'IC50', gdsc2_gdsc1_trunc, 'log_IC50_gdsc2', 'log_IC50_gdsc1'),
        ('GDSCv2_GDSCv1', 'All', 'Truncated', 'AUC_sig', gdsc2_gdsc1_trunc, 'AUC_sig_gdsc2', 'AUC_sig_gdsc1'),
        ('GDSCv2_GDSCv1', 'All', 'Truncated', 'AUC_trapz', gdsc2_gdsc1_trunc, 'AUC_trapz_gdsc2', 'AUC_trapz_gdsc1'),
        ('GDSCv2_GDSCv1', 'All', 'Truncated', 'AUC_sig_trunc', gdsc2_gdsc1_trunc, 'AUC_sig_norm_gdsc2', 'AUC_sig_norm_gdsc1'),
        # GDSC2 vs GDSC1 -  Effective, truncated-range
        ('GDSCv2_GDSCv1', 'Effective', 'Truncated', 'EC50', gdsc2_gdsc1_trunc_eff_EC50, 'log_EC50_gdsc2', 'log_EC50_gdsc1'),
        ('GDSCv2_GDSCv1', 'Effective', 'Truncated', 'IC50', gdsc2_gdsc1_trunc_eff_IC50, 'log_IC50_gdsc2', 'log_IC50_gdsc1'),
        ('GDSCv2_GDSCv1', 'Effective', 'Truncated', 'AUC_sig', gdsc2_gdsc1_trunc_eff_AUCsig, 'AUC_sig_gdsc2', 'AUC_sig_gdsc1'),
        ('GDSCv2_GDSCv1', 'Effective', 'Truncated', 'AUC_trapz', gdsc2_gdsc1_trunc_eff_AUCtrapz, 'AUC_trapz_gdsc2', 'AUC_trapz_gdsc1'),
        ('GDSCv2_GDSCv1', 'Effective', 'Truncated', 'AUC_sig_trunc', gdsc2_gdsc1_trunc_eff_AUCsignorm, 'AUC_sig_norm_gdsc2', 'AUC_sig_norm_gdsc1'),
        # GDSC2 vs CTRP - All, full-range
        ('GDSCv2_CTRP', 'All', 'Full', 'EC50', gdsc2_ctrpv2_full, 'log_EC50_gdsc2', 'log_EC50_ctrpv2'),
        ('GDSCv2_CTRP', 'All', 'Full', 'IC50', gdsc2_ctrpv2_full, 'log_IC50_gdsc2', 'log_IC50_ctrpv2'),
        ('GDSCv2_CTRP', 'All', 'Full', 'AUC_sig', gdsc2_ctrpv2_full, 'AUC_sig_gdsc2', 'AUC_sig_ctrpv2'),
        ('GDSCv2_CTRP', 'All', 'Full', 'AUC_trapz', gdsc2_ctrpv2_full, 'AUC_trapz_gdsc2', 'AUC_trapz_ctrpv2'),
        ('GDSCv2_CTRP', 'All', 'Full', 'AUC_sig_trunc', gdsc2_ctrpv2_full, 'AUC_sig_norm_gdsc2', 'AUC_sig_norm_ctrpv2'),
        # GDSC2 vs CTRP - Effective, full-range
        ('GDSCv2_CTRP', 'Effective', 'Full', 'EC50', gdsc2_ctrpv2_full_eff_EC50, 'log_EC50_gdsc2', 'log_EC50_ctrpv2'),
        ('GDSCv2_CTRP', 'Effective', 'Full', 'IC50', gdsc2_ctrpv2_full_eff_IC50, 'log_IC50_gdsc2', 'log_IC50_ctrpv2'),
        ('GDSCv2_CTRP', 'Effective', 'Full', 'AUC_sig', gdsc2_ctrpv2_full_eff_AUCsig, 'AUC_sig_gdsc2', 'AUC_sig_ctrpv2'),
        ('GDSCv2_CTRP', 'Effective', 'Full', 'AUC_trapz', gdsc2_ctrpv2_full_eff_AUCtrapz, 'AUC_trapz_gdsc2', 'AUC_trapz_ctrpv2'),
        ('GDSCv2_CTRP', 'Effective', 'Full', 'AUC_sig_trunc', gdsc2_ctrpv2_full_eff_AUCsignorm, 'AUC_sig_norm_gdsc2', 'AUC_sig_norm_ctrpv2'),
        # GDSC2 vs CTRP - All, truncated-range
        ('GDSCv2_CTRP', 'All', 'Truncated', 'EC50', gdsc2_ctrpv2_trunc, 'log_EC50_gdsc2', 'log_EC50_ctrpv2'),
        ('GDSCv2_CTRP', 'All', 'Truncated', 'IC50', gdsc2_ctrpv2_trunc, 'log_IC50_gdsc2', 'log_IC50_ctrpv2'),
        ('GDSCv2_CTRP', 'All', 'Truncated', 'AUC_sig', gdsc2_ctrpv2_trunc, 'AUC_sig_gdsc2', 'AUC_sig_ctrpv2'),
        ('GDSCv2_CTRP', 'All', 'Truncated', 'AUC_trapz', gdsc2_ctrpv2_trunc, 'AUC_trapz_gdsc2', 'AUC_trapz_ctrpv2'),
        ('GDSCv2_CTRP', 'All', 'Truncated', 'AUC_sig_trunc', gdsc2_ctrpv2_trunc, 'AUC_sig_norm_gdsc2', 'AUC_sig_norm_ctrpv2'),
        # GDSC2 vs CTRP - Effective, truncated-range
        ('GDSCv2_CTRP', 'Effective', 'Truncated', 'EC50', gdsc2_ctrpv2_trunc_eff_EC50, 'log_EC50_gdsc2', 'log_EC50_ctrpv2'),
        ('GDSCv2_CTRP', 'Effective', 'Truncated', 'IC50', gdsc2_ctrpv2_trunc_eff_IC50, 'log_IC50_gdsc2', 'log_IC50_ctrpv2'),
        ('GDSCv2_CTRP', 'Effective', 'Truncated', 'AUC_sig', gdsc2_ctrpv2_trunc_eff_AUCsig, 'AUC_sig_gdsc2', 'AUC_sig_ctrpv2'),
        ('GDSCv2_CTRP', 'Effective', 'Truncated', 'AUC_trapz', gdsc2_ctrpv2_trunc_eff_AUCtrapz, 'AUC_trapz_gdsc2', 'AUC_trapz_ctrpv2'),
        ('GDSCv2_CTRP', 'Effective', 'Truncated', 'AUC_sig_trunc', gdsc2_ctrpv2_trunc_eff_AUCsignorm, 'AUC_sig_norm_gdsc2', 'AUC_sig_norm_ctrpv2'),
        # GDSC1 vs PRISM - All, full-range
        ('GDSCv1_PRISM', 'All', 'Full', 'EC50', gdsc1_prism_full, 'log_EC50_gdsc1', 'log_EC50_prism'),
        ('GDSCv1_PRISM', 'All', 'Full', 'IC50', gdsc1_prism_full, 'log_IC50_gdsc1', 'log_IC50_prism'),
        ('GDSCv1_PRISM', 'All', 'Full', 'AUC_sig', gdsc1_prism_full, 'AUC_sig_gdsc1', 'AUC_sig_prism'),
        ('GDSCv1_PRISM', 'All', 'Full', 'AUC_trapz', gdsc1_prism_full, 'AUC_trapz_gdsc1', 'AUC_trapz_prism'),
        ('GDSCv1_PRISM', 'All', 'Full', 'AUC_sig_trunc', gdsc1_prism_full, 'AUC_sig_norm_gdsc1', 'AUC_sig_norm_prism'),
        # GDSC1 vs PRISM - Effective, full-range
        ('GDSCv1_PRISM', 'Effective', 'Full', 'EC50', gdsc1_prism_full_eff_EC50, 'log_EC50_gdsc1', 'log_EC50_prism'),
        ('GDSCv1_PRISM', 'Effective', 'Full', 'IC50', gdsc1_prism_full_eff_IC50, 'log_IC50_gdsc1', 'log_IC50_prism'),
        ('GDSCv1_PRISM', 'Effective', 'Full', 'AUC_sig', gdsc1_prism_full_eff_AUCsig, 'AUC_sig_gdsc1', 'AUC_sig_prism'),
        ('GDSCv1_PRISM', 'Effective', 'Full', 'AUC_trapz', gdsc1_prism_full_eff_AUCtrapz, 'AUC_trapz_gdsc1', 'AUC_trapz_prism'),
        ('GDSCv1_PRISM', 'Effective', 'Full', 'AUC_sig_trunc', gdsc1_prism_full_eff_AUCsignorm, 'AUC_sig_norm_gdsc1', 'AUC_sig_norm_prism'),
        # GDSC1 vs PRISM - All, truncated-range
        ('GDSCv1_PRISM', 'All', 'Truncated', 'EC50', gdsc1_prism_trunc, 'log_EC50_gdsc1', 'log_EC50_prism'),
        ('GDSCv1_PRISM', 'All', 'Truncated', 'IC50', gdsc1_prism_trunc, 'log_IC50_gdsc1', 'log_IC50_prism'),
        ('GDSCv1_PRISM', 'All', 'Truncated', 'AUC_sig', gdsc1_prism_trunc, 'AUC_sig_gdsc1', 'AUC_sig_prism'),
        ('GDSCv1_PRISM', 'All', 'Truncated', 'AUC_trapz', gdsc1_prism_trunc, 'AUC_trapz_gdsc1', 'AUC_trapz_prism'),
        ('GDSCv1_PRISM', 'All', 'Truncated', 'AUC_sig_trunc', gdsc1_prism_trunc, 'AUC_sig_norm_gdsc1', 'AUC_sig_norm_prism'),
        # GDSC1 vs PRISM - Effective, truncated-range
        ('GDSCv1_PRISM', 'Effective', 'Truncated', 'EC50', gdsc1_prism_trunc_eff_EC50, 'log_EC50_gdsc1', 'log_EC50_prism'),
        ('GDSCv1_PRISM', 'Effective', 'Truncated', 'IC50', gdsc1_prism_trunc_eff_IC50, 'log_IC50_gdsc1', 'log_IC50_prism'),
        ('GDSCv1_PRISM', 'Effective', 'Truncated', 'AUC_sig', gdsc1_prism_trunc_eff_AUCsig, 'AUC_sig_gdsc1', 'AUC_sig_prism'),
        ('GDSCv1_PRISM', 'Effective', 'Truncated', 'AUC_trapz', gdsc1_prism_trunc_eff_AUCtrapz, 'AUC_trapz_gdsc1', 'AUC_trapz_prism'),
        ('GDSCv1_PRISM', 'Effective', 'Truncated', 'AUC_sig_trunc', gdsc1_prism_trunc_eff_AUCsignorm, 'AUC_sig_norm_gdsc1', 'AUC_sig_norm_prism'),
        # GDSC1 vs CTRP - All, full-range
        ('GDSCv1_CTRP', 'All', 'Full', 'EC50', gdsc1_ctrpv2_full, 'log_EC50_gdsc1', 'log_EC50_ctrpv2'),
        ('GDSCv1_CTRP', 'All', 'Full', 'IC50', gdsc1_ctrpv2_full, 'log_IC50_gdsc1', 'log_IC50_ctrpv2'),
        ('GDSCv1_CTRP', 'All', 'Full', 'AUC_sig', gdsc1_ctrpv2_full, 'AUC_sig_gdsc1', 'AUC_sig_ctrpv2'),
        ('GDSCv1_CTRP', 'All', 'Full', 'AUC_trapz', gdsc1_ctrpv2_full, 'AUC_trapz_gdsc1', 'AUC_trapz_ctrpv2'),
        ('GDSCv1_CTRP', 'All', 'Full', 'AUC_sig_trunc', gdsc1_ctrpv2_full, 'AUC_sig_norm_gdsc1', 'AUC_sig_norm_ctrpv2'),
        # GDSC1 vs CTRP - Effective, full-range
        ('GDSCv1_CTRP', 'Effective', 'Full', 'EC50', gdsc1_ctrpv2_full_eff_EC50, 'log_EC50_gdsc1', 'log_EC50_ctrpv2'),
        ('GDSCv1_CTRP', 'Effective', 'Full', 'IC50', gdsc1_ctrpv2_full_eff_IC50, 'log_IC50_gdsc1', 'log_IC50_ctrpv2'),
        ('GDSCv1_CTRP', 'Effective', 'Full', 'AUC_sig', gdsc1_ctrpv2_full_eff_AUCsig, 'AUC_sig_gdsc1', 'AUC_sig_ctrpv2'),
        ('GDSCv1_CTRP', 'Effective', 'Full', 'AUC_trapz', gdsc1_ctrpv2_full_eff_AUCtrapz, 'AUC_trapz_gdsc1', 'AUC_trapz_ctrpv2'),
        ('GDSCv1_CTRP', 'Effective', 'Full', 'AUC_sig_trunc', gdsc1_ctrpv2_full_eff_AUCsignorm, 'AUC_sig_norm_gdsc1', 'AUC_sig_norm_ctrpv2'),
        # GDSC1 vs CTRP - All, truncated-range
        ('GDSCv1_CTRP', 'All', 'Truncated', 'EC50', gdsc1_ctrpv2_trunc, 'log_EC50_gdsc1', 'log_EC50_ctrpv2'),
        ('GDSCv1_CTRP', 'All', 'Truncated', 'IC50', gdsc1_ctrpv2_trunc, 'log_IC50_gdsc1', 'log_IC50_ctrpv2'),
        ('GDSCv1_CTRP', 'All', 'Truncated', 'AUC_sig', gdsc1_ctrpv2_trunc, 'AUC_sig_gdsc1', 'AUC_sig_ctrpv2'),
        ('GDSCv1_CTRP', 'All', 'Truncated', 'AUC_trapz', gdsc1_ctrpv2_trunc, 'AUC_trapz_gdsc1', 'AUC_trapz_ctrpv2'),
        ('GDSCv1_CTRP', 'All', 'Truncated', 'AUC_sig_trunc', gdsc1_ctrpv2_trunc, 'AUC_sig_norm_gdsc1', 'AUC_sig_norm_ctrpv2'),
        # GDSC1 vs CTRP - Effective, truncated-range
        ('GDSCv1_CTRP', 'Effective', 'Truncated', 'EC50', gdsc1_ctrpv2_trunc_eff_EC50, 'log_EC50_gdsc1', 'log_EC50_ctrpv2'),
        ('GDSCv1_CTRP', 'Effective', 'Truncated', 'IC50', gdsc1_ctrpv2_trunc_eff_IC50, 'log_IC50_gdsc1', 'log_IC50_ctrpv2'),
        ('GDSCv1_CTRP', 'Effective', 'Truncated', 'AUC_sig', gdsc1_ctrpv2_trunc_eff_AUCsig, 'AUC_sig_gdsc1', 'AUC_sig_ctrpv2'),
        ('GDSCv1_CTRP', 'Effective', 'Truncated', 'AUC_trapz', gdsc1_ctrpv2_trunc_eff_AUCtrapz, 'AUC_trapz_gdsc1', 'AUC_trapz_ctrpv2'),
        ('GDSCv1_CTRP', 'Effective', 'Truncated', 'AUC_sig_trunc', gdsc1_ctrpv2_trunc_eff_AUCsignorm, 'AUC_sig_norm_gdsc1', 'AUC_sig_norm_ctrpv2'),
        # PRISM vs CTRP - All, full-range
        ('PRISM_CTRP', 'All', 'Full', 'EC50', prism_ctrpv2_full, 'log_EC50_prism', 'log_EC50_ctrpv2'),
        ('PRISM_CTRP', 'All', 'Full', 'IC50', prism_ctrpv2_full, 'log_IC50_prism', 'log_IC50_ctrpv2'),
        ('PRISM_CTRP', 'All', 'Full', 'AUC_sig', prism_ctrpv2_full, 'AUC_sig_prism', 'AUC_sig_ctrpv2'),
        ('PRISM_CTRP', 'All', 'Full', 'AUC_trapz', prism_ctrpv2_full, 'AUC_trapz_prism', 'AUC_trapz_ctrpv2'),
        ('PRISM_CTRP', 'All', 'Full', 'AUC_sig_trunc', prism_ctrpv2_full, 'AUC_sig_norm_prism', 'AUC_sig_norm_ctrpv2'),
        # PRISM vs CTRP - Effective, full-range
        ('PRISM_CTRP', 'Effective', 'Full', 'EC50', prism_ctrpv2_full_eff_EC50, 'log_EC50_prism', 'log_EC50_ctrpv2'),
        ('PRISM_CTRP', 'Effective', 'Full', 'IC50', prism_ctrpv2_full_eff_IC50, 'log_IC50_prism', 'log_IC50_ctrpv2'),
        ('PRISM_CTRP', 'Effective', 'Full', 'AUC_sig', prism_ctrpv2_full_eff_AUCsig, 'AUC_sig_prism', 'AUC_sig_ctrpv2'),
        ('PRISM_CTRP', 'Effective', 'Full', 'AUC_trapz', prism_ctrpv2_full_eff_AUCtrapz, 'AUC_trapz_prism', 'AUC_trapz_ctrpv2'),
        ('PRISM_CTRP', 'Effective', 'Full', 'AUC_sig_trunc', prism_ctrpv2_full_eff_AUCsignorm, 'AUC_sig_norm_prism', 'AUC_sig_norm_ctrpv2'),
        # PRISM vs CTRP - All, truncated-range
        ('PRISM_CTRP', 'All', 'Truncated', 'EC50', prism_ctrpv2_trunc, 'log_EC50_prism', 'log_EC50_ctrpv2'),
        ('PRISM_CTRP', 'All', 'Truncated', 'IC50', prism_ctrpv2_trunc, 'log_IC50_prism', 'log_IC50_ctrpv2'),
        ('PRISM_CTRP', 'All', 'Truncated', 'AUC_sig', prism_ctrpv2_trunc, 'AUC_sig_prism', 'AUC_sig_ctrpv2'),
        ('PRISM_CTRP', 'All', 'Truncated', 'AUC_trapz', prism_ctrpv2_trunc, 'AUC_trapz_prism', 'AUC_trapz_ctrpv2'),
        ('PRISM_CTRP', 'All', 'Truncated', 'AUC_sig_trunc', prism_ctrpv2_trunc, 'AUC_sig_norm_prism', 'AUC_sig_norm_ctrpv2'),
        # PRISM vs CTRP - Effective, truncated-range
        ('PRISM_CTRP', 'Effective', 'Truncated', 'EC50', prism_ctrpv2_trunc_eff_EC50, 'log_EC50_prism', 'log_EC50_ctrpv2'),
        ('PRISM_CTRP', 'Effective', 'Truncated', 'IC50', prism_ctrpv2_trunc_eff_IC50, 'log_IC50_prism', 'log_IC50_ctrpv2'),
        ('PRISM_CTRP', 'Effective', 'Truncated', 'AUC_sig', prism_ctrpv2_trunc_eff_AUCsig, 'AUC_sig_prism', 'AUC_sig_ctrpv2'),
        ('PRISM_CTRP', 'Effective', 'Truncated', 'AUC_trapz', prism_ctrpv2_trunc_eff_AUCtrapz, 'AUC_trapz_prism', 'AUC_trapz_ctrpv2'),
        ('PRISM_CTRP', 'Effective', 'Truncated', 'AUC_sig_trunc', prism_ctrpv2_trunc_eff_AUCsignorm, 'AUC_sig_norm_prism', 'AUC_sig_norm_ctrpv2')
    ]
    # Run in parallel - uses all CPU cores
    print("Starting bootstrap calculations...")
    all_corr_ci = Parallel(n_jobs=-1, verbose=10)(
        delayed(process_one_comparison_all_effective_drugs)(db_pair, type_drug, type_dose, response, df, col1, col2)
        for db_pair, type_drug, type_dose, response, df, col1, col2 in all_comparisons
    )


else:
    # Define all comparisons as tuples
    all_comparisons = [
        # GDSC2 vs PRISM - Full
        ('GDSCv2_PRISM', 'Full', 'EC50', gdsc2_prism_full, 'log_EC50_gdsc2', 'log_EC50_prism'),
        ('GDSCv2_PRISM', 'Full', 'IC50', gdsc2_prism_full, 'log_IC50_gdsc2', 'log_IC50_prism'),
        ('GDSCv2_PRISM', 'Full', 'AUC_sig', gdsc2_prism_full, 'AUC_sig_gdsc2', 'AUC_sig_prism'),
        ('GDSCv2_PRISM', 'Full', 'AUC_trapz', gdsc2_prism_full, 'AUC_trapz_gdsc2', 'AUC_trapz_prism'),
        ('GDSCv2_PRISM', 'Full', 'AUC_sig_trunc', gdsc2_prism_full, 'AUC_sig_norm_gdsc2', 'AUC_sig_norm_prism'),
        # GDSC2 vs PRISM - Truncated
        ('GDSCv2_PRISM', 'Truncated', 'EC50', gdsc2_prism_trunc, 'log_EC50_gdsc2', 'log_EC50_prism'),
        ('GDSCv2_PRISM', 'Truncated', 'IC50', gdsc2_prism_trunc, 'log_IC50_gdsc2', 'log_IC50_prism'),
        ('GDSCv2_PRISM', 'Truncated', 'AUC_sig', gdsc2_prism_trunc, 'AUC_sig_gdsc2', 'AUC_sig_prism'),
        ('GDSCv2_PRISM', 'Truncated', 'AUC_trapz', gdsc2_prism_trunc, 'AUC_trapz_gdsc2', 'AUC_trapz_prism'),
        ('GDSCv2_PRISM', 'Truncated', 'AUC_sig_trunc', gdsc2_prism_trunc, 'AUC_sig_norm_gdsc2', 'AUC_sig_norm_prism'),
        # GDSC2 vs GDSC1 - Full
        ('GDSCv2_GDSCv1', 'Full', 'EC50', gdsc2_gdsc1_full, 'log_EC50_gdsc2', 'log_EC50_gdsc1'),
        ('GDSCv2_GDSCv1', 'Full', 'IC50', gdsc2_gdsc1_full, 'log_IC50_gdsc2', 'log_IC50_gdsc1'),
        ('GDSCv2_GDSCv1', 'Full', 'AUC_sig', gdsc2_gdsc1_full, 'AUC_sig_gdsc2', 'AUC_sig_gdsc1'),
        ('GDSCv2_GDSCv1', 'Full', 'AUC_trapz', gdsc2_gdsc1_full, 'AUC_trapz_gdsc2', 'AUC_trapz_gdsc1'),
        ('GDSCv2_GDSCv1', 'Full', 'AUC_sig_trunc', gdsc2_gdsc1_full, 'AUC_sig_norm_gdsc2', 'AUC_sig_norm_gdsc1'),
        # GDSC2 vs GDSC1 - Truncated
        ('GDSCv2_GDSCv1', 'Truncated', 'EC50', gdsc2_gdsc1_trunc, 'log_EC50_gdsc2', 'log_EC50_gdsc1'),
        ('GDSCv2_GDSCv1', 'Truncated', 'IC50', gdsc2_gdsc1_trunc, 'log_IC50_gdsc2', 'log_IC50_gdsc1'),
        ('GDSCv2_GDSCv1', 'Truncated', 'AUC_sig', gdsc2_gdsc1_trunc, 'AUC_sig_gdsc2', 'AUC_sig_gdsc1'),
        ('GDSCv2_GDSCv1', 'Truncated', 'AUC_trapz', gdsc2_gdsc1_trunc, 'AUC_trapz_gdsc2', 'AUC_trapz_gdsc1'),
        ('GDSCv2_GDSCv1', 'Truncated', 'AUC_sig_trunc', gdsc2_gdsc1_trunc, 'AUC_sig_norm_gdsc2', 'AUC_sig_norm_gdsc1'),
        # GDSC2 vs CTRP - Full
        ('GDSCv2_CTRP', 'Full', 'EC50', gdsc2_ctrpv2_full, 'log_EC50_gdsc2', 'log_EC50_ctrpv2'),
        ('GDSCv2_CTRP', 'Full', 'IC50', gdsc2_ctrpv2_full, 'log_IC50_gdsc2', 'log_IC50_ctrpv2'),
        ('GDSCv2_CTRP', 'Full', 'AUC_sig', gdsc2_ctrpv2_full, 'AUC_sig_gdsc2', 'AUC_sig_ctrpv2'),
        ('GDSCv2_CTRP', 'Full', 'AUC_trapz', gdsc2_ctrpv2_full, 'AUC_trapz_gdsc2', 'AUC_trapz_ctrpv2'),
        ('GDSCv2_CTRP', 'Full', 'AUC_sig_trunc', gdsc2_ctrpv2_full, 'AUC_sig_norm_gdsc2', 'AUC_sig_norm_ctrpv2'),
        # GDSC2 vs CTRP - Truncated
        ('GDSCv2_CTRP', 'Truncated', 'EC50', gdsc2_ctrpv2_trunc, 'log_EC50_gdsc2', 'log_EC50_ctrpv2'),
        ('GDSCv2_CTRP', 'Truncated', 'IC50', gdsc2_ctrpv2_trunc, 'log_IC50_gdsc2', 'log_IC50_ctrpv2'),
        ('GDSCv2_CTRP', 'Truncated', 'AUC_sig', gdsc2_ctrpv2_trunc, 'AUC_sig_gdsc2', 'AUC_sig_ctrpv2'),
        ('GDSCv2_CTRP', 'Truncated', 'AUC_trapz', gdsc2_ctrpv2_trunc, 'AUC_trapz_gdsc2', 'AUC_trapz_ctrpv2'),
        ('GDSCv2_CTRP', 'Truncated', 'AUC_sig_trunc', gdsc2_ctrpv2_trunc, 'AUC_sig_norm_gdsc2', 'AUC_sig_norm_ctrpv2'),
        # GDSC1 vs PRISM - Full
        ('GDSCv1_PRISM', 'Full', 'EC50', gdsc1_prism_full, 'log_EC50_gdsc1', 'log_EC50_prism'),
        ('GDSCv1_PRISM', 'Full', 'IC50', gdsc1_prism_full, 'log_IC50_gdsc1', 'log_IC50_prism'),
        ('GDSCv1_PRISM', 'Full', 'AUC_sig', gdsc1_prism_full, 'AUC_sig_gdsc1', 'AUC_sig_prism'),
        ('GDSCv1_PRISM', 'Full', 'AUC_trapz', gdsc1_prism_full, 'AUC_trapz_gdsc1', 'AUC_trapz_prism'),
        ('GDSCv1_PRISM', 'Full', 'AUC_sig_trunc', gdsc1_prism_full, 'AUC_sig_norm_gdsc1', 'AUC_sig_norm_prism'),
        # GDSC1 vs PRISM - Truncated
        ('GDSCv1_PRISM', 'Truncated', 'EC50', gdsc1_prism_trunc, 'log_EC50_gdsc1', 'log_EC50_prism'),
        ('GDSCv1_PRISM', 'Truncated', 'IC50', gdsc1_prism_trunc, 'log_IC50_gdsc1', 'log_IC50_prism'),
        ('GDSCv1_PRISM', 'Truncated', 'AUC_sig', gdsc1_prism_trunc, 'AUC_sig_gdsc1', 'AUC_sig_prism'),
        ('GDSCv1_PRISM', 'Truncated', 'AUC_trapz', gdsc1_prism_trunc, 'AUC_trapz_gdsc1', 'AUC_trapz_prism'),
        ('GDSCv1_PRISM', 'Truncated', 'AUC_sig_trunc', gdsc1_prism_trunc, 'AUC_sig_norm_gdsc1', 'AUC_sig_norm_prism'),
        # GDSC1 vs CTRP - Full
        ('GDSCv1_CTRP', 'Full', 'EC50', gdsc1_ctrpv2_full, 'log_EC50_gdsc1', 'log_EC50_ctrpv2'),
        ('GDSCv1_CTRP', 'Full', 'IC50', gdsc1_ctrpv2_full, 'log_IC50_gdsc1', 'log_IC50_ctrpv2'),
        ('GDSCv1_CTRP', 'Full', 'AUC_sig', gdsc1_ctrpv2_full, 'AUC_sig_gdsc1', 'AUC_sig_ctrpv2'),
        ('GDSCv1_CTRP', 'Full', 'AUC_trapz', gdsc1_ctrpv2_full, 'AUC_trapz_gdsc1', 'AUC_trapz_ctrpv2'),
        ('GDSCv1_CTRP', 'Full', 'AUC_sig_trunc', gdsc1_ctrpv2_full, 'AUC_sig_norm_gdsc1', 'AUC_sig_norm_ctrpv2'),
        # GDSC1 vs CTRP - Truncated
        ('GDSCv1_CTRP', 'Truncated', 'EC50', gdsc1_ctrpv2_trunc, 'log_EC50_gdsc1', 'log_EC50_ctrpv2'),
        ('GDSCv1_CTRP', 'Truncated', 'IC50', gdsc1_ctrpv2_trunc, 'log_IC50_gdsc1', 'log_IC50_ctrpv2'),
        ('GDSCv1_CTRP', 'Truncated', 'AUC_sig', gdsc1_ctrpv2_trunc, 'AUC_sig_gdsc1', 'AUC_sig_ctrpv2'),
        ('GDSCv1_CTRP', 'Truncated', 'AUC_trapz', gdsc1_ctrpv2_trunc, 'AUC_trapz_gdsc1', 'AUC_trapz_ctrpv2'),
        ('GDSCv1_CTRP', 'Truncated', 'AUC_sig_trunc', gdsc1_ctrpv2_trunc, 'AUC_sig_norm_gdsc1', 'AUC_sig_norm_ctrpv2'),
        # PRISM vs CTRP - Full
        ('PRISM_CTRP', 'Full', 'EC50', prism_ctrpv2_full, 'log_EC50_prism', 'log_EC50_ctrpv2'),
        ('PRISM_CTRP', 'Full', 'IC50', prism_ctrpv2_full, 'log_IC50_prism', 'log_IC50_ctrpv2'),
        ('PRISM_CTRP', 'Full', 'AUC_sig', prism_ctrpv2_full, 'AUC_sig_prism', 'AUC_sig_ctrpv2'),
        ('PRISM_CTRP', 'Full', 'AUC_trapz', prism_ctrpv2_full, 'AUC_trapz_prism', 'AUC_trapz_ctrpv2'),
        ('PRISM_CTRP', 'Full', 'AUC_sig_trunc', prism_ctrpv2_full, 'AUC_sig_norm_prism', 'AUC_sig_norm_ctrpv2'),
        # PRISM vs CTRP - Truncated
        ('PRISM_CTRP', 'Truncated', 'EC50', prism_ctrpv2_trunc, 'log_EC50_prism', 'log_EC50_ctrpv2'),
        ('PRISM_CTRP', 'Truncated', 'IC50', prism_ctrpv2_trunc, 'log_IC50_prism', 'log_IC50_ctrpv2'),
        ('PRISM_CTRP', 'Truncated', 'AUC_sig', prism_ctrpv2_trunc, 'AUC_sig_prism', 'AUC_sig_ctrpv2'),
        ('PRISM_CTRP', 'Truncated', 'AUC_trapz', prism_ctrpv2_trunc, 'AUC_trapz_prism', 'AUC_trapz_ctrpv2'),
        ('PRISM_CTRP', 'Truncated', 'AUC_sig_trunc', prism_ctrpv2_trunc, 'AUC_sig_norm_prism', 'AUC_sig_norm_ctrpv2')
    ]
    
    # Run in parallel - uses all CPU cores
    print("Starting bootstrap calculations...")
    all_corr_ci = Parallel(n_jobs=-1, verbose=10)(
        delayed(process_one_comparison)(db_pair, type_dose, response, df, col1, col2)
        for db_pair, type_dose, response, df, col1, col2 in all_comparisons
    )

# Convert to DataFrame
corr_ci_df = pd.DataFrame(all_corr_ci)


# Save to CSV
corr_ci_df.to_csv(args.output_path, index=False)