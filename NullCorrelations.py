import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata
from joblib import Parallel, delayed

parser = argparse.ArgumentParser()
parser.add_argument('--GDSC1_data', type=str, help='Path to GDSC1 response data')
parser.add_argument('--GDSC2_data', type=str, help='Path to GDSC2 response data')
parser.add_argument('--PRISM_data', type=str, help='Path to PRISM response data')
parser.add_argument('--CTRP_data', type=str, help='Path to CTRP response data')
parser.add_argument('--output_path', type=str, required=True)

args = parser.parse_args()

def per_drug_permutation_test_vectorized(db1_data, db2_data, metric='log_IC50',
                                         n_permutations=1000, min_cell_lines=3,
                                         seed=42):
    """
    Vectorized per-drug correlation with permutation test.
    Creates two distributions: observed per-drug correlations vs permuted.
    
    Parameters:
    -----------
    db1_data, db2_data : pd.DataFrame
        DataFrames with ['Drug ID', 'Cell Line', metric]
    metric : str
        Response metric to compare
    n_permutations : int
        Number of permutations per drug
    min_cell_lines : int
        Minimum cell lines required per drug for correlation
    
    Returns:
    --------
    dict with observed/permuted distributions and statistics
    """
    np.random.seed(seed)
    
    # Merge on shared drug-cell line pairs
    merged = pd.merge(
        db1_data[['Drug ID', 'Cell Line', metric]],
        db2_data[['Drug ID', 'Cell Line', metric]],
        on=['Drug ID', 'Cell Line'],
        suffixes=('_db1', '_db2')
    )
    merged = merged.dropna(subset=[f'{metric}_db1', f'{metric}_db2'])
    
    # ===== OBSERVED: Per-drug correlations =====
    observed_correlations = []
    drug_info = []
    
    for drug in merged['Drug ID'].unique():
        drug_data = merged[merged['Drug ID'] == drug]
        
        if len(drug_data) >= min_cell_lines:
            x = drug_data[f'{metric}_db1'].values
            y = drug_data[f'{metric}_db2'].values
            r, _ = spearmanr(x, y)
            observed_correlations.append(r)
            drug_info.append({
                'drug': drug,
                'n_cell_lines': len(drug_data),
                'x': x,
                'y': y
            })
    
    observed_correlations = np.array(observed_correlations)
    n_drugs = len(observed_correlations)
    
    if n_drugs == 0:
        return None
    
    # ===== PERMUTED: Vectorized per-drug permutation =====
    # Pre-allocate array for all permuted correlations
    # Shape: (n_drugs * n_permutations,)
    permuted_correlations = np.zeros(n_drugs * n_permutations)
    
    idx = 0
    for drug_dict in drug_info:
        x = drug_dict['x']
        y = drug_dict['y']
        n = len(x)
        
        # Pre-compute ranks
        x_ranks = rankdata(x)
        y_ranks = rankdata(y)
        
        # Center x ranks (doesn't change across permutations)
        x_centered = x_ranks - x_ranks.mean()
        x_std = np.sqrt(np.sum(x_centered**2))
        
        # Generate all permutation indices at once
        # Shape: (n_permutations, n_cell_lines)
        perm_indices = np.random.rand(n_permutations, n).argsort(axis=1)
        
        # Apply all permutations
        y_ranks_permuted = y_ranks[perm_indices]
        
        # Vectorized correlation computation
        y_centered = y_ranks_permuted - y_ranks_permuted.mean(axis=1, keepdims=True)
        y_stds = np.sqrt(np.sum(y_centered**2, axis=1))
        
        numerators = np.sum(x_centered * y_centered, axis=1)
        drug_perm_corrs = numerators / (x_std * y_stds)
        
        # Store in pre-allocated array
        permuted_correlations[idx:idx+n_permutations] = drug_perm_corrs
        idx += n_permutations
    
    # ===== Statistics =====
    mean_observed = np.mean(observed_correlations)
    median_observed = np.median(observed_correlations)
    mean_permuted = np.mean(permuted_correlations)
    
    # P-value: bootstrap test if mean observed > mean permuted
    # Resample from permuted distribution to create null for mean
    n_bootstrap = 10000
    null_means = np.zeros(n_bootstrap)
    
    for i in range(n_bootstrap):
        null_sample = np.random.choice(permuted_correlations, size=n_drugs, replace=True)
        null_means[i] = np.mean(null_sample)
    
    # Two-tailed p-value
    p_value = np.mean(null_means >= mean_observed)
    
    return {
        'observed_correlations': observed_correlations,
        'permuted_correlations': permuted_correlations,
        'mean_observed': mean_observed,
        'median_observed': median_observed,
        'mean_permuted': mean_permuted,
        'p_value': p_value,
        'n_drugs': n_drugs,
        'n_permutations': n_permutations,
        'metric': metric,
        'drug_info': drug_info  # For detailed analysis
    }

def run_all_per_drug_comparisons(databases_dict, metrics, n_permutations=1000):
    """
    Run all pairwise per-drug comparisons for all metrics.
    
    Parameters:
    -----------
    databases_dict : dict
        {'GDSCv2': df, 'PRISM': df, ...}
    metrics : list
        ['IC50', 'EC50', 'AUC_sig', 'AUC_trapz']
    """
    db_names = list(databases_dict.keys())
    results_summary = []
    all_results = {}
    
    for i, db1_name in enumerate(db_names):
        for db2_name in db_names[i+1:]:
            
            print(f"\n{'='*60}")
            print(f"Comparing {db1_name} vs {db2_name}")
            print(f"{'='*60}")
            
            for metric in metrics:
                print(f"  {metric}...", end=' ')
                
                result = per_drug_permutation_test_vectorized(
                    databases_dict[db1_name],
                    databases_dict[db2_name],
                    metric=metric,
                    n_permutations=n_permutations
                )
                
                if result is not None:
                    # Store full result
                    key = f"{db1_name}_vs_{db2_name}_{metric}"
                    all_results[key] = result
                    
                    # Summary statistics
                    results_summary.append({
                        'DB1': db1_name,
                        'DB2': db2_name,
                        'Metric': metric,
                        'Mean_r': result['mean_observed'],
                        'Median_r': result['median_observed'],
                        'P_value': result['p_value'],
                        'N_drugs': result['n_drugs'],
                        'N_permutations': n_permutations
                    })
                    
                    print(f"r={result['mean_observed']:.2f}, p={result['p_value']:.4f}, n={result['n_drugs']}")
                else:
                    print("No shared drugs")
    
    return pd.DataFrame(results_summary), all_results


# Load data
databases = {
    'GDSCv1': pd.read_csv(args.GDSC1_data),
    'GDSCv2': pd.read_csv(args.GDSC2_data),
    'PRISM': pd.read_csv(args.PRISM_data),
    'CTRP': pd.read_csv(args.CTRP_data)
}

metrics = ['log_IC50', 'log_EC50', 'AUC_sig', 'AUC_trapz']

# Run all comparisons
summary_df, all_results = run_all_per_drug_comparisons(
    databases, 
    metrics, 
    n_permutations=1000
)

# Save summary
summary_df.to_csv(args.output_path, index=False)