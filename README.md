# Harmonization and Integration of Pharmacogenomics Screens

## Overview
This repo provides the code to calculate reproducibility, as measured by Pearson Correlation, between cancer pharmacogenomics databases. Further details of how this code was implemented can be found in "Harmonization and Integration of Pharmacogenomics Screens".

## Environment Set-Up
Use `environment.yml` to set up a virtual environment (harmonization) with needed dependencies.
```
conda env create -f environment.yml -n harmonization
```

## Data Availability 
Raw input data for calculating drug response profiles are downloadable from: [Raw input data](https://zenodo.org/records/17196024)    
Recalculated drug response profiles from the truncated dose range are downladable from: [Truncated drug response](https://doi.org/10.5281/zenodo.17194793)

## Drug Response Profile Calculation
Using replicate-collapsed relative viabilities as input, run the following command to calculate drug response profiles: 
```
python DrugResponseProfileFitCalculation.py
   --data
   --drug_col
   --cell_line_col
   --dose_col
   --viability_col
   --truncate
   --force_ic50
   --output_path
```
**Arguments**
- `--data`: Path to input file containing replicate-collapsed relative viabilities
- `--drug_col`: Name of column containing drug names
- `--cell_line_col`: Name of column containing cell line names
- `--dose_col`: Name of column containing dose values
- `--viability_col`: Name of column containing relative viabilities
- `--truncate`: Include flag to truncate dose range (optional)
- `--force_ic50`: Include flag to include all IC50 values
- `--output_path`: Path to store output file 

## Cross-database Spearman Correlation Calculation
Using full- and truncated-range drug response profiles for each database as input, run the following command to calculate cross-database Spearman correlations:
```
python PerDatabaseCorrelations.py
   --gdsc2_full 
   --gdsc2_trunc
   --prism_full
   --prism_trunc
   --gdsc1_full
   --gdsc1_trunc
   --ctrp_full
   --ctrp_trunc
   --compare_ic50
   --compare_all_effective_drugs
   --output_path
```
**Arguments**
- `--gdsc2_full`: Path to input file containing GDSCv2 full-range drug response profiles
- `--gdsc2_trunc`: Path to input file containing GDSCv2 truncated-range drug response profiles
- `--prism_full`: Path to input file containing PRISM full-range drug response profiles
- `--prism_trunc`: Path to input file containing PRISM truncated-range drug response profiles
- `--gdsc1_full`: Path to input file containing GDSCv1 full-range drug response profiles
- `--gdsc1_trunc`: Path to input file containing GDSCv1 truncated-range drug response profiles
- `--ctrp_full`: Path to input file containing CTRP full-range drug response profiles
- `--ctrp_trunc`: Path to input file containing CTRP truncated-range drug response profiles
- `--compare_ic50`: Include flag to calculate correlations for all IC50 vs. 1nM < IC50 < 100 uM (optional)
- `--compare_all_effective_drugs`: Include flag to calculate correlations for all drugs vs. effective drugs (optional)
- `--output_path`: Path to store output file 

For specifically successive releases of PRISM screens, use full- and truncated-range drug response profiles for each screen as input, and run the following command to calculate cross-dataset Spearman correlations:
```
python PerPRISMScreenCorrelations.py
   --prism_public_full
   --prism_public_trunc
   --mts021_full
   --mts021_trunc
   --mts022_full
   --mts022_trunc
   --mts023_full
   --mts023_trunc
   --mts024_full
   --mts024_trunc
   --mts025_full
   --mts025_trunc
   --mts026_full
   --mts026_trunc
   --output_path
```
**Arguments**
- `--prism_public_full`: Path to input file containing full-range drug response profiles from the published PRISM screen
- `--prism_public_trunc`: Path to input file containing truncated-range drug response profiles from the published PRISM screen
- `--mts021_full`: Path to input file containing full-range drug response profiles from PRISM Run 21
- `--mts021_trunc`: Path to input file containing truncated-range drug response profiles from PRISM Run 21
- `--mts022_full`: Path to input file containing full-range drug response profiles from PRISM Run 22
- `--mts022_trunc`: Path to input file containing truncated-range drug response profiles from PRISM Run 22
- `--mts023_full`: Path to input file containing full-range drug response profiles from PRISM Run 23
- `--mts023_trunc`: Path to input file containing truncated-range drug response profiles from PRISM Run 23
- `--mts024_full`: Path to input file containing full-range drug response profiles from PRISM Run 24
- `--mts024_trunc`: Path to input file containing truncated-range drug response profiles from PRISM Run 24
- `--mts025_full`: Path to input file containing full-range drug response profiles from PRISM Run 25
- `--mts025_trunc`: Path to input file containing truncated-range drug response profiles from PRISM Run 25
- `--mts026_full`: Path to input file containing full-range drug response profiles from PRISM Run 26
- `--mts026_trunc`: Path to input file containing truncated-range drug response profiles from PRISM Run 26
- `--output_path`: Path to store output file 


## Permuted Spearman Correlation Calculation

Using drug response profiles for each database as input, run the following command to calculate permuted Spearman correlations:
```
python NullCorrelations.py
   --GDSC1_data
   --GDSC2_data
   --PRISM_data
   --CTRP_data
   --output_path
```
**Arguments**
- `--GDSC1_data`: Path to input file containing GDSCv1 drug response profiles
- `--GDSC2_data`: Path to input file containing GDSCv2 drug response profiles
- `--PRISM_data`: Path to input file containing PRISM drug response profiles
- `--CTRP_data`: Path to input file containing CTRP drug response profiles
- `--output_path`: Path to store output file

