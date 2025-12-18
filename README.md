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
See the original implementation of fit_curve.py from PNNL-CompBio/coderdata: v2.1.0. https://doi.org/10.5281/zenodo.17290565. 

## Usage
1. Calculate drug response profiles for the full and truncated dose ranges with:
   `CODER_drugscoring_submit.sh`
2. Calculate database pair Pearson Correlations with:
   `PerDatabaseCorrelations.py`
3. Generate Pearson correlations between permuted response profiles for empirical null distributions with:
   `NullCorrelations.py`

