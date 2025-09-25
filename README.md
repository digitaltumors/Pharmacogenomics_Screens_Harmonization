# Harmonization and Integration of Pharmacogenomics Screens

## Overview
This repo provides the code to calculate reproducibility, as measured by Pearson Correlation, between cancer pharmacogenomics databases. Further details of how this code was implemented can be found in "Harmonization and Integration of Pharmacogenomics Screens".

## Environment Set-Up
Use `environment.yml` to install needed dependencies.
```
conda env create python== --name harmonization --file=environment.yml
```

## Data Availability 
Raw input data for calculating drug response profiles are downloadable from: [Raw input data](https://doi.org/10.5281/zenodo.17196025)    
Recalculated drug response profiles from the truncated dose range are downladable from: [Truncated drug response](https://doi.org/10.5281/zenodo.17194793)

## Usage
1. Calculate drug response profiles for the full and truncated dose ranges with:
   `CODER_drugscoring_submit.sh`
2. Calculate database pair Pearson Correlations with:
   `PerDatabaseCorrelations.py`
3. Generate Pearson correlations between permuted response profiles for empirical null distributions with:
   `NullCorrelations.py`
