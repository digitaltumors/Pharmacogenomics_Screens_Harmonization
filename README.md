# Pharmacogenomic_Screens_Harmonization

## Overview
This repo provides the code to calculate reproducibility, as measured by Pearson Correlation, between cancer Pharmacogenomics databases. Details of how this code was implemented in "Harmonization and Integration of Pharmacogenomics Screens".

## Environment Set-Up
Use environment.yml provided to install dependencies
```
conda env create python== --name harmonization --file=environment.yml
```
## Usage
1. Calculate drug response profiles for the full and truncated dose response with:
   CODER_drugscoring_submit.sh
2. Calculate database pair Pearson Correlations with:
   PerDatabaseCorrelations.py
3. Generate null Pearson correlations by shuffling the cell line identifiers of one database with:
   NullCorrelations.py
