#!/bin/bash

#SBATCH --job-name=
#SBATCH --partition=
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --time=04-00:00:00
#SBATCH --output={path/to/out}.out

set -euo pipefail

# ========= EDIT THESE =========
INPUT="/path/to/input/INPUT.csv"
OUTPUT="/path/to/output/OUTPUT_SCORES.csv"
SCRIPT="/path/to/coder/script/fit_curve.py"

# Toggle filtering: 0 = off, 1 = on (keeps 0.03–10 µM)
FILTER_DOSE=1

# Column name for dose in your CSV
DOSE_COL="dose"

# Is DOSE_COL already in log10(µM)? 1=yes, 0=no (i.e., linear µM)
IS_LOG10=1
# ==============================

# Env
source /cellar/users/nmattson/miniconda3/etc/profile.d/conda.sh
conda activate jupyter

# Compute log10 bounds for 0.03–10 µM
LOG10_MIN="-1.5228787452803376"  # log10(0.03)
LOG10_MAX="1.0"                  # log10(10)

INPUT_TO_USE="$INPUT"

if [[ "$FILTER_DOSE" -eq 1 ]]; then
  TMPDIR="${TMPDIR:-/tmp}"
  FILTERED_CSV="$TMPDIR/filtered_${SLURM_JOB_ID}.csv"

  python - <<PY
import sys, math
import pandas as pd

inp = "$INPUT"
outp = "$FILTERED_CSV"
dose_col = "$DOSE_COL"
is_log10 = ${IS_LOG10}
log10_min = float("$LOG10_MIN")
log10_max = float("$LOG10_MAX")

df = pd.read_csv(inp)

if dose_col not in df.columns:
    raise SystemExit(f"ERROR: column '{dose_col}' not in {inp}. Found: {list(df.columns)[:12]}...")

# Build a log10 dose series for filtering
if is_log10 == 1:
    logdose = pd.to_numeric(df[dose_col], errors="coerce")
else:
    # assume linear µM -> convert to log10(µM)
    lin = pd.to_numeric(df[dose_col], errors="coerce")
    logdose = lin.apply(lambda v: math.log10(v) if pd.notnull(v) and v>0 else float("nan"))

mask = (logdose >= log10_min) & (logdose <= log10_max)
df2 = df.loc[mask].copy()
df2.to_csv(outp, index=False)

kept = int(mask.sum())
total = int(mask.notna().sum())
print(f"[filter] kept {kept}/{total} rows between log10 dose [{log10_min}, {log10_max}]")
PY

  INPUT_TO_USE="$FILTERED_CSV"
fi

# === Run your existing script ===
# If your script *doesn't* take args and discovers INPUT/OUTPUT internally,
# just leave this line as-is:
python -u "$SCRIPT"

# If your script supports explicit args like --input/--output, you can instead do:
# python -u "$SCRIPT" --input "$INPUT_TO_USE" --output "$OUTPUT"

echo "Done. Used input: $INPUT_TO_USE"