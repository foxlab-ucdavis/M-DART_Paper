"""
Aggregate per-subject whole-brain signature-expression timecourse betas
(output of timecourse_analysis_signature_expression.py) into one subject-level
long file and one group-level (mean/SEM across subjects) file.

Inputs:
    SUBJ_BASE/<subject>/MDART_timecourse_wholebrain_signature_expression/
        <subject>_betas_timecourse_MDART_wholebrain_sig_exp.csv

Outputs (under SUBJ_BASE/MDART_timecourse_wholebrain_signature_expression/):
    updated_MDART_timecourse_wholebrain_signature_expression_subject_level_betas_timecourse.csv
    updated_MDART_timecourse_wholebrain_signature_expression_group_level_betas_timecourse.csv
"""

import os
import glob
import pandas as pd

# -----------------------
# Paths: edit for your own system/data layout
# -----------------------
SUBJ_BASE = r"/quobyte/dfoxgrp/BACKED-UP/RDC/data/pe_analysis/pe_output"

# -----------------------
# 1) Discover per-subject CSVs
# -----------------------
subject_dirs = sorted(glob.glob(os.path.join(SUBJ_BASE, "RDC*")))
csv_paths = []
missing = []
multi_match = []

for sdir in subject_dirs:
    sub_id = os.path.basename(sdir)
    timecourse_dir = os.path.join(sdir, "MDART_timecourse_wholebrain_signature_expression")
    found = glob.glob(os.path.join(timecourse_dir, "*_betas_timecourse_MDART_wholebrain_sig_exp.csv"))

    if len(found) == 0:
        missing.append(sub_id)
        continue
    if len(found) > 1:
        # keep the most recent if duplicates exist
        found.sort(key=os.path.getmtime, reverse=True)
        multi_match.append((sub_id, found))

    csv_paths.append(found[0])

print(f"Found {len(csv_paths)} subject file(s). Missing {len(missing)}:", missing[:10], "..." if len(missing) > 10 else "")
if multi_match:
    print(f"Subjects with multiple matches (kept most recent): {[s for s, _ in multi_match]}")

if not csv_paths:
    raise FileNotFoundError("No *_betas_timecourse_MDART_wholebrain_sig_exp.csv files found. Check the directory structure and filenames.")

# -----------------------
# 2) Load & concatenate
# -----------------------
dfs = []
required_cols = {"time_s", "subject", "alignment", "signature"}

for p in csv_paths:
    df = pd.read_csv(p)
    have = set(df.columns)
    if not required_cols.issubset(have):
        missing_cols = sorted(required_cols - have)
        raise ValueError(f"{p} is missing columns: {missing_cols}")
    dfs.append(df)

df_all = pd.concat(dfs, ignore_index=True)

# -----------------------
# 2a) Identify beta columns dynamically
# -----------------------
beta_cols = [c for c in df_all.columns if c.startswith("beta_")]
print("Beta columns found:", beta_cols)

if not beta_cols:
    raise ValueError("No beta columns found in concatenated data.")

# -----------------------
# 2b) SANITY CHECK: one row per subject x alignment x signature x time
# -----------------------
check = (
    df_all
    .groupby(["time_s", "subject", "alignment", "signature"])
    .size()
    .reset_index(name="n_rows")
)

print("Sanity check n_rows counts:")
print(check["n_rows"].value_counts().sort_index())

# -----------------------
# 2c) Save subject-level long file (this is what you'll use for stats)
# -----------------------
out_dir = os.path.join(SUBJ_BASE, "MDART_timecourse_wholebrain_signature_expression")
os.makedirs(out_dir, exist_ok=True)

out_csv_subject = os.path.join(
    out_dir,
    "updated_MDART_timecourse_wholebrain_signature_expression_subject_level_betas_timecourse.csv",
)
df_all.to_csv(out_csv_subject, index=False)
print(f"Saved subject-level long timecourse to:\n{out_csv_subject}")

# -----------------------
# 3) Average across subjects
# -----------------------
agg_dict = {"subject": ("subject", "nunique")}
for col in beta_cols:
    agg_dict[f"mean_{col}"] = (col, "mean")
    agg_dict[f"sem_{col}"] = (col, "sem")

df_group = (
    df_all
    .groupby(["time_s", "alignment", "signature"], as_index=False)
    .agg(**agg_dict)
    .rename(columns={"subject": "n_subjects"})
    .sort_values(["signature", "alignment", "time_s"])
)

# -----------------------
# 4) Save output
# -----------------------
out_csv_group = os.path.join(
    out_dir,
    "updated_MDART_timecourse_wholebrain_signature_expression_group_level_betas_timecourse.csv",
)
df_group.to_csv(out_csv_group, index=False)
print(f"Saved group-average timecourse to:\n{out_csv_group}")
