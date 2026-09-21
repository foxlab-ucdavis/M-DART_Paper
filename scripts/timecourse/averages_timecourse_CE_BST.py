"""
Aggregate per-subject Ce/BST ROI timecourse betas (output of
timecourse_analysis_BST_CE_mean_roi.py) into one subject-level long file and
one group-level (mean/SEM across subjects) file.

Inputs:
    SUBJ_BASE/<subject>/Timecourse_MDART_CE_BST_mean_roi/
        <subject>_betas_timecourse.csv

Outputs (under SUBJ_BASE/MDART_timecourse_CE_BST_mean_roi/):
    MDART_timecourse_CE_BST_mean_roi_subject_level_betas_timecourse.csv
    MDART_timecourse_CE_BST_mean_roi_group_level_betas_timecourse.csv
"""

import os
import glob
import pandas as pd

# -----------------------
# Paths: edit for your own system/data layout
# -----------------------
SUBJ_BASE = r"/quobyte/dfoxgrp/BACKED-UP/RDC/data/pe_analysis/pe_output"

# -----------------------
# 2) Discover per-subject CSVs
# -----------------------
subject_dirs = sorted(glob.glob(os.path.join(SUBJ_BASE, "RDC*")))
csv_paths = []
missing = []
multi_match = []

for sdir in subject_dirs:
    sub_id = os.path.basename(sdir)
    timecourse_dir = os.path.join(sdir, "Timecourse_MDART_CE_BST_mean_roi")
    found = glob.glob(os.path.join(timecourse_dir, "*_betas_timecourse.csv"))

    if len(found) == 0:
        missing.append(sub_id)
        continue
    if len(found) > 1:
        found.sort(key=os.path.getmtime, reverse=True)
        multi_match.append((sub_id, found))

    csv_paths.append(found[0])

print(
    f"Found {len(csv_paths)} subject file(s). "
    f"Missing {len(missing)}: {missing[:10]}",
    "..." if len(missing) > 10 else ""
)
if multi_match:
    print(f"Subjects with multiple matches (kept most recent): {[s for s, _ in multi_match]}")

if not csv_paths:
    raise FileNotFoundError("No subject beta timecourse CSVs found.")

# -----------------------
# 3) Load & concatenate
# -----------------------
dfs = []
required_cols = {"time_s", "subject", "alignment", "roi"}

for p in csv_paths:
    df = pd.read_csv(p)
    have = set(df.columns)
    if not required_cols.issubset(have):
        missing_cols = sorted(required_cols - have)
        raise ValueError(f"{p} is missing columns: {missing_cols}")
    dfs.append(df)

df_all = pd.concat(dfs, ignore_index=True)

# -----------------------
# 3a) Identify beta columns dynamically
# -----------------------
beta_cols = [c for c in df_all.columns if c.startswith("beta_")]
print("Beta columns found:", beta_cols)

if not beta_cols:
    raise ValueError("No beta columns found in concatenated data.")

# -----------------------
# 3b) SANITY CHECK
# one row per subject × alignment × roi × time
# -----------------------
check = (
    df_all
    .groupby(["time_s", "subject", "alignment", "roi"])
    .size()
    .reset_index(name="n_rows")
)

print("Sanity check n_rows counts:")
print(check["n_rows"].value_counts().sort_index())

# -----------------------
# 3c) Save subject-level long file
# -----------------------
out_dir = os.path.join(SUBJ_BASE, "MDART_timecourse_CE_BST_mean_roi")
os.makedirs(out_dir, exist_ok=True)

out_csv_subject = os.path.join(
    out_dir,
    "MDART_timecourse_CE_BST_mean_roi_subject_level_betas_timecourse.csv"
)
df_all.to_csv(out_csv_subject, index=False)
print(f"Saved subject-level long timecourse to:\n{out_csv_subject}")

# -----------------------
# 4) Average across subjects
# -----------------------
agg_dict = {"subject": ("subject", "nunique")}
for col in beta_cols:
    agg_dict[f"mean_{col}"] = (col, "mean")
    agg_dict[f"sem_{col}"] = (col, "sem")

df_group = (
    df_all
    .groupby(["time_s", "alignment", "roi"], as_index=False)
    .agg(**agg_dict)
    .rename(columns={"subject": "n_subjects"})
    .sort_values(["roi", "alignment", "time_s"])
)

# -----------------------
# 5) Save output
# -----------------------
out_csv_group = os.path.join(
    out_dir,
    "MDART_timecourse_CE_BST_mean_roi_group_level_betas_timecourse.csv"
)
df_group.to_csv(out_csv_group, index=False)
print(f"Saved group-average timecourse to:\n{out_csv_group}")