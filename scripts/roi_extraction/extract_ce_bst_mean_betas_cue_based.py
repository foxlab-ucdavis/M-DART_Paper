#!/usr/bin/env python3
"""
Extract mean beta values within the Ce and BST anatomical masks for every
regressor produced by the cueType first-level design (one regressor per
unique cue probability/ambiguity level), across subjects.

Unlike extract_ce_bst_mean_betas.py (a fixed, named regressor list), this
script auto-discovers every beta map in FIRST_LEVEL_SUBDIR and extracts all
of them. Downstream code (Figure 2, Panels c-d) selects the regressors it
needs via regex on the "regressor" column (bare non-ambiguous probability
levels like "p010_c0" and ambiguity levels like "a010_c0"); other regressors
in this subdir (e.g. cue_risk_const, outcome, rating, or the "pNNN_amb"
ambiguous-trial probability labels) are extracted too but simply go unused
by that filter.

Inputs:
    Per-subject first-level beta maps under
        BASE_PATH/<subject>/<FIRST_LEVEL_SUBDIR>/<regressor filename>
    Ce and BST binary ROI masks (CE_MASK_PATH, BST_MASK_PATH)

Outputs (under OUT_DIR = OUT_BASE_DIR/FIRST_LEVEL_SUBDIR):
    ce_bst_mean_betas_long.csv   one row per (subject, regressor, region) -> mean beta
    ce_bst_mean_betas_wide.csv   one row per subject, one column per regressor_region
"""

import os
import sys
from typing import List, Optional

import numpy as np
import pandas as pd
import nibabel as nib

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.roi_utils import build_beta_df, load_mask_bool, nanmean_in_mask, check_same_shape

# =====================================
# Paths: edit these for your own system/data layout
# =====================================

BASE_PATH = "/quobyte/dfoxgrp/BACKED-UP/RDC/data/InitialAnalysisStartN104_20241111"

FIRST_LEVEL_SUBDIR = "baeta_maps_cueType_fullnus_062625_noderivative_unsmoothed"

CE_MASK_PATH = "/quobyte/dfoxgrp/BACKED-UP/RDC/data/pe_analysis/ROIs/CIT168_Bilateral_1mm_MNI_partial_vol_matched_2mmResize_gt25.nii.gz"
BST_MASK_PATH = "/quobyte/dfoxgrp/BACKED-UP/RDC/data/pe_analysis/ROIs/Blackford_Bilateral_BNST_3T_ANTS_081216_2mmResize_gt25.nii.gz"

OUT_BASE_DIR = "/quobyte/dfoxgrp/BACKED-UP/RDC/data/MDART_replication_mesude/derived/fmri/group_level"
OUT_DIR = os.path.join(OUT_BASE_DIR, FIRST_LEVEL_SUBDIR)
OUT_LONG_CSV = os.path.join(OUT_DIR, "ce_bst_mean_betas_long.csv")
OUT_WIDE_CSV = os.path.join(OUT_DIR, "ce_bst_mean_betas_wide.csv")

# Regressor filenames whose sign should be flipped before averaging (none currently).
NEGATIVE_REGRESSORS: List[str] = []


def extract_ce_bst_means_all(
    beta_df: pd.DataFrame,
    ce_mask: np.ndarray,
    bst_mask: np.ndarray,
    negative_regressors: Optional[List[str]] = None,
) -> pd.DataFrame:
    rows = []
    negative_regressors = set(negative_regressors or [])

    for reg_file in beta_df.columns:  # all regressors found in FIRST_LEVEL_SUBDIR
        reg_label = reg_file.split(".nii")[0]

        for subj in beta_df.index:
            fp = beta_df.loc[subj, reg_file]

            if not isinstance(fp, str) or not os.path.exists(fp):
                rows.extend([
                    (subj, reg_label, "ce", np.nan),
                    (subj, reg_label, "bst", np.nan),
                ])
                continue

            img = nib.load(fp).get_fdata()
            check_same_shape(img, ce_mask, f"{subj}/{reg_file}")
            check_same_shape(img, bst_mask, f"{subj}/{reg_file} (bst_mask)")

            if reg_file in negative_regressors:
                img = -img

            rows.append((subj, reg_label, "ce", nanmean_in_mask(img, ce_mask)))
            rows.append((subj, reg_label, "bst", nanmean_in_mask(img, bst_mask)))

    return pd.DataFrame(rows, columns=["subject", "regressor", "region", "mean_beta"])


def long_to_wide(df_long: pd.DataFrame) -> pd.DataFrame:
    wide = (
        df_long
        .assign(col=lambda d: d["regressor"] + "_" + d["region"])
        .pivot(index="subject", columns="col", values="mean_beta")
        .reset_index()
    )
    wide.columns.name = None
    return wide


def main():
    print("[INFO] Loading subject list...")
    subject_ids = sorted(
        d for d in os.listdir(BASE_PATH)
        if d.startswith("RDC") and os.path.isdir(os.path.join(BASE_PATH, d, FIRST_LEVEL_SUBDIR))
    )
    print(f"[INFO] Found {len(subject_ids)} subjects under {BASE_PATH}")

    print("[INFO] Building beta_df...")
    beta_df = build_beta_df(
        base_path=BASE_PATH,
        subject_ids=subject_ids,
        first_level_subdir=FIRST_LEVEL_SUBDIR,
        regressor_names=None,  # infer all beta files from an example subject folder
    )

    print("[INFO] Loading ROI masks...")
    ce_mask = load_mask_bool(CE_MASK_PATH)
    bst_mask = load_mask_bool(BST_MASK_PATH)

    # sanity check once
    did_check = False
    for subj in subject_ids:
        for reg in beta_df.columns:
            fp = beta_df.loc[subj, reg]
            if isinstance(fp, str) and os.path.exists(fp):
                img = nib.load(fp).get_fdata()
                check_same_shape(img, ce_mask, "Ce mask")
                check_same_shape(img, bst_mask, "BST mask")
                did_check = True
                break
        if did_check:
            break

    if not did_check:
        raise RuntimeError(
            "[ERROR] Could not find any existing beta maps for the requested regressors "
            "to validate mask shapes. Check BASE_PATH/FIRST_LEVEL_SUBDIR/regressor filenames."
        )

    print("[INFO] Extracting ROI means...")
    df_long = extract_ce_bst_means_all(
        beta_df=beta_df,
        ce_mask=ce_mask,
        bst_mask=bst_mask,
        negative_regressors=NEGATIVE_REGRESSORS,
    )

    os.makedirs(OUT_DIR, exist_ok=True)
    df_long.to_csv(OUT_LONG_CSV, index=False)
    print(f"[OK] Saved long df: {OUT_LONG_CSV}")

    df_wide = long_to_wide(df_long)
    df_wide.to_csv(OUT_WIDE_CSV, index=False)
    print(f"[OK] Saved wide df: {OUT_WIDE_CSV}")


if __name__ == "__main__":
    main()
