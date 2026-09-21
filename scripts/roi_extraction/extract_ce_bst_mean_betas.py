#!/usr/bin/env python3
"""
Extract mean beta values within the Ce (CIT168 partial-volume-matched amygdala)
and BST (Blackford BNST) anatomical masks, for a fixed set of first-level
regressors, across subjects.

Regressor -> first-level subdirectory/filename mapping is fixed in
REGRESSOR_CONFIG below (previously run one regressor at a time by editing and
re-running this script; now processes all of them in a single run).

Inputs:
    Per-subject first-level beta maps under
        BASE_PATH/<subject>/<first_level_subdir>/<regressor filename>
    Ce and BST binary ROI masks (CE_MASK_PATH, BST_MASK_PATH)

Outputs (one pair of files per first_level_subdir, under OUT_BASE_DIR/<first_level_subdir>/):
    ce_bst_mean_betas_long.csv   one row per (subject, regressor, region) -> mean beta
    ce_bst_mean_betas_wide.csv   one row per subject, one column per regressor_region

Regressors are grouped by their first_level_subdir, so contrasts with their own
dedicated GLM folder (e.g. 50% vs 0%) get their own output file, while the
pooled trio (Probability/Risk/Ambiguity) -- which come from the same combined
GLM and are reported together -- share one file.
"""

import os
import sys
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import nibabel as nib

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.roi_utils import build_beta_df, load_mask_bool, nanmean_in_mask, check_same_shape

# =====================================
# Paths: edit these for your own system/data layout
# =====================================

BASE_PATH = "/quobyte/dfoxgrp/BACKED-UP/RDC/data/InitialAnalysisStartN104_20241111"

CE_MASK_PATH = "/quobyte/dfoxgrp/BACKED-UP/RDC/data/pe_analysis/ROIs/CIT168_Bilateral_1mm_MNI_partial_vol_matched_2mmResize_gt25.nii.gz"
BST_MASK_PATH = "/quobyte/dfoxgrp/BACKED-UP/RDC/data/pe_analysis/ROIs/Blackford_Bilateral_BNST_3T_ANTS_081216_2mmResize_gt25.nii.gz"

OUT_BASE_DIR = "/quobyte/dfoxgrp/BACKED-UP/RDC/data/MDART_replication_mesude/derived/fmri/group_level"

# label -> (first_level_subdir, regressor filename)
REGRESSOR_CONFIG: Dict[str, Tuple[str, str]] = {
    "Probability (risk/certain trials)": ("baeta_maps_fullNoPE_fullnus_062625_unsmoothed", "cue_p_risk_c0.nii.gz"),
    '"Risk" (Var[p-threat])': ("baeta_maps_fullNoPE_fullnus_062625_unsmoothed", "cue_r_risk_c0.nii.gz"),
    "Ambiguity (unknown probability)": ("baeta_maps_fullNoPE_fullnus_062625_unsmoothed", "cue_a_amb_c0.nii.gz"),
    "50% vs 0% probability": ("baeta_maps_c50vs0_fullnus_070725_unsmoothed", "c50vs0_c0.nii.gz"),
    "10% vs 0% probability": ("baeta_maps_c10vs0_fullnus_070725_unsmoothed", "c10vs0_c0.nii.gz"),
    "Uncertain Threat vs Certain Threat": (
        "baeta_maps_UncertainThreatVsCertainThreat_fullnus_062625",
        "cue_UncertainThreat_vs_CertainThreat_c0.nii.gz",
    ),
}

# Regressor filenames whose sign should be flipped before averaging (none currently).
NEGATIVE_REGRESSORS: List[str] = []


def extract_ce_bst_means(
    beta_dfs: Dict[str, pd.DataFrame],
    regressor_config: Dict[str, Tuple[str, str]],
    ce_mask: np.ndarray,
    bst_mask: np.ndarray,
    negative_regressors: List[str],
) -> pd.DataFrame:
    rows = []
    negative_regressors = set(negative_regressors)

    for reg_label, (subdir, reg_file) in regressor_config.items():
        beta_df = beta_dfs[subdir]

        for subj in beta_df.index:
            fp = beta_df.loc[subj, reg_file]

            if pd.isna(fp) or not os.path.exists(str(fp)):
                rows.extend([
                    (subj, reg_label, "ce", np.nan),
                    (subj, reg_label, "bst", np.nan),
                ])
                continue

            img = nib.load(fp).get_fdata()
            check_same_shape(img, ce_mask, f"{subj}/{reg_label} (ce_mask)")
            check_same_shape(img, bst_mask, f"{subj}/{reg_label} (bst_mask)")

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
        if os.path.isdir(os.path.join(BASE_PATH, d)) and d.startswith("RDC")
    )
    print(f"[INFO] Found {len(subject_ids)} subjects under {BASE_PATH}")

    print("[INFO] Building beta_df per first-level subdirectory...")
    subdirs_needed = list(dict.fromkeys(subdir for subdir, _ in REGRESSOR_CONFIG.values()))
    beta_dfs: Dict[str, pd.DataFrame] = {}
    for subdir in subdirs_needed:
        beta_dfs[subdir] = build_beta_df(
            base_path=BASE_PATH,
            subject_ids=subject_ids,
            first_level_subdir=subdir,
            regressor_names=None,
        )
        print(f"[INFO] beta_df shape: {beta_dfs[subdir].shape} (subjects x regressors) from {subdir}")

    missing = [
        (label, subdir, fname)
        for label, (subdir, fname) in REGRESSOR_CONFIG.items()
        if fname not in beta_dfs[subdir].columns
    ]
    if missing:
        raise ValueError(
            f"These REGRESSOR_CONFIG entries were not found as beta_df columns: {missing}\n"
            f"Check the mapping against files actually present under BASE_PATH."
        )

    print("[INFO] Loading ROI masks...")
    ce_mask = load_mask_bool(CE_MASK_PATH)
    bst_mask = load_mask_bool(BST_MASK_PATH)

    # Group regressors by their first_level_subdir, so each subdir gets its own
    # output file (the pooled trio shares a subdir, so they naturally share one).
    regressors_by_subdir: Dict[str, Dict[str, Tuple[str, str]]] = {}
    for label, (subdir, fname) in REGRESSOR_CONFIG.items():
        regressors_by_subdir.setdefault(subdir, {})[label] = (subdir, fname)

    for subdir, subdir_regressor_config in regressors_by_subdir.items():
        print(f"[INFO] Extracting ROI means for {subdir} ({list(subdir_regressor_config)})...")
        df_long = extract_ce_bst_means(
            beta_dfs=beta_dfs,
            regressor_config=subdir_regressor_config,
            ce_mask=ce_mask,
            bst_mask=bst_mask,
            negative_regressors=NEGATIVE_REGRESSORS,
        )

        out_dir = os.path.join(OUT_BASE_DIR, subdir)
        os.makedirs(out_dir, exist_ok=True)

        out_long_csv = os.path.join(out_dir, "ce_bst_mean_betas_long.csv")
        df_long.to_csv(out_long_csv, index=False)
        print(f"[OK] Saved long df: {out_long_csv}")

        out_wide_csv = os.path.join(out_dir, "ce_bst_mean_betas_wide.csv")
        df_wide = long_to_wide(df_long)
        df_wide.to_csv(out_wide_csv, index=False)
        print(f"[OK] Saved wide df: {out_wide_csv}")


if __name__ == "__main__":
    main()
