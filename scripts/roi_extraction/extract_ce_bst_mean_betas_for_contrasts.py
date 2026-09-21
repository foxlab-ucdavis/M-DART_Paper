#!/usr/bin/env python3
"""
Extract mean beta values within the Ce and BST anatomical masks for the four
adjacent probability-bin contrasts (10-30 vs 0, 40-60 vs 10-30, 70-90 vs 40-60,
100 vs 70-90), across subjects, plus a group-level summary (mean/SEM/CI).

Inputs:
    Per-subject first-level contrast beta maps under
        BASE_PATH/<subject>/<first_level_subdir>/<contrast filename>
        (see CONTRAST_CONFIG for the exact mapping)
    Ce and BST binary ROI masks (CE_MASK_PATH, BST_MASK_PATH)

Outputs:
    OUT_SUBJECT_CSV  one row per (subject, contrast, region) -> mean beta
    OUT_GROUP_CSV    one row per (contrast, region) -> n, mean, SEM, CI95, CI68
"""

import os
import sys
from typing import List, Dict, Tuple, Set

import numpy as np
import pandas as pd
import nibabel as nib
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.roi_utils import load_mask_bool, nanmean_in_mask, check_same_shape

# =====================================
# Paths: edit these for your own system/data layout
# =====================================

BASE_PATH = "/quobyte/dfoxgrp/BACKED-UP/RDC/data/InitialAnalysisStartN104_20241111"

CE_MASK_PATH = "/quobyte/dfoxgrp/BACKED-UP/RDC/data/pe_analysis/ROIs/CIT168_Bilateral_1mm_MNI_partial_vol_matched_2mmResize_gt25.nii.gz"
BST_MASK_PATH = "/quobyte/dfoxgrp/BACKED-UP/RDC/data/pe_analysis/ROIs/Blackford_Bilateral_BNST_3T_ANTS_081216_2mmResize_gt25.nii.gz"

OUT_DIR = "/quobyte/dfoxgrp/BACKED-UP/RDC/data/MDART_replication_mesude/derived/fmri/group_level/ce_bst_mean_for_contrasts"
OUT_SUBJECT_CSV = os.path.join(OUT_DIR, "ce_bst_mean_betas_for_contrasts_subject_level.csv")
OUT_GROUP_CSV = os.path.join(OUT_DIR, "ce_bst_mean_betas_for_contrasts_group_summary.csv")

# label -> (first_level_subdir, contrast_filename)
CONTRAST_CONFIG: Dict[str, Tuple[str, str]] = {
    "10-30% vs 0%": ("baeta_maps_c102030vs0_fullnus_070725_unsmoothed", "c102030vs0_c0.nii.gz"),
    "40-60% vs 10-30%": ("baeta_maps_c405060vs102030_fullnus_070725_unsmoothed", "c405060vs102030_c0.nii.gz"),
    "70-90% vs 40-60%": ("baeta_maps_c708090vs405060_fullnus_070725_unsmoothed", "c708090vs405060_c0.nii.gz"),
    "100% vs 70-90%": ("baeta_maps_c100vs708090_fullnus_070725_unsmoothed", "c100vs708090_c0.nii.gz"),
}

NEGATIVE_REGRESSORS: Set[str] = set()  # add labels here if sign should be flipped


# =====================================
# Functions
# =====================================

def get_subject_ids(base_path: str, contrast_config: Dict[str, Tuple[str, str]]) -> List[str]:
    all_subdirs = {subdir for subdir, _ in contrast_config.values()}
    return sorted(
        d for d in os.listdir(base_path)
        if d.startswith("RDC") and any(
            os.path.isdir(os.path.join(base_path, d, subdir))
            for subdir in all_subdirs
        )
    )


def extract_ce_bst_means(
    base_path: str,
    subject_ids: List[str],
    contrast_config: Dict[str, Tuple[str, str]],
    ce_mask: np.ndarray,
    bst_mask: np.ndarray,
    negative_regressors: Set[str],
) -> pd.DataFrame:
    rows = []

    for reg_label, (subdir, reg_file) in contrast_config.items():
        for subj in subject_ids:
            fp = os.path.join(base_path, subj, subdir, reg_file)

            if not os.path.exists(fp):
                rows.extend([
                    (subj, reg_label, "ce", np.nan),
                    (subj, reg_label, "bst", np.nan),
                ])
                continue

            img = nib.load(fp).get_fdata()
            check_same_shape(img, ce_mask, f"{subj}/{reg_label} vs ce_mask")
            check_same_shape(img, bst_mask, f"{subj}/{reg_label} vs bst_mask")

            if reg_label in negative_regressors:
                img = -img

            rows.append((subj, reg_label, "ce", nanmean_in_mask(img, ce_mask)))
            rows.append((subj, reg_label, "bst", nanmean_in_mask(img, bst_mask)))

    return pd.DataFrame(rows, columns=["subject", "regressor", "region", "mean_beta"])


def compute_group_summary(df_long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (reg, region), group in df_long.groupby(["regressor", "region"]):
        vals = group["mean_beta"].dropna()
        n = len(vals)
        mean = vals.mean()
        sem = stats.sem(vals)
        ci95 = stats.t.interval(0.95, df=n - 1, loc=mean, scale=sem)
        ci68 = stats.t.interval(0.68, df=n - 1, loc=mean, scale=sem)
        rows.append({
            "regressor": reg,
            "region": region,
            "n": n,
            "mean": mean,
            "sem": sem,
            "ci95_lower": ci95[0],
            "ci95_upper": ci95[1],
            "ci68_lower": ci68[0],
            "ci68_upper": ci68[1],
        })
    return pd.DataFrame(rows)


# =====================================
# Main
# =====================================

def main():
    print("[INFO] Loading subject list...")
    subject_ids = get_subject_ids(BASE_PATH, CONTRAST_CONFIG)
    print(f"[INFO] Found {len(subject_ids)} subjects")

    print("[INFO] Loading ROI masks...")
    ce_mask = load_mask_bool(CE_MASK_PATH)
    bst_mask = load_mask_bool(BST_MASK_PATH)

    # sanity-check mask shapes against one subject's image
    did_check = False
    for reg_label, (subdir, reg_file) in CONTRAST_CONFIG.items():
        for subj in subject_ids:
            fp = os.path.join(BASE_PATH, subj, subdir, reg_file)
            if os.path.exists(fp):
                img = nib.load(fp).get_fdata()
                check_same_shape(img, ce_mask, "ce_mask")
                check_same_shape(img, bst_mask, "bst_mask")
                did_check = True
                break
        if did_check:
            break

    if not did_check:
        raise RuntimeError("[ERROR] Could not find any existing beta map. Check paths in CONTRAST_CONFIG.")

    print("[INFO] Extracting ROI means...")
    df_long = extract_ce_bst_means(
        base_path=BASE_PATH,
        subject_ids=subject_ids,
        contrast_config=CONTRAST_CONFIG,
        ce_mask=ce_mask,
        bst_mask=bst_mask,
        negative_regressors=NEGATIVE_REGRESSORS,
    )

    os.makedirs(OUT_DIR, exist_ok=True)

    df_long.to_csv(OUT_SUBJECT_CSV, index=False)
    print(f"[OK] Saved subject-level df: {OUT_SUBJECT_CSV}")

    df_group = compute_group_summary(df_long)
    df_group.to_csv(OUT_GROUP_CSV, index=False)
    print(f"[OK] Saved group summary df: {OUT_GROUP_CSV}")


if __name__ == "__main__":
    main()
