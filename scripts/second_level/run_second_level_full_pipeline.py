"""
Second-level (group) modeling -> FDR correction -> pos/neg maps -> 1mm resampling.

Merges what used to be two separate steps (run_second_level.py +
prepare-group-level-t-maps.ipynb) into one pipeline so the FDR alpha, the
exact (not re-derived) p-values, and the degrees of freedom are shared and
consistent end to end.

Pipeline
--------
1. Build beta_df (subjects x regressors) from first-level beta maps on disk.
2. For each regressor of interest, run an intercept-only (one-sample t-test)
   voxelwise regression across subjects -> whole-brain t-map and p-map.
3. Save uncorrected t/p maps.
4. Apply BH-FDR correction directly on the exact p-maps from step 2:
     - "pooled" regressors share one BH-FDR pass (one p-value vector, one
       rejection threshold), e.g. parametric probability/risk/ambiguity
       regressors that are reported together.
     - all other regressors are corrected independently.
5. Save, per regressor: FDR-corrected signed t-map (non-survivors -> NaN),
   positive-only map, and flipped negative-only map.
6. Resample every FDR output to 1mm (nearest-neighbor) against a high-res
   template for figures, re-applying the native-resolution critical |t|
   threshold after resampling so interpolation can't smear a fringe of
   spurious sub-threshold voxels across cluster borders.
7. Save an FDR summary log (N subjects, voxels tested, voxels surviving,
   critical t, per regressor).

Output folder will include:

second_level_final_contrasts/
│
├── uncorrected/                                   ← step 2-3, always saved, all 7 regressors
│   ├── {analysis_name}_t.nii.gz                   ← whole-brain t-map (signed, no thresholding)
│   └── {analysis_name}_p.nii.gz                   ← whole-brain p-map (exact, from nltools regress())
│
├── corrected_FDR0.005/                             ← step 4-5
│   ├── 1_positive_and_negative/
│   │   └── {prefix}_FDR0.005.nii.gz                ← signed t-map, non-survivors set to NaN
│   ├── 2_positive_only/
│   │   └── {prefix}_FDR0.005_posOnly.nii.gz        ← positive surviving voxels only, rest = 0
│   └── 3_negative_only_and_flipped/
│       └── {prefix}_FDR0.005_negOnlyFlipped.nii.gz ← negative surviving voxels, sign-flipped to positive, rest = 0
│
├── corrected_FDR0.005_1mm_nearest_neighbor/        ← step 5-6, same 3-file pattern per regressor
│   ├── 1_positive_and_negative/{prefix}_FDR0.005.nii.gz
│   ├── 2_positive_only/{prefix}_FDR0.005_posOnly.nii.gz
│   └── 3_negative_only_and_flipped/{prefix}_FDR0.005_negOnlyFlipped.nii.gz
│
└── fdr_log.csv                                     ← one row per regressor

"""

import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import nibabel as nib
from scipy.stats import t as t_dist
from nilearn.image import resample_img
from nltools.data import Brain_Data, Design_Matrix
from statsmodels.stats.multitest import fdrcorrection

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.roi_utils import build_beta_df


# ============================================================
# 0) Configuration -- review/edit before running
# ============================================================

# ---- Paths: edit these for your own system/data layout ----
BASE_PATH = "/quobyte/dfoxgrp/BACKED-UP/RDC/data/InitialAnalysisStartN104_20241111"
OUT_BASE_DIR = "/quobyte/dfoxgrp/BACKED-UP/RDC/data/MDART_replication_mesude/derived/fmri/group_level"
# High-res template used for 1mm nearest-neighbor resampling in step 6; must be
# reachable from wherever this script runs.
TEMPLATE_PATH = "/path/to/templates_rois/jsfnMNI152js1DBrainqDq4rFo_Mean104_20240918.nii.gz"

# Set True once to print every regressor filename nltools finds in each
# first_level_subdir used below, so REGRESSORS_OF_INTEREST can be
# checked/fixed against what's actually on disk before trusting any results.
PRINT_AVAILABLE_COLUMNS = True

# prefix used in output filenames -> (first_level_subdir, regressor filename)
#
# The pooled trio (p_risk / r_risk / a_amb) are three regressors from the
# same combined GLM. Each individual contrast instead comes from its own
# dedicated single-contrast GLM folder.
#
# Note: prefix "5" is intentionally absent -- that regressor was dropped from
# the final analysis. Numbering is otherwise unchanged from the original runs.
REGRESSORS_OF_INTEREST: Dict[str, Tuple[str, str]] = {
    "1_cue_p_risk_c0": ("baeta_maps_fullNoPE_fullnus_062625", "cue_p_risk_c0.nii.gz"),
    "2_cue_r_risk_c0": ("baeta_maps_fullNoPE_fullnus_062625", "cue_r_risk_c0.nii.gz"),
    "3_cue_a_amb_c0": ("baeta_maps_fullNoPE_fullnus_062625", "cue_a_amb_c0.nii.gz"),
    "4_cue_CertainThreat_vs_CertainSafe_c0": (
        "baeta_maps_CertainThreatVsCertainSafe_fullnus_062625",
        "cue_CertainThreat_vs_CertainSafe_c0.nii.gz",
    ),
    "6_c50vs0_c0": ("baeta_maps_c50vs0_fullnus_070725", "c50vs0_c0.nii.gz"),
    "7_cue_UncertainThreat_vs_CertainThreat_c0": (
        "baeta_maps_UncertainThreatVsCertainThreat_fullnus_062625",
        "cue_UncertainThreat_vs_CertainThreat_c0.nii.gz",
    ),
    "8_cue_Amb_vs_Risk_c0": ("baeta_maps_AmbVsRisk_fullnus_062625", "cue_Amb_vs_Risk_c0.nii.gz"),
}

# Prefixes that get ONE shared BH-FDR pass (pooled p-values, one rejection
# threshold). Everything else in REGRESSORS_OF_INTEREST is corrected
# independently.
POOLED_PREFIXES: List[str] = [
    "1_cue_p_risk_c0",
    "2_cue_r_risk_c0",
    "3_cue_a_amb_c0",
]

# Regressors whose intercept sign should be flipped before the one-sample
# t-test (e.g. ["cue_r_risk_c0.nii.gz"]).
NEG_COL_LIST: List[str] = []

FDR_ALPHA = 0.005  # decided value; revisit if this changes

OUT_FOLDER = "second_level_final_contrasts"
UNCORR_DIR = os.path.join(OUT_BASE_DIR, OUT_FOLDER, "uncorrected")
CORR_DIR = os.path.join(OUT_BASE_DIR, OUT_FOLDER, f"corrected_FDR{FDR_ALPHA}")
RESAMPLED_DIR = os.path.join(OUT_BASE_DIR, OUT_FOLDER, f"corrected_FDR{FDR_ALPHA}_1mm_nearest_neighbor")

CORR_SUBDIRS = {
    "pos_neg": os.path.join(CORR_DIR, "1_positive_and_negative"),
    "pos": os.path.join(CORR_DIR, "2_positive_only"),
    "neg": os.path.join(CORR_DIR, "3_negative_only_and_flipped"),
}
RESAMPLED_SUBDIRS = {
    "pos_neg": os.path.join(RESAMPLED_DIR, "1_positive_and_negative"),
    "pos": os.path.join(RESAMPLED_DIR, "2_positive_only"),
    "neg": os.path.join(RESAMPLED_DIR, "3_negative_only_and_flipped"),
}


# ============================================================
# 2) Second-level regression: one regressor -> whole-brain t/p maps
# ============================================================
def run_intercept_regression(
    beta_df: pd.DataFrame,
    column: str,
    neg_col_list: Sequence[str],
) -> Optional[dict]:
    """
    Intercept-only (one-sample t-test) voxelwise regression across subjects
    for a single regressor column. Returns None if fewer than 2 subjects
    have that beta map.
    """
    first_level_files = beta_df[column].dropna().values
    n_sub = len(first_level_files)
    if n_sub < 2:
        print(f"[WARN] Not enough subjects for {column}. Skipping.")
        return None

    dm = Design_Matrix(np.ones(n_sub), columns=["constant"])
    first_level = Brain_Data(list(first_level_files))
    first_level.X = dm * -1 if column in neg_col_list else dm

    stats = first_level.regress()
    t_nii = stats["t"].to_nifti()
    p_nii = stats["p"].to_nifti()

    return dict(
        t_img=np.asarray(t_nii.get_fdata(), dtype=float),
        p_img=np.asarray(p_nii.get_fdata(), dtype=float),
        affine=t_nii.affine,
        n_sub=n_sub,
        df=n_sub - 1,
    )


def save_uncorrected(analysis_name: str, t_img: np.ndarray, p_img: np.ndarray, affine: np.ndarray, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    nib.save(nib.Nifti1Image(t_img, affine), os.path.join(out_dir, f"{analysis_name}_t.nii.gz"))
    nib.save(nib.Nifti1Image(p_img, affine), os.path.join(out_dir, f"{analysis_name}_p.nii.gz"))


# ============================================================
# 3) FDR correction (pooled or individual), on EXACT p-values
# ============================================================
def fdr_individual(p_img: np.ndarray, df: int, alpha: float) -> Tuple[np.ndarray, np.ndarray, Optional[float]]:
    """
    BH-FDR on one p-map. Returns (reject_mask, q_img, t_star).
    t_star is the critical |t| corresponding to the largest rejected p
    (None if nothing survives) -- used later to re-threshold after
    resampling, since a NaN-based reject mask can't survive interpolation.
    """
    p_vec_full = p_img.ravel()
    finite = np.isfinite(p_vec_full)
    p_vec = p_vec_full[finite]

    reject_full = np.zeros(p_vec_full.shape, dtype=bool)
    q_full = np.full(p_vec_full.shape, np.nan, dtype=float)

    t_star = None
    if p_vec.size > 0:
        reject_vec, q_vec = fdrcorrection(p_vec, alpha=alpha)
        reject_full[finite] = reject_vec
        q_full[finite] = q_vec
        if np.any(reject_vec):
            p_star = float(p_vec[reject_vec].max())
            t_star = float(t_dist.isf(p_star / 2.0, df))

    return reject_full.reshape(p_img.shape), q_full.reshape(p_img.shape), t_star


def fdr_pooled(
    p_imgs: Sequence[np.ndarray],
    dfs: Sequence[int],
    alpha: float,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[Optional[float]]]:
    """
    One BH-FDR pass across several pooled p-maps (shared rejection
    threshold / shared critical p-value). Because each map can have its
    own df (different subjects missing per regressor), the shared critical
    p-value is converted to a per-map critical |t| using that map's own df
    -- a single shared df would be wrong if n_sub differs across the
    pooled regressors.
    """
    valid_masks, p_vecs = [], []
    for p_img in p_imgs:
        valid = np.isfinite(p_img.ravel())
        valid_masks.append(valid)
        p_vecs.append(p_img.ravel()[valid])

    all_p = np.concatenate(p_vecs)
    reject_all, q_all = fdrcorrection(all_p, alpha=alpha)
    p_star = float(all_p[reject_all].max()) if np.any(reject_all) else None

    reject_masks, q_imgs, t_stars = [], [], []
    start = 0
    for p_img, valid, p_vec, df in zip(p_imgs, valid_masks, p_vecs, dfs):
        stop = start + p_vec.size
        reject_full = np.zeros(p_img.size, dtype=bool)
        q_full = np.full(p_img.size, np.nan, dtype=float)
        reject_full[valid] = reject_all[start:stop]
        q_full[valid] = q_all[start:stop]
        start = stop

        reject_masks.append(reject_full.reshape(p_img.shape))
        q_imgs.append(q_full.reshape(p_img.shape))
        t_stars.append(float(t_dist.isf(p_star / 2.0, df)) if p_star is not None else None)

    return reject_masks, q_imgs, t_stars


# ============================================================
# 4) Save corrected / positive-only / flipped-negative-only maps
# ============================================================
def save_thresholded_maps(
    t_img: np.ndarray,
    affine: np.ndarray,
    reject_mask: np.ndarray,
    prefix: str,
    subdirs: Dict[str, str],
    alpha: float,
) -> Tuple[str, str, str]:
    for d in subdirs.values():
        os.makedirs(d, exist_ok=True)

    corrected = t_img.copy()
    corrected[~reject_mask] = np.nan
    corrected_path = os.path.join(subdirs["pos_neg"], f"{prefix}_FDR{alpha}.nii.gz")
    nib.save(nib.Nifti1Image(corrected, affine), corrected_path)

    thresholded = np.where(reject_mask, t_img, 0.0)
    pos_map = np.where(thresholded > 0, thresholded, 0.0)
    neg_map = np.where(thresholded < 0, -thresholded, 0.0)

    pos_path = os.path.join(subdirs["pos"], f"{prefix}_FDR{alpha}_posOnly.nii.gz")
    neg_path = os.path.join(subdirs["neg"], f"{prefix}_FDR{alpha}_negOnlyFlipped.nii.gz")
    nib.save(nib.Nifti1Image(pos_map, affine), pos_path)
    nib.save(nib.Nifti1Image(neg_map, affine), neg_path)

    return corrected_path, pos_path, neg_path


# ============================================================
# 5) Resample FDR outputs to 1mm for visualization
# ============================================================
def resample_to_1mm(
    prefix: str,
    t_star: Optional[float],
    uncorrected_t_path: str,
    template_img: nib.Nifti1Image,
    out_subdirs: Dict[str, str],
    alpha: float,
) -> None:
    """
    Resample the ORIGINAL UNCORRECTED t-map (continuous, never NaN'd) to the
    template grid with nearest-neighbor interpolation, then re-apply the
    native-resolution critical |t| threshold. Re-thresholding after
    resampling is required because interpolation otherwise blends real
    values across cluster borders regardless of how a mask is handled,
    leaving a fringe of spurious near-zero voxels around every cluster.
    """
    if t_star is None:
        print(f"[WARN] No FDR survivors for {prefix}; skipping resample.")
        return

    for d in out_subdirs.values():
        os.makedirs(d, exist_ok=True)

    raw_img = nib.load(uncorrected_t_path)
    resampled = resample_img(
        raw_img,
        target_affine=template_img.affine,
        target_shape=template_img.shape,
        interpolation="nearest",
    )
    resampled_data = resampled.get_fdata()
    affine = resampled.affine

    result_data = np.where(np.abs(resampled_data) >= t_star, resampled_data, 0.0)

    signed = np.where(result_data != 0, result_data, np.nan)
    nib.save(nib.Nifti1Image(signed, affine), os.path.join(out_subdirs["pos_neg"], f"{prefix}_FDR{alpha}.nii.gz"))

    pos = np.where(result_data > 0, result_data, 0.0)
    nib.save(nib.Nifti1Image(pos, affine), os.path.join(out_subdirs["pos"], f"{prefix}_FDR{alpha}_posOnly.nii.gz"))

    neg = np.where(result_data < 0, -result_data, 0.0)
    nib.save(nib.Nifti1Image(neg, affine), os.path.join(out_subdirs["neg"], f"{prefix}_FDR{alpha}_negOnlyFlipped.nii.gz"))

    print(f"[INFO] Resampled {prefix} (t* = {t_star:.4f})")


# ============================================================
# 6) Main pipeline
# ============================================================
def main() -> None:
    subject_ids = sorted(
        d for d in os.listdir(BASE_PATH)
        if os.path.isdir(os.path.join(BASE_PATH, d)) and d.startswith("RDC")
    )
    print(f"[INFO] Found {len(subject_ids)} subjects under {BASE_PATH}")

    # Build one beta_df per distinct first_level_subdir referenced in
    # REGRESSORS_OF_INTEREST (the pooled trio shares one; each individual
    # contrast has its own dedicated folder).
    subdirs_needed = list(dict.fromkeys(subdir for subdir, _ in REGRESSORS_OF_INTEREST.values()))
    beta_dfs: Dict[str, pd.DataFrame] = {}
    for subdir in subdirs_needed:
        beta_dfs[subdir] = build_beta_df(
            base_path=BASE_PATH,
            subject_ids=subject_ids,
            first_level_subdir=subdir,
            regressor_names=None,
        )
        print(f"[INFO] beta_df shape: {beta_dfs[subdir].shape} (subjects x regressors) from {subdir}")

    if PRINT_AVAILABLE_COLUMNS:
        for subdir, df in beta_dfs.items():
            print(f"[INFO] Available regressor columns in {subdir}:")
            for c in df.columns:
                print(f"  {c}")

    missing = [
        (prefix, subdir, fname)
        for prefix, (subdir, fname) in REGRESSORS_OF_INTEREST.items()
        if fname not in beta_dfs[subdir].columns
    ]
    if missing:
        raise ValueError(
            f"These REGRESSORS_OF_INTEREST (prefix, subdir, filename) entries were not found as "
            f"beta_df columns: {missing}\n"
            f"Check the mapping against the printed column lists above."
        )

    # ---- Step 2-3: second-level regression + save uncorrected maps ----
    regression_results: Dict[str, dict] = {}
    for prefix, (subdir, fname) in REGRESSORS_OF_INTEREST.items():
        analysis_name = fname.split(".")[0]
        beta_df = beta_dfs[subdir]
        print(f"\n[INFO] {prefix} ({analysis_name}) from {subdir}: N={beta_df[fname].dropna().shape[0]}")

        result = run_intercept_regression(beta_df, fname, NEG_COL_LIST)
        if result is None:
            continue

        save_uncorrected(analysis_name, result["t_img"], result["p_img"], result["affine"], UNCORR_DIR)
        result["analysis_name"] = analysis_name
        result["first_level_subdir"] = subdir
        regression_results[prefix] = result

    # ---- Step 4: FDR correction (pooled group + everything else individually) ----
    fdr_log = []

    pooled_prefixes = [p for p in POOLED_PREFIXES if p in regression_results]
    if pooled_prefixes:
        p_imgs = [regression_results[p]["p_img"] for p in pooled_prefixes]
        dfs = [regression_results[p]["df"] for p in pooled_prefixes]
        reject_masks, q_imgs, t_stars = fdr_pooled(p_imgs, dfs, FDR_ALPHA)

        for prefix, reject_mask, q_img, t_star in zip(pooled_prefixes, reject_masks, q_imgs, t_stars):
            r = regression_results[prefix]
            save_thresholded_maps(r["t_img"], r["affine"], reject_mask, prefix, CORR_SUBDIRS, FDR_ALPHA)
            r["reject_mask"], r["t_star"] = reject_mask, t_star
            fdr_log.append(dict(
                prefix=prefix, analysis_name=r["analysis_name"], first_level_subdir=r["first_level_subdir"],
                n_subjects=r["n_sub"], fdr_group="pooled", voxels_tested=int(np.isfinite(r["p_img"]).sum()),
                voxels_surviving_fdr=int(reject_mask.sum()), t_star=t_star, fdr_alpha=FDR_ALPHA,
            ))

    individual_prefixes = [p for p in regression_results if p not in pooled_prefixes]
    for prefix in individual_prefixes:
        r = regression_results[prefix]
        reject_mask, q_img, t_star = fdr_individual(r["p_img"], r["df"], FDR_ALPHA)
        save_thresholded_maps(r["t_img"], r["affine"], reject_mask, prefix, CORR_SUBDIRS, FDR_ALPHA)
        r["reject_mask"], r["t_star"] = reject_mask, t_star
        fdr_log.append(dict(
            prefix=prefix, analysis_name=r["analysis_name"], first_level_subdir=r["first_level_subdir"],
            n_subjects=r["n_sub"], fdr_group="individual", voxels_tested=int(np.isfinite(r["p_img"]).sum()),
            voxels_surviving_fdr=int(reject_mask.sum()), t_star=t_star, fdr_alpha=FDR_ALPHA,
        ))

    pd.DataFrame(fdr_log).to_csv(os.path.join(OUT_BASE_DIR, OUT_FOLDER, "fdr_log.csv"), index=False)
    print(f"\n[INFO] Saved FDR log: {os.path.join(OUT_BASE_DIR, OUT_FOLDER, 'fdr_log.csv')}")

    # ---- Step 5-6: resample everything to 1mm for visualization ----
    template_img = nib.load(TEMPLATE_PATH)
    for prefix, r in regression_results.items():
        uncorrected_t_path = os.path.join(UNCORR_DIR, f"{r['analysis_name']}_t.nii.gz")
        resample_to_1mm(prefix, r["t_star"], uncorrected_t_path, template_img, RESAMPLED_SUBDIRS, FDR_ALPHA)

    print("\n[INFO] Pipeline complete.")


if __name__ == "__main__":
    main()
