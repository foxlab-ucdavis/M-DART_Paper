"""
Shared helpers for beta-map bookkeeping and ROI-mean extraction.

Used by src/second_level/run_second_level_full_pipeline.py and the scripts in
src/roi_extraction/.
"""

import os
import glob
from typing import List, Optional

import numpy as np
import pandas as pd
import nibabel as nib


def build_beta_df(
    base_path: str,
    subject_ids: List[str],
    first_level_subdir: str,
    regressor_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    beta_df: rows = subjects, columns = regressor filenames, values = full
    path to each subject's beta map (NaN if missing).
    """
    if regressor_names is None:
        example_subj = None
        for s in subject_ids:
            d = os.path.join(base_path, s, first_level_subdir)
            if os.path.isdir(d):
                example_subj = s
                break
        if example_subj is None:
            raise RuntimeError(
                f"Could not find {first_level_subdir} in any subject folder under {base_path}"
            )

        regressor_paths = glob.glob(os.path.join(base_path, example_subj, first_level_subdir, "*.nii*"))
        regressor_names = sorted(os.path.basename(p) for p in regressor_paths)

    beta_df = pd.DataFrame(index=subject_ids, columns=regressor_names, dtype=object)
    for subj in subject_ids:
        subj_dir = os.path.join(base_path, subj, first_level_subdir)
        for reg in regressor_names:
            fp = os.path.join(subj_dir, reg)
            beta_df.loc[subj, reg] = fp if os.path.exists(fp) else np.nan

    return beta_df


def load_mask_bool(mask_path: str) -> np.ndarray:
    return nib.load(mask_path).get_fdata() > 0


def nanmean_in_mask(img: np.ndarray, mask: np.ndarray) -> float:
    vals = img[mask]
    return float(np.nanmean(vals)) if vals.size > 0 else np.nan


def check_same_shape(arr: np.ndarray, ref: np.ndarray, name: str) -> None:
    if arr.shape != ref.shape:
        raise ValueError(f"[ERROR] Shape mismatch for {name}: {arr.shape} vs {ref.shape}")
