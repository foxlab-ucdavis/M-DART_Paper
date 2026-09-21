"""
Per-ROI (Ce, BST) timecourse extraction and trial-wise regression.

For each subject: extracts mean BOLD timeseries within each ROI in
MDART_timecourse_ROIs, upsamples and epochs it around cue and outcome onsets,
then fits a combined single-model regression (risky + ambiguous trials
together, via gated predictors) at every timepoint.

Inputs (per subject, under SUBJ_ROOT):
    func/*rec-S4WMgRado*.nii.gz    preprocessed BOLD, one file per run
    beh/*.txt                     behavioral event files, one file per run
ROI masks: every NIfTI file under ROI_DIR (see build_roi_map_from_dir for
naming conventions).

Outputs (per subject, under SAVE_PATH/<subject>/Timecourse_MDART_CE_BST_mean_roi):
    <subject>_epochs_wide.csv                    all epochs (cue + outcome, all ROIs)
    <subject>_epochs_wide_cue.csv                cue-aligned epochs only
    <subject>_epochs_wide_outcome.csv            outcome-aligned epochs only
    <subject>_betas_timecourse.csv               per-timepoint betas, averaged across runs
    <subject>_betas_timecourse_BYRUN.csv         per-timepoint betas, one row per run

Usage:
    python timecourse_analysis_BST_CE_mean_roi.py [--start_id 102] [--only_subject RDC167]
"""

import os
import glob
import re
import numpy as np
import pandas as pd
import nibabel as nib
import statsmodels.api as sm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--start_id", type=int, default=102)
parser.add_argument("--only_subject", type=str, default=None,
                    help="If provided (e.g., RDC167), run only that subject.")
args = parser.parse_args()

START_ID = args.start_id

#=================HELPER FUNCTIONS=================#

def rdc_num_from_dir(subj_dir: str) -> int:
    m = re.search(r"RDC(\d+)$", os.path.basename(subj_dir))
    return int(m.group(1)) if m else -1


def build_roi_map_from_dir(roi_dir):
    """
    Scan `roi_dir` for ROI files and build a dict:
        { "Peak1_(...)" : "/full/path/to/ROI_01_Peak1_(...).nii.gz",
          "ASTR_L_p0.10": "/full/path/to/ASTR_L_p10.nii.gz",
          ... }

    Supported filename patterns:
      1) ROI_<number>_<name>.nii[.gz]
         e.g. ROI_01_Peak1_(-30.0,-10.0,-16.0)_posOut_negPthreat.nii.gz
         -> key: Peak1_(-30.0,-10.0,-16.0)_posOut_negPthreat

      2) Generic names without the ROI_ prefix:
         e.g. ASTR_L_p10.nii.gz
         -> key: ASTR_L_p0.10   (interpreted as p = 10/100 = 0.10)
    """
    roi_map = {}

    for fname in sorted(os.listdir(roi_dir)):
        # only consider NIfTI files
        if not (fname.lower().endswith(".nii") or fname.lower().endswith(".nii.gz")):
            continue

        full_path = os.path.join(roi_dir, fname)

        # Strip extension to get the base name
        if fname.lower().endswith(".nii.gz"):
            base = fname[:-7]  # remove ".nii.gz"
        elif fname.lower().endswith(".nii"):
            base = fname[:-4]  # remove ".nii"
        else:
            continue

        # Case 1: ROI_<num>_<name> pattern (original behavior)
        if base.startswith("ROI_"):
            m = re.match(r"^ROI_\d+_(.+)$", base)
            if not m:
                raise ValueError(
                    f"Filename does not match expected pattern "
                    f"'ROI_<num>_<name>.nii[.gz]': {fname}"
                )
            roi_name = m.group(1)

        else:
            # Case 2: generic name
            # If it looks like <prefix>_p<digits>, convert digits to a probability:
            #   ASTR_L_p10 -> ASTR_L_p0.10  (10 -> 0.10)
            m_prob = re.match(r"^(.*)_p(\d+)$", base)
            if m_prob:
                prefix, digits = m_prob.groups()
                prob = int(digits) / 100.0  # 10 -> 0.10, 5 -> 0.05, 25 -> 0.25, etc.
                roi_name = f"{prefix}_p{prob:0.2f}"
            else:
                # Otherwise just use the base name
                roi_name = base

        roi_map[roi_name] = full_path

    if not roi_map:
        raise ValueError(f"No ROI NIfTI files found in directory: {roi_dir}")

    return roi_map

def load_roi_bank_no_checks(roi_map, materialize_masks=True, require_binary=True):
    bank = {}
    for name, path in roi_map.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"ROI '{name}' not found: {path}")
        img = nib.load(path)
        mask = None

        if materialize_masks:
            data = img.get_fdata()
            if require_binary:
                mask = (data != 0)
            else:
                mask = data

        bank[name] = {"path": path, "img": img, "mask": mask}
    return bank

def load_roi_bank_from_dir(roi_dir, materialize_masks=True, require_binary=True):
    roi_map = build_roi_map_from_dir(roi_dir)
    return load_roi_bank_no_checks(roi_map,
                                   materialize_masks=materialize_masks,
                                   require_binary=require_binary)

RUN_RE = re.compile(r"task-MDART[_-]run-(\d+)", re.IGNORECASE)

def map_files_by_run(pattern):
    """
    Given a glob pattern, return {run_id: filepath} and warn if a file
    doesn't contain a recognizable run number.
    """
    files = glob.glob(pattern)
    out = {}
    for f in sorted(files):
        m = RUN_RE.search(os.path.basename(f))
        if m is None:
            print(f"[WARN] Could not extract run from filename: {f}")
            continue
        run_id = int(m.group(1))
        out[run_id] = f
    return out

def extract_roi_timeseries(
    fmri_path,
    roi_mask_bool,
    TR,
    mode="mean",
    detrend=False,
    zscore=False,
):
    """
    Extract mean or voxelwise timeseries from a 4D fMRI image within a boolean ROI mask.

    Parameters
    ----------
    fmri_path : str or path-like
        Path to 4D NIfTI file.
    roi_mask_bool : np.ndarray (X,Y,Z) of bool
        ROI mask in the same space as the fMRI data.
    TR : float
        Repetition time in seconds.
    mode : {"mean", "voxelwise"}
        - "mean": return (t_run, y_mean) with shape (T,)
        - "voxelwise": return (t_run, Y) with shape (T, V)
    detrend : bool
        If True, remove linear trend (intercept + time) from each voxel.
    zscore : bool
        If True, z-score each voxel across time.

    Returns
    -------
    t_run : np.ndarray, shape (T,)
        Time points in seconds at TR resolution.
    y_out : np.ndarray
        Mean or voxelwise timeseries depending on `mode`.
    """
    img = nib.load(fmri_path)
    data = img.get_fdata()  # (X, Y, Z, T)
    if data.ndim != 4:
        raise ValueError(f"Expected 4D fMRI data, got shape {data.shape}")

    # Ensure mask is boolean and same spatial shape
    roi_mask_bool = np.asarray(roi_mask_bool, dtype=bool)
    if roi_mask_bool.shape != data.shape[:3]:
        raise ValueError(
            f"ROI mask shape {roi_mask_bool.shape} does not match "
            f"fMRI spatial shape {data.shape[:3]}"
        )

    # Extract voxel x time -> (T, V)
    roi_data = data[roi_mask_bool]  # (V, T)
    n_vox = roi_data.shape[0]
    if n_vox == 0:
        raise ValueError("ROI mask contains zero voxels.")

    Y = roi_data.astype(float).T  # (T, V)
    Tn = Y.shape[0]
    t_run = np.arange(Tn, dtype=float) * float(TR)

    # Optional detrend (linear: intercept + time)
    if detrend:
        t = t_run[:, None]
        X = np.c_[np.ones_like(t), t]  # (T, 2)
        beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
        Y = Y - X @ beta  # residuals

    # Optional z-score each voxel across time
    if zscore:
        m = np.nanmean(Y, axis=0, keepdims=True)
        s = np.nanstd(Y, axis=0, ddof=1, keepdims=True)
        s[s == 0] = 1.0
        Y = (Y - m) / s

    if mode == "mean":
        y_mean = np.nanmean(Y, axis=1)  # (T,)
        return t_run, y_mean
    elif mode == "voxelwise":
        return t_run, Y
    else:
        raise ValueError("mode must be 'mean' or 'voxelwise'")

def upsample_timeseries(t_run, y_run, upsample_hz, TR):
    """
    Return high-res grid and values via linear interpolation.

    Parameters
    ----------
    t_run : array-like, shape (T,)
        Native time points (seconds).
    y_run : array-like, shape (T,) or (T, V)
        Signal at each TR.
    upsample_hz : float
        Target sampling frequency in Hz (e.g., 10.0 → dt = 0.1 s).
    TR : float
        Native repetition time in seconds, used for sanity checking.

    Returns
    -------
    t_hi : np.ndarray, shape (T_hi,)
        High-resolution time grid (seconds).
    y_hi : np.ndarray
        Interpolated signal at t_hi. Shape (T_hi,) or (T_hi, V).
    dt : float
        High-resolution time step in seconds (1 / upsample_hz).
    """
    t_run = np.asarray(t_run, dtype=float)
    y_run = np.asarray(y_run)
    dt = 1.0 / float(upsample_hz)

    if t_run.ndim != 1:
        raise ValueError("t_run must be 1D array of timepoints in seconds.")

    # Optional sanity check on TR spacing
    if t_run.size > 1:
        approx_TR = np.median(np.diff(t_run))
        if not np.isclose(approx_TR, TR, rtol=1e-3, atol=1e-3):
            print(
                f"[WARN] Median TR from t_run ({approx_TR:.4f}s) "
                f"differs from provided TR ({TR:.4f}s)."
            )

    # Define high-res grid
    t_start = t_run[0]
    t_end   = t_run[-1] + TR  # cover until just past last TR center
    t_hi = np.arange(t_start, t_end, dt)

    # Interpolate
    if y_run.ndim == 1:
        y_hi = np.interp(t_hi, t_run, y_run, left=np.nan, right=np.nan)
    elif y_run.ndim == 2:
        n_vox = y_run.shape[1]
        y_hi = np.full((t_hi.size, n_vox), np.nan, dtype=float)
        for j in range(n_vox):
            y_hi[:, j] = np.interp(t_hi, t_run, y_run[:, j], left=np.nan, right=np.nan)
    else:
        raise ValueError("y_run must be 1D or 2D (T,) or (T, V).")

    return t_hi, y_hi, dt

def epoch_trials_from_upsampled(
    t_hi,
    y_hi,
    events_df,
    upsample_hz,
    tmin=-1.0,
    tmax=18.0,
    onset_col="TrialStartTime",
    trial_type_col="TrialTypeCode",
    outcome_col="ReinforcerInTime",
    rating_col="RatingStartTime",
    extra_cols=None,
):
    """
    Epoch a high-resolution ROI timeseries into trial-locked segments.

    Parameters
    ----------
    t_hi : np.ndarray, shape (T_hi,)
        High-resolution time grid in seconds (from upsample_timeseries).
    y_hi : np.ndarray, shape (T_hi,)
        High-resolution ROI signal (mean signal; 1D for now).
    events_df : pd.DataFrame
        Event/behavior table with trial onsets and optional outcome/rating times.
    upsample_hz : float
        Sampling rate of t_hi (e.g. 10.0 Hz → dt = 0.1 s).
    tmin, tmax : float
        Window (in seconds) relative to trial onset to extract, e.g. -1..18.
    onset_col : str
        Column name in events_df for trial onset time (seconds).
    trial_type_col : str
        Column name for trial type (e.g., TrialTypeCode).
    outcome_col, rating_col : str
        Column names for outcome and rating times (seconds), if present.
    extra_cols : list of str or None
        Extra trial-level columns to copy from events_df into each row.

    Returns
    -------
    df_epochs : pd.DataFrame
        Long/tidy DataFrame. Each row is one (trial, timepoint).
        Columns include:
            trial_index, trial_type, onset_s, outcome_s, rating_s,
            time_rel_s, signal, (extra_cols...), plus whatever you add later.
    """
    if extra_cols is None:
        extra_cols = []

    # Ensure arrays
    t_hi = np.asarray(t_hi, dtype=float)
    y_hi = np.asarray(y_hi, dtype=float)

    if t_hi.ndim != 1:
        raise ValueError("t_hi must be 1D array of timepoints in seconds.")
    if y_hi.ndim != 1:
        raise ValueError("y_hi must be 1D (mean ROI signal) for now.")

    dt = 1.0 / float(upsample_hz)
    rel_grid = np.arange(tmin, tmax, dt)  # e.g., -1.0 .. 18.0 (step dt)

    # High-res time origin (t=0 corresponds to first timepoint in t_hi)
    t0 = t_hi[0]

    # quick indexer from absolute times to nearest high-res index
    def time_to_idx(t_abs):
        """
        Map absolute times (seconds) onto indices in t_hi,
        assuming t_hi is regular with spacing dt and starts at t0.
        """
        return np.round((t_abs - t0) / dt).astype(int)

    out_rows = []

    for trial_idx, row in events_df.iterrows():
        # --- trial-level timings ---
        onset = float(row[onset_col])

        outcome_val = row.get(outcome_col, np.nan)
        rating_val  = row.get(rating_col, np.nan)

        outcome_s = float(outcome_val) if pd.notna(outcome_val) else np.nan
        rating_s  = float(rating_val)  if pd.notna(rating_val)  else np.nan

        # absolute times for this trial's window
        t_abs = onset + rel_grid  # vector
        idx = time_to_idx(t_abs)

        # mask out-of-bounds indices
        valid = (idx >= 0) & (idx < len(t_hi))

        # initialize with NaNs, fill valid positions
        y_seg = np.full_like(rel_grid, np.nan, dtype=float)
        y_seg[valid] = y_hi[idx[valid]]

        trial_type = row.get(trial_type_col, None)

        # extra trial-level info
        extra_data = {}
        for col in extra_cols:
            extra_data[col] = row.get(col, np.nan)

        base_dict = {
            "trial_index": trial_idx,
            "trial_type": trial_type,
            "onset_s": onset,
            "outcome_s": outcome_s,
            "rating_s": rating_s,
            "time_rel_s": rel_grid,
            "signal": y_seg,
        }
        base_dict.update(extra_data)

        df_trial = pd.DataFrame(base_dict)
        out_rows.append(df_trial)

    if out_rows:
        return pd.concat(out_rows, ignore_index=True)
    else:
        base_cols = [
            "trial_index",
            "trial_type",
            "onset_s",
            "outcome_s",
            "rating_s",
            "time_rel_s",
            "signal",
        ]
        return pd.DataFrame(columns=base_cols + list(extra_cols))

def load_events_table(path):
    """
    Read MDART response log.

    - Skips lines starting with '#'
    - Parses as tab-separated
    - Strips column name whitespace
    - Converts numeric-looking columns
    - Converts *Time columns from ms -> seconds

    Returns
    -------
    df : pd.DataFrame
        Events table with time columns in seconds.
    """
    # Read everything as string first, ignore comment lines
    df = pd.read_csv(
        path,
        sep=r"\t+",
        engine="python",
        comment="#",
        dtype=str
    )

    # Clean column names
    df.columns = df.columns.str.strip()

    # Convert "numeric-looking" columns to numeric where possible
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="ignore")

    # Time columns are in ms; convert to seconds if present
    time_cols_ms = [
        "TrialStartTime",
        "ReinforcerInTime",
        "ReinforcerOutTime",
        "WhiteNoiseStartTime",
        "RatingStartTime",
        "TrialEndTime",
    ]
    for c in time_cols_ms:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce") / 1000.0

    # Optional: canonical TrialType column
    if "TrialTypeCode" in df.columns and "TrialType" not in df.columns:
        df["TrialType"] = df["TrialTypeCode"]

    # Drop rows with no TrialStartTime at all (completely unusable trials)
    if "TrialStartTime" in df.columns:
        df = df.dropna(subset=["TrialStartTime"]).reset_index(drop=True)

    return df

def add_derived_task_columns(df):
    df = df.copy()

    df["p_threat"] = compute_demean_range01(df["ThreatPct"] / 100.0)
    df["r_risk"] = compute_demean_range01(0.5**2 - ((df["ThreatPct"] / 100.0) - 0.5) ** 2)
    df["rational_p_threat"] = compute_demean_range01((df["ThreatPct"] / 100.0) + (df["AmbiguousPct"] / 100.0) / 2.0)
    df["a_amb"] = compute_demean_range01(df["AmbiguousPct"] / 100.0)

    df["outcome"] = 2 * (df["IsaShockOut"] - 0.5)
    df["outcome_demeaned"] = df["IsaShockOut"] - df["IsaShockOut"].mean()

    return df

def compute_demean_range01(vals):
    # scale 0-1
    model_vals = (vals - np.min(vals)) / (np.max(vals) - np.min(vals))
    # subtract mean
    model_vals = model_vals - np.mean(model_vals)
    # zero center (again) and scale by 2
    model_vals = 2 * (model_vals - np.mean(model_vals))

    # fill NaNs with 0 (e.g., for missing trial types)
    model_vals.fillna(0, inplace=True)
    return model_vals

def compute_betas(df):
    """
    Compute timecourse betas for a given epochs_wide_risk dataframe.
    Model:
        BOLD(t) ~ outcome_demeaned + trial_type_coded
                  + p_threat_risky + r_risk_risky
                  + rational_p_threat_amb + a_amb_amb

    Original trial_type column:
        3   -> ambiguous
        not 3 -> risky

    Regression coding:
        risky     = +1
        ambiguous = -1
    """
    df = df.copy()

    # -----------------------------
    # Required columns
    # -----------------------------
    required = [
        "trial_type",
        "outcome_demeaned",
        "p_threat",
        "r_risk",
        "rational_p_threat",
        "a_amb",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # -----------------------------
    # Trial-type coding
    # -----------------------------
    orig_trial_type = df["trial_type"].copy()

    amb_mask = orig_trial_type == 3
    risky_mask = orig_trial_type != 3

    df["trial_type_coded"] = np.where(risky_mask, 1, -1)
    df["trial_kind_label"] = np.where(amb_mask, "ambiguous", "risky")

    # -----------------------------
    # Gated predictors
    # -----------------------------
    df["p_threat_risky"] = np.where(risky_mask, df["p_threat"], 0.0)
    df["r_risk_risky"] = np.where(risky_mask, df["r_risk"], 0.0)
    df["rational_p_threat_amb"] = np.where(amb_mask, df["rational_p_threat"], 0.0)
    df["a_amb_amb"] = np.where(amb_mask, df["a_amb"], 0.0)

    predictor_cols = [
        "outcome_demeaned",
        "trial_type_coded",
        "p_threat_risky",
        "r_risk_risky",
        "rational_p_threat_amb",
        "a_amb_amb",
    ]

    # -----------------------------
    # Time columns
    # -----------------------------
    time_cols = [c for c in df.columns if c.startswith("t_")]
    if not time_cols:
        raise ValueError("No time columns starting with 't_' were found.")

    def col_to_time(c):
        # Convert "t_-1.0s" → -1.0
        return float(c[2:-1])

    # Sort by actual time value
    time_cols = sorted(time_cols, key=col_to_time)
    times = [col_to_time(c) for c in time_cols]

    # -----------------------------
    # Fit OLS at each timepoint
    # -----------------------------
    betas = {col: [] for col in predictor_cols}
    n_trials_used = []  

    for t_col in time_cols:
        sub = df[predictor_cols + [t_col]].dropna().copy()
        n_trials_used.append(len(sub))

        if len(sub) == 0:
            for col in predictor_cols:
                betas[col].append(np.nan)
            continue

        y = sub[t_col]
        X = sm.add_constant(sub[predictor_cols], has_constant="add")
        model = sm.OLS(y, X).fit()

        for col in predictor_cols:
            betas[col].append(model.params.get(col, np.nan))

    out = pd.DataFrame({
        "time_s": times,
        "n_trials_used": n_trials_used,
        "beta_outcome_demeaned": betas["outcome_demeaned"],
        "beta_trial_type": betas["trial_type_coded"],
        "beta_p_threat_risky": betas["p_threat_risky"],
        "beta_r_risk_risky": betas["r_risk_risky"],
        "beta_rational_p_threat_amb": betas["rational_p_threat_amb"],
        "beta_a_amb_amb": betas["a_amb_amb"],
    })

    # 3) Return a clean dataframe
    return out

#=================MAIN LOOP=================#

def run_timecourse_for_subject(
    subj_dir,
    base_dir,
    save_path,
    roi_bank,
    tr=1.25,
    upsample_hz=10.0,
    tmin=-1.0,
    tmax=18.1,
    run_ids=(1, 2, 3, 4),
):
    """
    Run timecourse extraction + trial-wise regression for a single subject
    across all specified runs and ROIs.

    Parameters
    ----------
    subj_dir : str
        Either a full path to the subject directory (e.g., ".../RDC104")
        or a relative name (e.g., "RDC104") that will be resolved under base_dir/Subjects.
    base_dir : str
        Base directory for the project (used if subj_dir is not absolute).
    roi_bank : dict
        Dictionary of ROI definitions as produced by load_roi_bank_no_checks().
    tr, upsample_hz, tmin, tmax, run_ids : as before.

    Saves
    -----
      - {subject_id}_epochs_wide.csv
      - {subject_id}_betas_timecourse.csv

    Returns
    -------
    epochs_wide, betas_long : pd.DataFrame
    """

    # ----- 0. Resolve paths & subject ID -----    
    subject_id = os.path.basename(subj_dir)
    print(f"[INFO] Subject {subject_id}")

    save_dir = os.path.join(save_path, subject_id, "Timecourse_MDART_CE_BST_mean_roi")  # <-- adjust if your folder name differs
    os.makedirs(save_dir, exist_ok=True)

    # Map fMRI + behavior files by run
    func_dir = os.path.join(subj_dir, "func")
    beh_dir  = os.path.join(subj_dir, "beh")

    fmri_by_run = map_files_by_run(os.path.join(func_dir, "*rec-S4WMgRado*.nii.gz"))
    beh_by_run  = map_files_by_run(os.path.join(beh_dir, "*.txt"))
    
    # Restrict to user-specified run_ids
    available_runs = sorted(set(fmri_by_run.keys()) & set(beh_by_run.keys()))
    selected_runs = [r for r in run_ids if r in available_runs]

    if not selected_runs:
        raise RuntimeError(f"No valid runs for subject {subject_id}. Found runs: {available_runs}")

    print(f"[INFO]   Using runs: {selected_runs}")

    # ----- 1. Extract epochs for all runs × ROIs -----
    all_epoch_dfs = []

    for run_id in selected_runs:
        fmri_path = fmri_by_run[run_id]
        beh_path  = beh_by_run[run_id]

        events_df = load_events_table(beh_path)
        events_df = add_derived_task_columns(events_df)
        print(f"[INFO]   Processing run {run_id}: {len(events_df)} trials")

        for roi_name, roi_info in roi_bank.items():
            roi_mask = roi_info["mask"]
            print(f"[INFO]     ROI: {roi_name}")

            # 1) mean ROI signal at TR
            t_run, y_run = extract_roi_timeseries(
                fmri_path,
                roi_mask,
                TR=tr,
                mode="mean",
                detrend=True, # Change true if you want to detrend
                zscore=True, # Change true if you want to z-score
            )

            # 2) upsample to high-resolution grid
            t_hi, y_hi, dt = upsample_timeseries(
                t_run,
                y_run,
                upsample_hz=upsample_hz,
                TR=tr,
            )

            # 3.1) epoch trials (tmin..tmax)
            df_epochs_cue = epoch_trials_from_upsampled(
                t_hi,
                y_hi,
                events_df,
                upsample_hz=upsample_hz,
                tmin=tmin,
                tmax=tmax,
                onset_col="TrialStartTime",
                trial_type_col="TrialTypeCode",
                outcome_col="ReinforcerInTime",
                rating_col="RatingStartTime",
                extra_cols=["ThreatPct", "AmbiguousPct", "IsaShockOut",
                            "p_threat", "r_risk", "rational_p_threat", "a_amb",
                            "outcome", "outcome_demeaned",],)

            # Annotate
            df_epochs_cue["alignment"] = "cue"
            df_epochs_cue["subject"] = subject_id
            df_epochs_cue["run"]     = run_id
            df_epochs_cue["roi"]     = roi_name
            df_epochs_cue["dt"]      = dt
            df_epochs_cue["TR"]      = tr

            all_epoch_dfs.append(df_epochs_cue)

            # 3.2) epoch trials (tmin..tmax)
            df_epochs_outcome = epoch_trials_from_upsampled(
                t_hi,
                y_hi,
                events_df,
                upsample_hz=upsample_hz,
                tmin=tmin,
                tmax=tmax,
                onset_col="ReinforcerInTime",
                trial_type_col="TrialTypeCode",
                outcome_col="ReinforcerInTime",
                rating_col="RatingStartTime",
                extra_cols=["ThreatPct", "AmbiguousPct", "IsaShockOut",
                            "p_threat", "r_risk", "rational_p_threat", "a_amb",
                            "outcome", "outcome_demeaned",],)

            # Annotate
            df_epochs_outcome["alignment"] = "outcome"
            df_epochs_outcome["subject"] = subject_id
            df_epochs_outcome["run"]     = run_id
            df_epochs_outcome["roi"]     = roi_name
            df_epochs_outcome["dt"]      = dt
            df_epochs_outcome["TR"]      = tr

            all_epoch_dfs.append(df_epochs_outcome)

    if not all_epoch_dfs:
        raise RuntimeError(f"No epochs produced for subject {subject_id}.")

    epochs_long = pd.concat(all_epoch_dfs, ignore_index=True)
    print("[OK] epochs_long:", epochs_long.shape, "rows")

    # ----- 3. Long → wide (risk-only, implicit via ThreatPctRisk NaNs) -----
    id_cols = [
        "subject", "run", "roi",
        "trial_index", "trial_type",
        "ThreatPct", "AmbiguousPct", "IsaShockOut",
        "p_threat", "rational_p_threat", "r_risk", "a_amb",
        "outcome", "outcome_demeaned", 
        "onset_s", "outcome_s", "rating_s",
        "dt", "TR",
    ]

    epochs_wide = (
        epochs_long
        .pivot_table(
            index=id_cols + ["alignment"],   # each row = one (subject, run, roi, trial, alignment)
            columns="time_rel_s",
            values="signal"
        )
        .reset_index()
    )

    # Clean up time column names: t_-1.0s, t_0.0s, ...
    new_cols = []
    for c in epochs_wide.columns:
        if isinstance(c, (float, int)):
            new_cols.append(f"t_{c:0.1f}s")
        else:
            new_cols.append(c)
    epochs_wide.columns = new_cols

    print("[OK] epochs_wide:", epochs_wide.shape, "rows x cols")
    
    
    epochs_wide_cue     = epochs_wide.query("alignment == 'cue'")
    epochs_wide_outcome = epochs_wide.query("alignment == 'outcome'")

    print("[OK] epochs_wide_cue:", epochs_wide_cue.shape, "rows x cols")
    print("[OK] epochs_wide_outcome:", epochs_wide_outcome.shape, "rows x cols")


    # ----- 4. Timepoint-wise regression using compute_betas() -----

    all_betas = []
    all_betas_by_run = []

    for alignment_name, df_align in [
        ("cue", epochs_wide_cue),
        ("outcome", epochs_wide_outcome),
    ]:
        print(f"[INFO] Running regression for alignment: {alignment_name}")

        # Group by ROI
        for roi_name, df_roi in df_align.groupby("roi"):
            print(f"[INFO]   ROI: {roi_name} ({len(df_roi)} trials)")

            run_betas = []

            for run_id, df_run in df_roi.groupby("run"):
                n_trials = len(df_run)
                if n_trials < 3:
                    print(f"[WARN]     run {run_id}: too few trials ({n_trials}), skipping")
                    continue

                b = compute_betas(df_run)
                b["run"] = run_id
                b["n_trials"] = n_trials

                run_betas.append(b)

                # optional: store per-run betas
                b_run_out = b.copy()
                b_run_out["subject"] = subject_id
                b_run_out["roi"] = roi_name
                b_run_out["alignment"] = alignment_name
                all_betas_by_run.append(b_run_out)

            if not run_betas:
                print(f"[WARN]   ROI {roi_name}: no runs with enough trials")
                continue

            betas_run_long = pd.concat(run_betas, ignore_index=True)
            beta_cols = [c for c in betas_run_long.columns if c.startswith("beta_")]

            # --- average across runs (unweighted) ---
            betas_mean = (
                betas_run_long
                .groupby("time_s", as_index=False)[beta_cols]
                .mean()
            )

            betas_mean["subject"] = subject_id
            betas_mean["roi"] = roi_name
            betas_mean["alignment"] = alignment_name
            betas_mean["n_trials_used"] = (
                betas_run_long[["run", "n_trials"]]
                .drop_duplicates()["n_trials"]
                .sum()
            )
            betas_mean["n_runs_used"] = betas_run_long["run"].nunique()

            all_betas.append(betas_mean)

    # Final subject-level betas (averaged across runs)
    betas_long = pd.concat(all_betas, ignore_index=True)

    if all_betas_by_run:
        betas_by_run_long = pd.concat(all_betas_by_run, ignore_index=True)
        betas_by_run_path = os.path.join(save_dir, f"{subject_id}_betas_timecourse_BYRUN.csv")
        betas_by_run_long.to_csv(betas_by_run_path, index=False)
        print(f"[SAVE] betas_by_run -> {betas_by_run_path}")

    # ----- 5. Save outputs -----
    # --- Save risk-only cue and outcome epoch tables ---
    epochs_wide_cue_out = os.path.join(save_dir, f"{subject_id}_epochs_wide_cue.csv")
    epochs_wide_outcome_out = os.path.join(save_dir, f"{subject_id}_epochs_wide_outcome.csv")

    epochs_wide_cue.to_csv(epochs_wide_cue_out, index=False)
    epochs_wide_outcome.to_csv(epochs_wide_outcome_out, index=False)

    print(f"[SAVE] epochs_wide_cue     -> {epochs_wide_cue_out}")
    print(f"[SAVE] epochs_wide_outcome -> {epochs_wide_outcome_out}")

    # --- Save full epochs_wide and betas_long ---
    epochs_wide_out = os.path.join(save_dir, f"{subject_id}_epochs_wide.csv")
    betas_out_path  = os.path.join(save_dir, f"{subject_id}_betas_timecourse.csv")

    # Note: saving epochs_wide
    epochs_wide.to_csv(epochs_wide_out, index=False)
    betas_long.to_csv(betas_out_path, index=False)

    print(f"[SAVE] epochs_wide   -> {epochs_wide_out}")
    print(f"[SAVE] betas_long    -> {betas_out_path}")

    return epochs_wide, betas_long

#=================RUN SCRIPT=================#

# ---- Paths: edit these for your own system/data layout ----
base_dir = r"/quobyte/dfoxgrp/BACKED-UP/RDC/data/pe_analysis"
subj_root = r"/quobyte/dfoxgrp/BACKED-UP/RDC/data/InitialAnalysisStartN104_20241111"
save_path = r"/quobyte/dfoxgrp/BACKED-UP/RDC/data/pe_analysis/pe_output"

#start_subj = "RDC167"

# Get all subject directories
subj_dirs = sorted(glob.glob(os.path.join(subj_root, "RDC*")))
print(f"[INFO] N subjects: {len(subj_dirs)}")

subj_dirs = [d for d in subj_dirs if rdc_num_from_dir(d) >= START_ID]
print(f"[INFO] N subjects >= RDC{START_ID}: {len(subj_dirs)}")

if args.only_subject is not None:
    subj_dirs = [d for d in subj_dirs if os.path.basename(d) == args.only_subject]
    print(f"[INFO] Running only subject: {args.only_subject}")


roi_dir = os.path.join(base_dir, "MDART_timecourse_ROIs") #<-- adjust if your ROI folder name differs
print(f"[PATH] ROI dir    = {roi_dir}")

roi_bank = load_roi_bank_from_dir(
    roi_dir,
    materialize_masks=True,
    require_binary=True,
)

for subj_dir in subj_dirs:
    subj_id = os.path.basename(subj_dir)

    print("\n==============================")
    print(f"Processing subject: {subj_id}")
    print("==============================")

    epochs_wide_subj, betas_subj = run_timecourse_for_subject(
        subj_dir=subj_dir,
        roi_bank=roi_bank,
        base_dir=base_dir,
        save_path=save_path,
        tr=1.25,
        upsample_hz=10.0,
        tmin=-1.0,
        tmax=16.1,
        run_ids=(1, 2, 3, 4),
    )
