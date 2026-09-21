"""
Whole-brain signature-expression timecourse extraction and trial-wise
regression (complete pipeline).

For each subject and each whole-brain signature (Ceko negative affect,
LiuBecker ambiguous threat anticipation): computes a whole-brain
signature-expression timecourse per run, upsamples and epochs it around cue
and outcome onsets, then fits a combined single-model regression (risky +
ambiguous trials together, via trial-type-gated predictors) at every
timepoint. This is the single-model pipeline used for the paper; an earlier
version of this script fit risky and ambiguous trials as two separate
simpler models and a follow-up script (`wholebrain_signature_exp_refit_...`)
refit the combined model on its saved output -- both steps are now merged
into this one script.

Inputs (per subject, under SUBJ_ROOT):
    func/*rec-S4WMgRado*.nii.gz    preprocessed BOLD, one file per run
    beh/*.txt                     behavioral event files, one file per run
Signature weight images: SIGNATURE_PATHS (3D NIfTI, same space as BOLD data).

Outputs (per subject, under SAVE_PATH/<subject>/MDART_timecourse_wholebrain_signature_expression):
    <subject>_epochs_wide_MDART_wholebrain_sig_exp.csv            all epochs (cue + outcome, all signatures)
    <subject>_epochs_wide_cue_MDART_wholebrain_sig_exp.csv        cue-aligned epochs only
    <subject>_epochs_wide_outcome_MDART_wholebrain_sig_exp.csv    outcome-aligned epochs only
    <subject>_betas_timecourse_MDART_wholebrain_sig_exp.csv       per-timepoint betas, averaged across runs
    <subject>_betas_timecourse_MDART_wholebrain_sig_exp_BYRUN.csv per-timepoint betas, one row per run

Usage:
    python timecourse_analysis_signature_expression.py [--start_id 102] [--only_subject RDC167]
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


def extract_signature_expression_timeseries(
    fmri_path,
    signature_path,
    TR,
    detrend=False,
    zscore=False,
    expression_metric="dot",
):
    """
    Extract a whole-brain signature-expression timecourse from a 4D fMRI run.

    At each TR, this computes the multivoxel pattern expression of the
    whole-brain fMRI signal against a 3D signature image.

    Parameters
    ----------
    fmri_path : str or path-like
        Path to 4D fMRI NIfTI file.
    signature_path : str or path-like
        Path to 3D signature-weight NIfTI file.
    TR : float
        Repetition time in seconds.
    detrend : bool, default=False
        If True, remove linear trend (intercept + time) from each voxel's
        timecourse before computing expression.
    zscore : bool, default=False
        If True, z-score each voxel across time before computing expression.
    expression_metric : {"dot", "cosine"}, default="dot"
        - "dot": weighted sum across voxels at each timepoint
        - "cosine": cosine similarity between voxel pattern and signature
          at each timepoint

    Returns
    -------
    t_run : np.ndarray, shape (T,)
        Time points in seconds.
    expr : np.ndarray, shape (T,)
        Signature-expression timecourse, one value per TR.
    """
    # Load data
    fmri_img = nib.load(fmri_path)
    fmri_data = fmri_img.get_fdata()   # (X, Y, Z, T)

    sig_img = nib.load(signature_path)
    sig_data = sig_img.get_fdata()     # (X, Y, Z)

    # Basic checks
    if fmri_data.ndim != 4:
        raise ValueError(f"Expected 4D fMRI data, got shape {fmri_data.shape}")

    if sig_data.ndim != 3:
        raise ValueError(f"Expected 3D signature image, got shape {sig_data.shape}")

    if sig_data.shape != fmri_data.shape[:3]:
        raise ValueError(
            f"Signature shape {sig_data.shape} does not match "
            f"fMRI spatial shape {fmri_data.shape[:3]}"
        )
    
    if not np.allclose(sig_img.affine, fmri_img.affine):
        print("[WARN] Signature and fMRI affines differ.")

    # Build valid voxel mask
    valid_mask = np.isfinite(sig_data) & (sig_data != 0)

    n_vox = int(valid_mask.sum())
    if n_vox == 0:
        raise ValueError("Whole-brain valid mask contains zero voxels.")

    # Extract voxel data
    wb_fmri = fmri_data[valid_mask]        # (V, T)
    Y = wb_fmri.astype(float).T            # (T, V)
    w = sig_data[valid_mask].astype(float) # (V,)

    # Handle non-finite values in fMRI / signature
    Y[~np.isfinite(Y)] = np.nan
    w[~np.isfinite(w)] = 0.0

    # Time axis
    Tn = Y.shape[0]
    t_run = np.arange(Tn, dtype=float) * float(TR)

    # Optional detrending: voxelwise across time
    if detrend:
        t = t_run[:, None]
        X = np.c_[np.ones_like(t), t]
        beta, *_ = np.linalg.lstsq(X, np.nan_to_num(Y, nan=0.0), rcond=None)
        Y = Y - X @ beta

    # Optional z-score: voxelwise across time
    if zscore:
        m = np.nanmean(Y, axis=0, keepdims=True)
        s = np.nanstd(Y, axis=0, ddof=1, keepdims=True)
        s[s == 0] = 1.0
        Y = (Y - m) / s

    # Replace remaining NaNs before expression computation
    Y = np.nan_to_num(Y, nan=0.0)

    # Compute expression
    if expression_metric == "dot":
        expr = Y @ w

    elif expression_metric == "cosine":
        y_norm = np.linalg.norm(Y, axis=1)
        w_norm = np.linalg.norm(w)
        denom = y_norm * w_norm
        denom[denom == 0] = np.nan
        expr = (Y @ w) / denom

    else:
        raise ValueError("expression_metric must be 'dot' or 'cosine'")

    return t_run, expr

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
    Epoch a high-resolution timeseries into trial-locked segments.

    Parameters
    ----------
    t_hi : np.ndarray, shape (T_hi,)
        High-resolution time grid in seconds (from upsample_timeseries).
    y_hi : np.ndarray, shape (T_hi,)
        High-resolution signal (1D for now).
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

def compute_betas(df, verbose=False):
    """
    Fit combined single model at each timepoint for one dataframe
    (one subject x one alignment x one signature x one run).

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

    Replaces an earlier version of this pipeline that fit risky and
    ambiguous trials as two separate simpler models; this single combined
    model (with trial-type-gated predictors) is the one used for the paper.
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

    if verbose:
        print(
            df.groupby("trial_kind_label")[
                ["p_threat_risky", "r_risk_risky", "rational_p_threat_amb", "a_amb_amb"]
            ].mean()
        )

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
        return float(c[2:-1])

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

    return out

#=================MAIN LOOP=================#

def run_timecourse_for_subject(
    subj_dir,
    base_dir,
    save_path,
    signature_paths,
    tr=1.25,
    upsample_hz=10.0,
    tmin=-1.0,
    tmax=16.1,
    run_ids=(1, 2, 3, 4),
):
    """
    Run signature-expression timecourse extraction + trial-wise regression
    for a single subject across all specified runs, ROIs, and signatures.

    Returns
    -------
    epochs_wide, betas_long : pd.DataFrame
    """

    # ----- 0. Resolve paths & subject ID -----
    subject_id = os.path.basename(subj_dir)
    print(f"[INFO] Subject {subject_id}")

    save_dir = os.path.join(save_path, subject_id, "MDART_timecourse_wholebrain_signature_expression")
    os.makedirs(save_dir, exist_ok=True)

    func_dir = os.path.join(subj_dir, "func")
    beh_dir  = os.path.join(subj_dir, "beh")

    fmri_by_run = map_files_by_run(os.path.join(func_dir, "*rec-S4WMgRado*.nii.gz"))
    beh_by_run  = map_files_by_run(os.path.join(beh_dir, "*.txt"))

    available_runs = sorted(set(fmri_by_run.keys()) & set(beh_by_run.keys()))
    selected_runs = [r for r in run_ids if r in available_runs]

    if not selected_runs:
        raise RuntimeError(f"No valid runs for subject {subject_id}. Found runs: {available_runs}")

    print(f"[INFO]   Using runs: {selected_runs}")

    # ----- 1. Extract epochs for all runs × ROIs × signatures -----
    all_epoch_dfs = []
    roi_name = "whole-brain"  # placeholder since we're not doing ROIs here

    for run_id in selected_runs:
        fmri_path = fmri_by_run[run_id]
        beh_path  = beh_by_run[run_id]

        events_df = load_events_table(beh_path)
        events_df = add_derived_task_columns(events_df)
        print(f"[INFO]   Processing run {run_id}: {len(events_df)} trials")

        for sig_name, sig_path in signature_paths.items():
            print(f"[INFO]       Signature: {sig_name}")

            # 1) signature expression at TR resolution
            t_run, y_run = extract_signature_expression_timeseries(
                fmri_path=fmri_path,
                signature_path=sig_path,
                TR=tr,
                detrend=True,
                zscore=True,
                expression_metric="dot",
            )

            # 2) upsample to high-resolution grid
            t_hi, y_hi, dt = upsample_timeseries(
                t_run,
                y_run,
                upsample_hz=upsample_hz,
                TR=tr,
            )

            # 3.1) cue-aligned epochs
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

            df_epochs_cue["alignment"] = "cue"
            df_epochs_cue["subject"] = subject_id
            df_epochs_cue["run"] = run_id
            df_epochs_cue["roi"] = roi_name
            df_epochs_cue["signature"] = sig_name
            df_epochs_cue["dt"] = dt
            df_epochs_cue["TR"] = tr

            all_epoch_dfs.append(df_epochs_cue)

            # 3.2) outcome-aligned epochs
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

            df_epochs_outcome["alignment"] = "outcome"
            df_epochs_outcome["subject"] = subject_id
            df_epochs_outcome["run"] = run_id
            df_epochs_outcome["roi"] = roi_name
            df_epochs_outcome["signature"] = sig_name
            df_epochs_outcome["dt"] = dt
            df_epochs_outcome["TR"] = tr

            all_epoch_dfs.append(df_epochs_outcome)

    if not all_epoch_dfs:
        raise RuntimeError(f"No epochs produced for subject {subject_id}.")

    epochs_long = pd.concat(all_epoch_dfs, ignore_index=True)
    print("[OK] epochs_long:", epochs_long.shape, "rows")

    # ----- 3. Long → wide -----
    id_cols = [
        "subject", "run", "roi", "signature",
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
            index=id_cols + ["alignment"],
            columns="time_rel_s",
            values="signal"
        )
        .reset_index()
    )

    new_cols = []
    for c in epochs_wide.columns:
        if isinstance(c, (float, int)):
            new_cols.append(f"t_{c:0.1f}s")
        else:
            new_cols.append(c)
    epochs_wide.columns = new_cols

    print("[OK] epochs_wide:", epochs_wide.shape, "rows x cols")

    epochs_wide_cue = epochs_wide.query("alignment == 'cue'")
    epochs_wide_outcome = epochs_wide.query("alignment == 'outcome'")

    print("[OK] epochs_wide_cue:", epochs_wide_cue.shape, "rows x cols")
    print("[OK] epochs_wide_outcome:", epochs_wide_outcome.shape, "rows x cols")

    # ----- 4. Timepoint-wise regression (combined single model) -----
    all_betas = []
    all_betas_by_run = []

    for alignment_name, df_align in [
        ("cue", epochs_wide_cue),
        ("outcome", epochs_wide_outcome),
    ]:
        print(f"[INFO] Running regression for alignment: {alignment_name}")

        if df_align.empty:
            print(f"[WARN]   No trials for alignment={alignment_name}, skipping")
            continue

        for sig_name, df_sig in df_align.groupby("signature", sort=False):
            print(f"[INFO]   Signature: {sig_name} ({len(df_sig)} trials)")

            run_betas = []

            for run_id, df_run in df_sig.groupby("run", sort=False):
                n_trials = len(df_run)
                if n_trials < 6:
                    print(f"[WARN]     run {run_id}: too few trials ({n_trials}), skipping")
                    continue

                b = compute_betas(df_run)
                b["run"] = run_id
                b["n_trials"] = n_trials

                run_betas.append(b)

                b_run_out = b.copy()
                b_run_out["subject"] = subject_id
                b_run_out["signature"] = sig_name
                b_run_out["alignment"] = alignment_name
                all_betas_by_run.append(b_run_out)

            if not run_betas:
                print(f"[WARN]   Signature {sig_name}: no runs with enough trials")
                continue

            betas_run_long = pd.concat(run_betas, ignore_index=True)

            # identify beta columns dynamically
            beta_cols = [c for c in betas_run_long.columns if c.startswith("beta_")]

            betas_mean = (
                betas_run_long
                .groupby("time_s", as_index=False)[beta_cols]
                .mean()
            )

            betas_mean["subject"] = subject_id
            betas_mean["signature"] = sig_name
            betas_mean["alignment"] = alignment_name
            betas_mean["n_trials_used"] = (betas_run_long[["run", "n_trials"]].drop_duplicates()["n_trials"].sum())
            betas_mean["n_runs_used"] = betas_run_long["run"].nunique()

            all_betas.append(betas_mean)

    if not all_betas:
        raise RuntimeError(f"No betas produced for subject {subject_id}.")

    betas_long = pd.concat(all_betas, ignore_index=True)

    if all_betas_by_run:
        betas_by_run_long = pd.concat(all_betas_by_run, ignore_index=True)
        betas_by_run_path = os.path.join(save_dir, f"{subject_id}_betas_timecourse_MDART_wholebrain_sig_exp_BYRUN.csv")
        betas_by_run_long.to_csv(betas_by_run_path, index=False)
        print(f"[SAVE] betas_by_run -> {betas_by_run_path}")


    # ----- 5. Save outputs -----
    epochs_wide_cue_out = os.path.join(save_dir, f"{subject_id}_epochs_wide_cue_MDART_wholebrain_sig_exp.csv")
    epochs_wide_outcome_out = os.path.join(save_dir, f"{subject_id}_epochs_wide_outcome_MDART_wholebrain_sig_exp.csv")

    epochs_wide_cue.to_csv(epochs_wide_cue_out, index=False)
    epochs_wide_outcome.to_csv(epochs_wide_outcome_out, index=False)

    print(f"[SAVE] epochs_wide_cue     -> {epochs_wide_cue_out}")
    print(f"[SAVE] epochs_wide_outcome -> {epochs_wide_outcome_out}")

    epochs_wide_out = os.path.join(save_dir, f"{subject_id}_epochs_wide_MDART_wholebrain_sig_exp.csv")
    betas_out_path = os.path.join(save_dir, f"{subject_id}_betas_timecourse_MDART_wholebrain_sig_exp.csv")

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

# -----------------------------
# Signature setup
# -----------------------------
signature_path = {
    "Ceko_negative_affect": os.path.join(
        base_dir, "signature_weights", "CekoGeneralNegativeAffect_weights.nii.gz"
    ),
    "LiuBecker_threat_anticipation": os.path.join(
        base_dir, "signature_weights", "LiuBeckerAmbiguousThreat_weights.nii.gz"
    ),
}

for subj_dir in subj_dirs:
    subj_id = os.path.basename(subj_dir)

    save_dir = os.path.join(save_path, subj_id, "MDART_timecourse_wholebrain_signature_expression")  # adjust if needed

    print("\n==============================")
    print(f"Processing subject: {subj_id}")
    print("==============================")

    print("[INFO] Signatures to process:")
    for sig_name in signature_path.keys():
        print(f"   - {sig_name}")

    epochs_wide_subj, betas_subj = run_timecourse_for_subject(
        subj_dir=subj_dir,
        signature_paths=signature_path,
        base_dir=base_dir,
        save_path=save_path,
        tr=1.25,
        upsample_hz=10.0,
        tmin=-1.0,
        tmax=16.1,
        run_ids=(1, 2, 3, 4),
    )
