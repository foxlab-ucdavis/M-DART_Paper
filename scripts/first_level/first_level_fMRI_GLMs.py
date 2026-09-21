"""
First-level fMRI GLM (per subject).

Builds a first-level design matrix (design type selectable via CLI), convolves
task regressors with the HRF, appends nuisance regressors and motion-censoring,
regresses the concatenated 4-run BOLD timeseries, and writes one beta map per
regressor of interest.

Inputs (per subject, under DATA_ROOT):
    func/*rec-S4WMgRado*.nii.gz          preprocessed BOLD, one file per run
    func/*<NUISANCE_STRING>              nuisance regressors, one file per run
    func/*CensoredVolumeMatrix.csv       motion-censoring matrix, one file per run
    beh/*.txt                            behavioral event files, one file per run

Outputs (per subject, under OUTPUT_ROOT/<subject>/<OUTPUT_DIR>):
    design_matrix.csv                    final design matrix (post-censoring)
    <regressor>.nii.gz                   beta map, one per task regressor

Usage:
    python first_level_fMRI_GLMs.py <DESIGN_MATRIX> <OUTPUT_DIR> <NUISANCE_STRING>

    DESIGN_MATRIX   one of: parametric_modulation, cueType,
                    UncertainThreat_vs_CertainThreat, CertainThreat_vs_CertainSafe,
                    Amb_vs_Unamb, low_vs_none,
                    c102030vs0, c405060vs102030, c708090vs405060, c100vs708090,
                    c10vs0, c50vs0
    OUTPUT_DIR      subdirectory name (under OUTPUT_ROOT/<subject>) for beta maps
    NUISANCE_STRING filename pattern for the nuisance regressor file, e.g.
                    'FullNuisanceVariableSet.csv'
"""

# Standard library
import os
import re
import sys
import glob

# Third-party libraries
import numpy as np
import pandas as pd
import nibabel as nib

# nltools (neuroimaging-specific)
from nltools.data import Design_Matrix
from nltools.external import hrf
from nltools.data import Brain_Data

# ---- Paths: edit these for your own system/data layout ----
DATA_ROOT = '/quobyte/dfoxgrp/BACKED-UP/RDC/data/InitialAnalysisStartN104_20241111'
OUTPUT_ROOT = '/quobyte/dfoxgrp/BACKED-UP/RDC/data/pe_analysis/pe_output'


def run_first_level_glm_single_subject(DESIGN_MATRIX, OUTPUT_DIR, NUISANCE_STRING):

    # ---- Global parameters ----
    TR = 1.25  # Repetition time in seconds
    UPSAMPLE_FROM_SECS = 1000  # Upsampling resolution for design matrix construction (in ms)

    # ---- Confirm script parameters ----
    print('Running:', DESIGN_MATRIX, OUTPUT_DIR, NUISANCE_STRING)

    # ---- Get list of subject directories ----
    subj_dirs = glob.glob(os.path.join(DATA_ROOT, 'RDC*'))

    for subj_dir in subj_dirs:
        subj_id = os.path.basename(subj_dir)
        print(f'Processing subject: {subj_id}')

        # ---- Get fMRI files for this subject ----
        fmri_files = glob.glob(os.path.join(subj_dir, 'func', '*rec-S4WMgRado*.nii.gz'))
        fmri_files_dict = {
            int(re.findall(r'task-MDART_run-(\d+)', f)[0]): f
            for f in fmri_files
        }

        # Load one fMRI file to compute total scan length (assumes all runs same length)
        fmri_data = nib.load(fmri_files_dict[1])
        scan_length_secs = fmri_data.shape[3] * TR
        print(f'Total scan length: {scan_length_secs} seconds')

        # ---- Get nuisance regressor files ----
        nuisance_files = glob.glob(os.path.join(subj_dir, 'func', f'*{NUISANCE_STRING}'))
        nuisance_files_dict = {
            int(re.findall(r'task-MDART_run-(\d+)', f)[0]): f
            for f in nuisance_files
        }

        # ---- Get behavioral event files ----
        behavioral_files = glob.glob(os.path.join(subj_dir, 'beh', '*.txt'))
        behavioral_files_dict = {
            int(re.findall(r'task-MDART_run-(\d+)', f)[0]): f
            for f in behavioral_files
        }

        # ---- Get censoring (motion scrubbing) files ----
        censor_files = glob.glob(os.path.join(subj_dir, 'func', '*CensoredVolumeMatrix.csv'))
        censor_files_dict = {
            int(re.findall(r'task-MDART_run-(\d+)', f)[0]): f
            for f in censor_files
        }

        def compute_demean_range01(vals):
            """
            Scale input values to [0, 1] range, then mean-center and double center.

            Parameters:
                vals (pd.Series or np.ndarray): Input values to scale and center

            Returns:
                model_vals (same type as input): Transformed values
            """
            # Step 1: scale to [0, 1]
            model_vals = (vals - np.min(vals)) / (np.max(vals) - np.min(vals))

            # Step 2–3: mean-center and double center
            model_vals = model_vals - np.mean(model_vals)
            model_vals = 2 * (model_vals - np.mean(model_vals))

            # Step 4: handle NaNs
            model_vals.fillna(0, inplace=True)  # updated 6/25 to avoid cross-type averaging

            return model_vals

        def create_parametric_modulation_design_matrix(df, scan_length_secs, n_runs, UPSAMPLE_FROM_SECS):
            """
            Construct a design matrix that models cue, outcome, and rating periods
            with parametric regressors.

            """
            dm_columns_list = [
                'cue_const_risk', 'cue_p_risk', 'cue_r_risk',
                'outcome_const', 'outcome',
                'cue_const_amb', 'cue_p_amb', 'cue_a_amb',
                'rating'
            ]
            dm_columns_dict = {col: i for i, col in enumerate(dm_columns_list)}
            num_regressors = len(dm_columns_list)
            total_timepoints = int(scan_length_secs * n_runs * UPSAMPLE_FROM_SECS)
            dm_array = np.zeros((num_regressors, total_timepoints))

            for _, row in df.iterrows():
                start = int(row['TrialStartTime'])
                outcome_onset = int(row['ReinforcerInTime'])
                rating_onset = int(row['RatingStartTime'])
                rating_offset = rating_onset + int(row['RatingRT'])

                if row['TrialTypeCode'] != 3:  # Risky trials
                    dm_array[dm_columns_dict['cue_const_risk'], start:outcome_onset] = 1
                    dm_array[dm_columns_dict['cue_p_risk'], start:outcome_onset] = row['p_risk']
                    dm_array[dm_columns_dict['cue_r_risk'], start:outcome_onset] = row['r_risk']

                else:  # Ambiguous trials
                    dm_array[dm_columns_dict['cue_const_amb'], start:outcome_onset] = 1
                    dm_array[dm_columns_dict['cue_p_amb'], start:outcome_onset] = row['p_amb']
                    dm_array[dm_columns_dict['cue_a_amb'], start:outcome_onset] = row['a_amb']

                # Outcome phase (pooled across trial types)
                dm_array[dm_columns_dict['outcome_const'], outcome_onset:rating_onset] = 1
                dm_array[dm_columns_dict['outcome'], outcome_onset:rating_onset] = row['outcome']

                # Rating period
                dm_array[dm_columns_dict['rating'], rating_onset:rating_offset] = 1

            return dm_array, dm_columns_dict

        def create_UncertainThreat_vs_CertainThreat_design_matrix(df, scan_length_secs, n_runs, UPSAMPLE_FROM_SECS):
            """
            Construct a design matrix to contrast uncertain vs. certain threat during cue periods.

            """
            dm_columns_list = [
                'cue_NOT_UncertainThreat_and_CertainThreat',  # Binary regressor for Certain Safe only
                'cue_UncertainThreat_and_CertainThreat',      # Binary regressor for Risky, Ambiguous, Certain Threat
                'cue_UncertainThreat_vs_CertainThreat',       # Parametric contrast: Uncertain Threat (Risky + Ambiguous: 1) vs. Certain Threat (-1), NaN elsewhere
                'outcome_const',
                'outcome',
                'rating'
            ]
            dm_columns_dict = {col: i for i, col in enumerate(dm_columns_list)}
            num_regressors = len(dm_columns_list)
            total_timepoints = int(scan_length_secs * n_runs * UPSAMPLE_FROM_SECS)
            dm_array = np.zeros((num_regressors, total_timepoints))

            # Populate design matrix
            for _, row in df.iterrows():
                start = int(row['TrialStartTime'])
                outcome_onset = int(row['ReinforcerInTime'])
                rating_onset = int(row['RatingStartTime'])
                rating_end = rating_onset + int(row['RatingRT'])

                # Cue period 
                dm_array[dm_columns_dict['cue_NOT_UncertainThreat_and_CertainThreat'], start:outcome_onset] = row['NOT_UncertainThreat_and_CertainThreat']
                dm_array[dm_columns_dict['cue_UncertainThreat_and_CertainThreat'], start:outcome_onset] = row['UncertainThreat_and_CertainThreat']
                dm_array[dm_columns_dict['cue_UncertainThreat_vs_CertainThreat'], start:outcome_onset] = row['UncertainThreat_vs_CertainThreat']

                # Outcome period 
                dm_array[dm_columns_dict['outcome_const'], outcome_onset:rating_onset] = 1
                dm_array[dm_columns_dict['outcome'], outcome_onset:rating_onset] = row['outcome']

                # Rating period
                if row['RatingValue'] != -1:
                    dm_array[dm_columns_dict['rating'], rating_onset:rating_end] = 1

            return dm_array, dm_columns_dict

        def create_CertainThreat_vs_CertainSafe_design_matrix(df, scan_length_secs, n_runs, UPSAMPLE_FROM_SECS):
            """
            Construct a design matrix to contrast certain threat vs. certain safe trials.

            """
            dm_columns_list = [
                'cue_NOT_CertainThreat_and_CertainSafe', # Binary regressor for uncertain trials (Risk + Ambiguous)
                'cue_CertainThreat_and_CertainSafe',     # Binary regressor for trials with certain outcomes (Safe + Certain Threat)
                'cue_CertainThreat_vs_CertainSafe',      # Parametric contrast: Certain Threat (+1) vs. Certain Safe (–1), NaN elsewhere
                'outcome_const',
                'outcome',
                'rating'
            ]
            dm_columns_dict = {col: i for i, col in enumerate(dm_columns_list)}
            num_regressors = len(dm_columns_list)
            total_timepoints = int(scan_length_secs * n_runs * UPSAMPLE_FROM_SECS)
            dm_array = np.zeros((num_regressors, total_timepoints))

            for _, row in df.iterrows():
                start = int(row['TrialStartTime'])
                outcome_onset = int(row['ReinforcerInTime'])
                rating_onset = int(row['RatingStartTime'])
                rating_end = rating_onset + int(row['RatingRT'])

                # Cue period
                dm_array[dm_columns_dict['cue_NOT_CertainThreat_and_CertainSafe'], start:outcome_onset] = row['NOT_CertainThreat_and_CertainSafe']
                dm_array[dm_columns_dict['cue_CertainThreat_and_CertainSafe'], start:outcome_onset] = row['CertainThreat_and_CertainSafe']
                dm_array[dm_columns_dict['cue_CertainThreat_vs_CertainSafe'], start:outcome_onset] = row['CertainThreat_vs_CertainSafe']

                # Outcome period
                dm_array[dm_columns_dict['outcome_const'], outcome_onset:rating_onset] = 1
                dm_array[dm_columns_dict['outcome'], outcome_onset:rating_onset] = row['outcome']

                # Rating period
                if row['RatingValue'] != -1:
                    dm_array[dm_columns_dict['rating'], rating_onset:rating_end] = 1

            return dm_array, dm_columns_dict

        def create_Amb_vs_Risk_design_matrix(df, scan_length_secs, n_runs, UPSAMPLE_FROM_SECS):
            """
            Construct a design matrix to contrast ambiguous vs. unambiguous trials during the cue phase.

            """
            dm_columns_list = [
                'cue_NOT_Amb_and_Risk',   # Binary regressor for trials that are NOT ambiguous or risky (e.g., Certain Safe, Certain Threat)
                'cue_Amb_and_Risk',       # Binary regressor for trials that are ambiguous or risky
                'cue_Amb_vs_Risk',        # Parametric contrast: Ambiguous (+1) vs. Risky (–1), NaN elsewhere
                'outcome_const',          
                'outcome',               
                'rating'                  
            ]
            dm_columns_dict = {col: i for i, col in enumerate(dm_columns_list)}
            num_regressors = len(dm_columns_list)
            total_timepoints = int(scan_length_secs * n_runs * UPSAMPLE_FROM_SECS)
            dm_array = np.zeros((num_regressors, total_timepoints))

            for _, row in df.iterrows():
                start = int(row['TrialStartTime'])
                outcome_onset = int(row['ReinforcerInTime'])
                rating_onset = int(row['RatingStartTime'])
                rating_end = rating_onset + int(row['RatingRT'])

                # Cue period
                dm_array[dm_columns_dict['cue_NOT_Amb_and_Risk'], start:outcome_onset] = row['NOT_Amb_and_Risk']
                dm_array[dm_columns_dict['cue_Amb_and_Risk'], start:outcome_onset] = row['Amb_and_Risk']
                dm_array[dm_columns_dict['cue_Amb_vs_Risk'], start:outcome_onset] = row['Amb_vs_Risk']

                # Outcome period
                dm_array[dm_columns_dict['outcome_const'], outcome_onset:rating_onset] = 1
                dm_array[dm_columns_dict['outcome'], outcome_onset:rating_onset] = row['outcome']

                # Rating period
                if row['RatingValue'] != -1:
                    dm_array[dm_columns_dict['rating'], rating_onset:rating_end] = 1

            return dm_array, dm_columns_dict

        def create_low_vs_none_design_matrix(df, scan_length_secs, n_runs, UPSAMPLE_FROM_SECS):
            """
            Construct a design matrix to compare low-threat vs. no-threat trials at cue time.

            """
            dm_columns_list = [
                'cue_NOT_low_and_none',   # Binary regressor for trials not in low/no threat group
                'cue_low_and_none',       # Binary regressor for low threat and no threat trials
                'cue_low_vs_none',        # Parametric contrast: Low threat (+1) vs. No threat (–1), NaN elsewhere
                'outcome_const',          
                'outcome',               
                'rating'                  
            ]
            dm_columns_dict = {col: i for i, col in enumerate(dm_columns_list)}
            num_regressors = len(dm_columns_list)
            total_timepoints = int(scan_length_secs * n_runs * UPSAMPLE_FROM_SECS)
            dm_array = np.zeros((num_regressors, total_timepoints))

            for _, row in df.iterrows():
                start = int(row['TrialStartTime'])
                outcome_onset = int(row['ReinforcerInTime'])
                rating_onset = int(row['RatingStartTime'])
                rating_end = rating_onset + int(row['RatingRT'])

                # Cue period
                dm_array[dm_columns_dict['cue_NOT_low_and_none'], start:outcome_onset] = row['NOT_low_and_none']
                dm_array[dm_columns_dict['cue_low_and_none'], start:outcome_onset] = row['low_and_none']
                dm_array[dm_columns_dict['cue_low_vs_none'], start:outcome_onset] = row['low_vs_none']

                # Outcome period
                dm_array[dm_columns_dict['outcome_const'], outcome_onset:rating_onset] = 1
                dm_array[dm_columns_dict['outcome'], outcome_onset:rating_onset] = row['outcome']

                # Rating period
                if row['RatingValue'] != -1:
                    dm_array[dm_columns_dict['rating'], rating_onset:rating_end] = 1

            return dm_array, dm_columns_dict

        def create_cueType_design_matrix(df, scan_length_secs, n_runs, UPSAMPLE_FROM_SECS):
            """
            Construct a design matrix to model cue type (unambiguous vs. ambiguous) and 
            specific cue probability/ambiguity levels.

            """
            dm_columns_list = [
                'cue_risk_const',    # Risky trials: cue period constant boxcar
                'cue_amb_const',     # Ambiguous trials: cue period constant boxcar
                'outcome_const',     # Outcome period: constant boxcar
                'outcome',           # Outcome period: outcome-modulated
                'rating'             # Rating period: response duration
            ]

            # Add regressors for unique cue probabilities and ambiguity levels
            dm_columns_list.extend(list(df['cue_prob'].unique()))
            dm_columns_list.extend(list(df['cue_amb'].unique()))

            dm_columns_dict = {col: i for i, col in enumerate(dm_columns_list)}
            num_regressors = len(dm_columns_list)
            total_timepoints = int(scan_length_secs * n_runs * UPSAMPLE_FROM_SECS)
            dm_array = np.zeros((num_regressors, total_timepoints))

            for _, row in df.iterrows():
                start = int(row['TrialStartTime'])
                outcome_onset = int(row['ReinforcerInTime'])
                rating_onset = int(row['RatingStartTime'])
                rating_end = rating_onset + int(row['RatingRT'])

                if row['TrialTypeCode'] != 3:  # Risky trials
                    if row['cue_prob'] != '':
                        dm_array[dm_columns_dict['cue_risk_const'], start:outcome_onset] = 1
                        dm_array[dm_columns_dict[row['cue_prob']], start:outcome_onset] = 1

                else:  # Ambiguous trials
                    if row['cue_prob'] != '':
                        dm_array[dm_columns_dict['cue_amb_const'], start:outcome_onset] = 1
                        dm_array[dm_columns_dict[row['cue_prob']], start:outcome_onset] = 1
                    if row['cue_amb'] != '':
                        dm_array[dm_columns_dict[row['cue_amb']], start:outcome_onset] = 1

                # Outcome period
                dm_array[dm_columns_dict['outcome_const'], outcome_onset:rating_onset] = 1
                dm_array[dm_columns_dict['outcome'], outcome_onset:rating_onset] = row['outcome']

                # Rating period
                if row['RatingValue'] != -1:
                    dm_array[dm_columns_dict['rating'], rating_onset:rating_end] = 1

            return dm_array, dm_columns_dict

        def create_comparison_design_matrix(df, comparison, scan_length_secs, n_runs, UPSAMPLE_FROM_SECS):
            """
            Construct a design matrix for a specified cue-probability comparison.

            """
            dm_columns_list = [
                comparison + '_AND_const',   # Marks all trials included in this comparison
                comparison + '_NOT_const',   # Marks all other trials (not in the comparison)
                comparison,                  # Parametric contrast: High (+1) vs. Low (-1)
                'outcome_const',             # Boxcar during outcome period
                'outcome',                   # Parametric regressor scaled by trial outcome
                'rating'                     # Boxcar during rating period
            ]
            dm_columns_dict = {col: i for i, col in enumerate(dm_columns_list)}
            num_regressors = len(dm_columns_list)
            total_timepoints = int(scan_length_secs * n_runs * UPSAMPLE_FROM_SECS)
            dm_array = np.zeros((num_regressors, total_timepoints))

            for _, row in df.iterrows():
                start = int(row['TrialStartTime'])
                outcome_onset = int(row['ReinforcerInTime'])
                rating_onset = int(row['RatingStartTime'])
                rating_end = rating_onset + int(row['RatingRT'])

                # Cue period
                dm_array[dm_columns_dict[comparison + '_AND_const'], start:outcome_onset] = row[comparison + '_AND_const']
                dm_array[dm_columns_dict[comparison + '_NOT_const'], start:outcome_onset] = row[comparison + '_NOT_const']
                dm_array[dm_columns_dict[comparison], start:outcome_onset] = row[comparison]

                # Outcome period
                dm_array[dm_columns_dict['outcome_const'], outcome_onset:rating_onset] = 1
                dm_array[dm_columns_dict['outcome'], outcome_onset:rating_onset] = row['outcome']

                # Rating period
                if row['RatingValue'] != -1:
                    dm_array[dm_columns_dict['rating'], rating_onset:rating_end] = 1

            return dm_array, dm_columns_dict

        # --------------------------------------------------------------------
        # Load and combine behavioral data across runs
        # --------------------------------------------------------------------
        df_list = []
        for run in np.arange(1, 5):   # Loop over runs 1–4
            df_run = pd.read_csv(behavioral_files_dict[run], sep='\t', comment='#')
            df_run['run'] = run
            df_list.append(df_run)
        df = pd.concat(df_list, axis=0)

        # --------------------------------------------------------------------
        # Initialize threat probability variables (Risk vs Ambiguity)
        # --------------------------------------------------------------------
        df['ThreatPctRisk'] = df['ThreatPct'].where(df['TrialTypeCode'] != 3, np.nan)
        df['ThreatPctAmb'] = df['ThreatPct'].where(df['TrialTypeCode'] == 3, np.nan)
        df['AmbiguousPctAmb'] = df['AmbiguousPct'].where(df['TrialTypeCode'] == 3, np.nan)

        # --------------------------------------------------------------------
        # Parametric modulators for Risk and Ambiguity
        # --------------------------------------------------------------------
        df['p_risk'] = compute_demean_range01(df['ThreatPctRisk'] / 100)
        df['r_risk'] = compute_demean_range01(0.5**2 - ((df['ThreatPctRisk'] / 100) - 0.5) ** 2)  # risk magnitude

        df['p_amb'] = compute_demean_range01(
            (df['ThreatPctAmb'] / 100) + (df['AmbiguousPctAmb'] / 100) / 2
        )  # ambiguous probability estimate
        df['a_amb'] = compute_demean_range01(df['AmbiguousPctAmb'] / 100)  # ambiguity level

        df['outcome'] = 2 * (df['IsaShockOut'] - 0.5)  # rescaled outcome: -1 = no shock, +1 = shock
        df['outcome_demeaned'] = df['IsaShockOut'] - np.mean(df['IsaShockOut'])  # added 10/20/25

        # --------------------------------------------------------------------
        # Cue type labels (for fine-grained cueType design matrix)
        # --------------------------------------------------------------------
        prob_list, amb_list = [], []
        for prob, amb in df[['ThreatPct', 'AmbiguousPct']].values:
            if amb > 0:
                prob_str = f"p{int(prob + (amb / 2)):03d}_amb"
                amb_str = f"a{int(amb):03d}"
            else:
                prob_str = f"p{int(prob):03d}"
                amb_str = ''
            prob_list.append(prob_str)
            amb_list.append(amb_str)

        df['cue_prob'] = prob_list
        df['cue_amb'] = amb_list
        df['cue_type'] = [f"{p}_{a}" for p, a in zip(df['cue_prob'], df['cue_amb'])]

        # --------------------------------------------------------------------
        # Low vs. None contrast
        # --------------------------------------------------------------------
        low_vs_none_dict = {p: 0 for p in df['ThreatPct']}
        low_vs_none_dict[0] = -1
        low_vs_none_dict[10] = 1
        low_vs_none_dict[20] = 1
        df['low_vs_none'] = df['ThreatPct'].replace(low_vs_none_dict).copy()
        df['low_and_none'] = df['low_vs_none'].replace({-1: 1, 1: 1}).copy()
        df['NOT_low_and_none'] = df['low_and_none'].replace({1: 0, 0: 1}).copy()
        df['low_vs_none'] = compute_demean_range01(df['low_vs_none'])

        # --------------------------------------------------------------------
        # Predefined specific cue-probability comparisons
        # --------------------------------------------------------------------
        # --------------------------------------------------------------------
        # Trial-type contrasts (demeaned for use as parametric regressors)
        # --------------------------------------------------------------------
        df['UncertainThreat_vs_CertainThreat'] = compute_demean_range01(
             df['TrialTypeCode'].replace({1: np.nan, 2: 1, 3: 1, 4: -1})
        )
        df['CertainThreat_vs_CertainSafe'] = compute_demean_range01(
             df['TrialTypeCode'].replace({1: -1, 2: np.nan, 3: np.nan, 4: 1})
        )
        df['Amb_vs_Unamb'] = compute_demean_range01(
            df['TrialTypeCode'].replace({1: np.nan, 2: -1, 3: 1, 4: np.nan})
        )

        # --------------------------------------------------------------------
        # Dummy regressors (binary masks for trial-type groupings)
        # --------------------------------------------------------------------
        df['UncertainThreat_and_CertainThreat'] = df['TrialTypeCode'].replace({1: 0, 2: 1, 3: 1, 4: 1})
        df['NOT_UncertainThreat_and_CertainThreat'] = df['TrialTypeCode'].replace({1: 1, 2: 0, 3: 0, 4: 0})
        df['CertainThreat_and_CertainSafe'] = df['TrialTypeCode'].replace({1: 1, 2: 0, 3: 0, 4: 1})
        df['NOT_CertainThreat_and_CertainSafe'] = df['TrialTypeCode'].replace({1: 0, 2: 1, 3: 1, 4: 0})
        df['Amb_and_Unamb'] = df['TrialTypeCode'].replace({1: 0, 2: 1, 3: 1, 4: 0})
        df['NOT_Amb_and_Unamb'] = df['TrialTypeCode'].replace({1: 1, 2: 0, 3: 0, 4: 1})

        comparisons = {
            'c102030vs0': {'low': ['000'], 'high': ['010', '020', '030']},
            'c405060vs102030': {'low': ['010', '020', '030'], 'high': ['040', '050', '060']},
            'c708090vs405060': {'low': ['040', '050', '060'], 'high': ['070', '080', '090']},
            'c100vs708090': {'low': ['070', '080', '090'], 'high': ['100']},
            'c10vs0': {'low': ['000'], 'high': ['010']},
            'c50vs0': {'low': ['000'], 'high': ['050']}
        }

        for c in comparisons:
            comp_dict = {p: 0 for p in df['ThreatPct']}
            for p_str in comparisons[c]['low']:
                comp_dict[int(p_str)] = -1
            for p_str in comparisons[c]['high']:
                comp_dict[int(p_str)] = 1
            df[c] = df['ThreatPct'].replace(comp_dict).copy()
            df[c + '_AND_const'] = df[c].replace({-1: 1, 1: 1}).copy()
            df[c + '_NOT_const'] = df[c + '_AND_const'].replace({1: 0, 0: 1}).copy()
            df[c] = compute_demean_range01(df[c])

        # Final combined DataFrame for analysis
        combined_df = df.copy()

        # --------------------------------------------------------------------
        # Build first-level design matrices across runs
        # --------------------------------------------------------------------
        dm_list = []

        for run in np.arange(1, 5):  
            df_run = combined_df[combined_df['run'] == run].copy()
            sampling_freq = 1 * UPSAMPLE_FROM_SECS

            # ----------------------------------------------------------------
            # Select design matrix type based on user-specified flag
            # ----------------------------------------------------------------
            if DESIGN_MATRIX == 'parametric_modulation':
                dm_array, dm_columns_dict = create_parametric_modulation_design_matrix(df_run, scan_length_secs, 1, UPSAMPLE_FROM_SECS)
            elif DESIGN_MATRIX == 'cueType':
                dm_array, dm_columns_dict = create_cueType_design_matrix(df_run, scan_length_secs, 1, UPSAMPLE_FROM_SECS)
            elif DESIGN_MATRIX == 'UncertainThreat_vs_CertainThreat':
                dm_array, dm_columns_dict = create_UncertainThreat_vs_CertainThreat_design_matrix(df_run, scan_length_secs, 1, UPSAMPLE_FROM_SECS)
            elif DESIGN_MATRIX == 'CertainThreat_vs_CertainSafe':
                dm_array, dm_columns_dict = create_CertainThreat_vs_CertainSafe_design_matrix(df_run, scan_length_secs, 1, UPSAMPLE_FROM_SECS)
            elif DESIGN_MATRIX == 'Amb_vs_Risk':
                dm_array, dm_columns_dict = create_Amb_vs_Risk_design_matrix(df_run, scan_length_secs, 1, UPSAMPLE_FROM_SECS)
            elif DESIGN_MATRIX == 'low_vs_none':
                dm_array, dm_columns_dict = create_low_vs_none_design_matrix(df_run, scan_length_secs, 1, UPSAMPLE_FROM_SECS)
            elif DESIGN_MATRIX in ['c102030vs0', 'c405060vs102030', 'c708090vs405060', 'c100vs708090', 'c10vs0', 'c50vs0']:
                dm_array, dm_columns_dict = create_comparison_design_matrix(df_run, DESIGN_MATRIX, scan_length_secs, 1, UPSAMPLE_FROM_SECS)
            else:
                raise ValueError(f"{DESIGN_MATRIX} - design matrix has not been implemented.")

            # Convert to Design_Matrix object
            dm = Design_Matrix(dm_array.T, columns=dm_columns_dict, sampling_freq=sampling_freq)

            # ----------------------------------------------------------------
            # Convolution with HRF and temporal derivative
            # ----------------------------------------------------------------
            # Separate columns for HRF-only vs HRF + derivative convolution
            dm_columns_to_conv = [col for col in dm_columns_dict if 'outcome_const' not in col]
            dm_columns_to_conv_timeDerivative = [col for col in dm_columns_dict if 'outcome_const' in col]

            # Downsample before convolution to avoid NaNs in derivative
            dm = dm.downsample(1 / TR)

            # Create HRF kernel with time derivative
            glover_hrf = hrf.glover_hrf(tr=TR, oversampling=1.0)
            glover_derivative = hrf.glover_time_derivative(tr=TR, oversampling=1.0)
            glover_with_derivative = np.vstack([glover_hrf, glover_derivative]).T

            # Convolve regressors
            dm_conv = dm.convolve(columns=dm_columns_to_conv)
            dm_conv = dm_conv.convolve(conv_func=glover_with_derivative, columns=dm_columns_to_conv_timeDerivative)

            # Diagnostic printout
            print("After convolution:")
            print(dm_conv.shape)
            print(dm_conv.columns)

            # ----------------------------------------------------------------
            # High-pass filtering: polynomial + DCT basis
            # ----------------------------------------------------------------
            dm_conv_filt = dm_conv.add_poly(order=3, include_lower=True)
            dm_conv_filt = dm_conv_filt.add_dct_basis(duration=128)

            dm_list.append(dm_conv_filt)

        # --------------------------------------------------------------------
        # Concatenate design matrices across runs
        # --------------------------------------------------------------------
        for i, dm in enumerate(dm_list):
            if i == 0:
                dm_conv_filt = dm
            else:
                dm_conv_filt = dm_conv_filt.append(dm, axis=0)

        # --------------------------------------------------------------------
        # Add nuisance regressors (motion, etc.) across runs
        # --------------------------------------------------------------------
        dm_nuisance_list = []
        for run in np.arange(1, 5):
            nuisance = pd.read_csv(nuisance_files_dict[run], header=None)
            nuisance.columns = [f"nuisance_{col}" for col in nuisance.columns]
            dm_nuisance_run = Design_Matrix(nuisance, sampling_freq=1 / TR)

            if run == 1:
                dm_nuisance = dm_nuisance_run
            else:
                dm_nuisance = dm_nuisance.append(dm_nuisance_run, axis=0)

        # Combine task and nuisance regressors
        dm_conv_filt_nuisance = pd.concat([dm_conv_filt, dm_nuisance], axis=1)
        print("Final design matrix shape:", dm_conv_filt_nuisance.shape)

        # --------------------------------------------------------------------
        # Apply frame censoring across runs
        # --------------------------------------------------------------------
        frames_to_keep = []

        for run in np.arange(1, 5):
            censor = pd.read_csv(censor_files_dict[run], header=None)

            # A frame is kept if its censor row sums to 0 (i.e., no flagged volumes)
            not_censor = (censor.sum(axis=1) == 0).to_list()
            frames_to_keep.extend(not_censor)

        # Drop censored frames from the design matrix
        dm_conv_filt_nuisance_drop = dm_conv_filt_nuisance[frames_to_keep]

        # --------------------------------------------------------------------
        # Save the final design matrix
        # --------------------------------------------------------------------
        save_dir = os.path.join(OUTPUT_ROOT, subj_id)
        design_matrix_final = dm_conv_filt_nuisance_drop.copy()

        # Ensure output directory exists
        os.makedirs(save_dir, exist_ok=True)

        # Debugging checkpoint
        print("Final design matrix columns:")
        print(design_matrix_final.columns)

        # Save design matrix to CSV
        output_file = os.path.join(save_dir, 'design_matrix.csv')
        design_matrix_final.to_csv(output_file, index=False)

        print(f"Design matrix saved for subject {subj_id} at: {output_file}")

        # --------------------------------------------------------------------
        # Concatenate fMRI runs and apply censoring
        # --------------------------------------------------------------------
        fmri_list = []
        print("fMRI files by run:", fmri_files_dict)

        for run in np.arange(1, 5):
            print(f"Appending run {run}")
            fmri_list.append(fmri_files_dict[run])

        # Concatenate fMRI data across runs (time dimension = axis=3)
        full_fmri = nib.funcs.concat_images(fmri_list, check_affines=True, axis=3)

        # Apply censoring to remove bad frames
        keep_fmri = nib.Nifti1Image(
            full_fmri.get_fdata()[:, :, :, frames_to_keep],
            full_fmri.affine,
            full_fmri.header
        )

        fmri = Brain_Data(keep_fmri)

        # --------------------------------------------------------------------
        # Attach final design matrix to fMRI object and run GLM
        # --------------------------------------------------------------------
        fmri.X = dm_conv_filt_nuisance_drop
        stats = fmri.regress()  # Regression step (GLM fitting)

        # --------------------------------------------------------------------
        # Save beta maps for each regressor of interest
        # --------------------------------------------------------------------
        newpath = os.path.join(save_dir, OUTPUT_DIR)
        os.makedirs(newpath, exist_ok=True)

        for i, col in enumerate(dm_conv_filt_nuisance_drop.filter(regex='_c0')):
            # Debugging: confirm regressor name matches column
            print(f"Saving beta map for regressor: {col}")
            outfile = os.path.join(newpath, f"{col}.nii.gz")
            print(f"Output file: {outfile}")

            stats['beta'][i].write(outfile)

# --------------------------------------------------------------------
# Parse command-line arguments
# --------------------------------------------------------------------
DESIGN_MATRIX = sys.argv[1]  # e.g., 'cueType', 'c10vs0'
OUTPUT_DIR = sys.argv[2]     # directory name where betas will be saved
NUISANCE_STRING = sys.argv[3]  # e.g., 'FullNuisanceVariableSet.csv'

# --------------------------------------------------------------------
# Run analysis for a single subject
# --------------------------------------------------------------------
run_first_level_glm_single_subject(
    DESIGN_MATRIX,
    OUTPUT_DIR,
    NUISANCE_STRING
)

# --------------------------------------------------------------------
# Example arguments (for reference/debugging)
# --------------------------------------------------------------------
# DESIGN_MATRIX = 'full'
# OUTPUT_DIR = 'beta_maps_pAmbIncludesAmb_62625'
# NUISANCE_STRING = 'FullNuisanceVariableSet.csv'
#
# Alternatives:
# DESIGN_MATRIX = 'cueType'
# DESIGN_MATRIX = 'UncertainThreat_vs_CertainThreat'
# DESIGN_MATRIX = 'CertainThreat_vs_CertainSafe'
# DESIGN_MATRIX = 'Amb_vs_Risk'
