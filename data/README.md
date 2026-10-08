# Bundled data

Small, derived data files that let every notebook in [`figure_notebooks/`](../figure_notebooks/)
run on its own. Notebooks read them via relative paths (`../data/...`), so run the notebooks from
inside `figure_notebooks/`.

Sample sizes: behavioral ratings N = 104, skin conductance (SCR) N = 102, fMRI N = 101
(see the manuscript's Methods for exclusions). Subject IDs (`RDCxxx`) are pseudonymous study IDs.

**Provenance** says which script in [`scripts/`](../scripts/) produced each file. Script outputs
were renamed when they were bundled here. "Not produced by code in this repository" means the
file came from an upstream step whose code is not included here.

## `figure1/` — Figure 1 (also used by Supplementary Figure 1)

| File | Contents | Provenance | Used by |
|------|----------|------------|---------|
| `behavioral_data.csv` | Trial-level task data (14,110 trials, 104 subjects): cue Probability/Risk/Ambiguity, outcome, distress rating (0–6), and single-trial SCR estimates (`scr`, plus within-subject z-scored `scr_zscored`) | Not produced by code in this repository | `Figure1`, `SuppFig1` |
| `signature_expression_by_regressor.csv` | Per-subject pattern-expression scores (dot product of first-level beta map and signature weights) of the Probability (`p_risk`), Risk (`r_risk`) and Ambiguity (`a_amb`) maps for the negative-affect (`Ceko_NE`) and anticipatory-anxiety (`LiuBecker_Amb`) signatures; 101 subjects | Not produced by code in this repository | `Figure1`, `SuppFig1` |
| `wholebrain_timecourse_subject.csv` | Per-subject timepoint-by-timepoint regression betas (Probability, Risk, Ambiguity, rational probability, trial type, outcome) for whole-brain signature expression, cue- and outcome-aligned; 101 subjects | [`timecourse_analysis_signature_expression.py`](../scripts/timecourse/timecourse_analysis_signature_expression.py) → [`averages_timecourse_whole_brain_signature.py`](../scripts/timecourse/averages_timecourse_whole_brain_signature.py) (subject-level output) | `Figure1` (panel l) |
| `wholebrain_timecourse_group.csv` | Group mean and SEM of the above at each timepoint | [`averages_timecourse_whole_brain_signature.py`](../scripts/timecourse/averages_timecourse_whole_brain_signature.py) (group-level output) | `Figure1` (panel l) |

## `figure2/` — Figure 2

| File | Contents | Provenance | Used by |
|------|----------|------------|---------|
| `ce_bst_betas_long_fullNoPE.csv` | Mean Ce and BST betas for Probability, Risk and Ambiguity (parametric first-level model); 101 subjects | [`extract_ce_bst_mean_betas.py`](../scripts/roi_extraction/extract_ce_bst_mean_betas.py) | `Figure2` (panel a) |
| `ce_bst_betas_long_cueType.csv` | Mean Ce and BST betas for each discrete cue level (`pNNN_c0` = unambiguous probability level, `pNNN_amb_c0` = rational probability on ambiguous trials, `aNNN_c0` = ambiguity level, plus cue/outcome/rating regressors); 101 subjects | [`extract_ce_bst_mean_betas_cue_based.py`](../scripts/roi_extraction/extract_ce_bst_mean_betas_cue_based.py) | `Figure2` (panels c–d) |
| `roi_ce.nii`, `roi_bst.nii` | Bilateral Ce and BST ROI masks (2-mm MNI grid); see Methods, *EA ROIs* | Input masks, not produced by code in this repository | `Figure2` |
| `template_T1.nii.gz` | 1-mm MNI-space mean T1 template used as the background for brain figures | Not produced by code in this repository | `Figure2`, `SuppBrainFigs_plot_brain_slices` |

## `figure3/` — Figure 3

| File | Contents | Provenance | Used by |
|------|----------|------------|---------|
| `ce_bst_betas_long_c50vs0.csv` | Mean Ce and BST betas for the 50% vs 0% threat-probability contrast; 101 subjects | [`extract_ce_bst_mean_betas.py`](../scripts/roi_extraction/extract_ce_bst_mean_betas.py) | `Figure3` (panel b) |
| `ce_bst_betas_long_cueType.csv` | Identical copy of `figure2/ce_bst_betas_long_cueType.csv` | See `figure2/` | `Figure3` (panels h–j) |
| `contrasts_subject_level.csv` | Mean Ce and BST betas for the four adjacent probability-bin contrasts; 101 subjects | [`extract_ce_bst_mean_betas_for_contrasts.py`](../scripts/roi_extraction/extract_ce_bst_mean_betas_for_contrasts.py) (subject-level output) | `Figure3` (panel k) |
| `contrasts_group_summary.csv` | Group mean, SEM and confidence intervals of the above | [`extract_ce_bst_mean_betas_for_contrasts.py`](../scripts/roi_extraction/extract_ce_bst_mean_betas_for_contrasts.py) (group-level output) | `Figure3` (panel k) |
| `popt_df_seed51.csv` | Fitted sigmoid slope (`k`) and midpoint (`x0`) for each of 1,000 bootstrap iterations per region | Bootstrap cell of [`Figure3.ipynb`](../figure_notebooks/Figure3.ipynb) (seed 51) | `Figure3` (panels h–i) |
| `smooth_df_seed51.csv` | Fitted sigmoid curves (200 points) for each bootstrap iteration and region | Bootstrap cell of [`Figure3.ipynb`](../figure_notebooks/Figure3.ipynb) (seed 51) | `Figure3` (panel j) |

`Figure3.ipynb` recomputes the bootstrap from `ce_bst_betas_long_cueType.csv` and reports the
statistics from that fresh run. It then reloads the two `*_seed51.csv` files for plotting, so the
figure does not depend on `curve_fit` converging identically across platforms. With the versions
in [`environment.yml`](../environment.yml), the recomputed fits match the bundled files to
floating-point precision.

## `figure4/` — Figure 4 and Supplementary Figure 7 (non-human primates)

Behavior-coding spreadsheets for the BST-lesion experiment. Each sheet holds one behavior (e.g.
freezing, locomotion, vocalizations). Rows are animals (`SUBJ`), with cage-mate pair (`PAIR`) and
group (`GRP`: `EXP` = BST lesion, `CON` = control). Columns are 5-min bins. None of these files
was produced by code in this repository. All are read by
[`Figure4_and_SuppFig7(NHP).ipynb`](../figure_notebooks/Figure4_and_SuppFig7(NHP).ipynb).

| File | Paradigm | Animals in file |
|------|----------|-----------------|
| `nec30aln30_t1.xls` | HIP-Persistence, first exposure: 30-min no-eye-contact (NEC1–6) then 30-min alone (A1–6) | 6 lesion, 6 control |
| `nec30aln30_t2.xls` | HIP-Persistence, second exposure | 5 lesion, 5 control |
| `novel.xls` | Social interaction with a novel conspecific (S1–6) | 7 lesion, 7 control |
| `upet.xls` | NEC-only assessment before (`PRE`) and after (`POST1`) lesion | 7 lesion, 7 control |

## `suppfig1/` — Supplementary Figure 1

| File | Contents | Provenance | Used by |
|------|----------|------------|---------|
| `signature_expression_categorical_contrasts.csv` | Per-subject pattern-expression scores of the Certain Threat vs Certain Safe, Uncertain Threat vs Certain Threat, and Ambiguous vs Unambiguous (`Amb_vs_Risk`) contrast maps for the two signatures; 101 subjects | Not produced by code in this repository | `SuppFig1` (panels d–e) |

## `suppfig5/` — Supplementary Figure 5

| File | Contents | Provenance | Used by |
|------|----------|------------|---------|
| `roi_timecourse_subject.csv` | Per-subject timepoint-by-timepoint regression betas for mean Ce and BST activity, cue- and outcome-aligned; 101 subjects | [`timecourse_analysis_BST_CE_mean_roi.py`](../scripts/timecourse/timecourse_analysis_BST_CE_mean_roi.py) → [`averages_timecourse_CE_BST.py`](../scripts/timecourse/averages_timecourse_CE_BST.py) (subject-level output) | `SuppFig5` |
| `roi_timecourse_group.csv` | Group mean and SEM of the above at each timepoint | [`averages_timecourse_CE_BST.py`](../scripts/timecourse/averages_timecourse_CE_BST.py) (group-level output) | `SuppFig5` |

## `suppbrainfigs/` — thresholded whole-brain maps for the supplementary brain-slice figures

Group-level t-maps thresholded at FDR q < 0.005, positive effects only, resampled to the 1-mm
template grid (nearest neighbor). Voxels that do not survive the threshold are 0. All six are
outputs of [`run_second_level_full_pipeline.py`](../scripts/second_level/run_second_level_full_pipeline.py)
and are read by
[`SuppBrainFigs_plot_brain_slices.ipynb`](../figure_notebooks/SuppBrainFigs_plot_brain_slices.ipynb).
Probability, Risk and Ambiguity share one FDR family (see Methods).

| File | Contrast |
|------|----------|
| `probability_FDR0.005_posOnly.nii.gz` | Parametric Probability |
| `risk_FDR0.005_posOnly.nii.gz` | Parametric Risk |
| `ambiguity_FDR0.005_posOnly.nii.gz` | Parametric Ambiguity |
| `certainThreat_vs_certainSafe_FDR0.005_posOnly.nii.gz` | Certain Threat vs Certain Safe |
| `uncertainThreat_vs_certainThreat_FDR0.005_posOnly.nii.gz` | Uncertain Threat vs Certain Threat |
| `c50vs0_FDR0.005_posOnly.nii.gz` | 50% vs 0% threat probability |

Neuroimaging maps are also available on NeuroVault (see the manuscript's Resource Sharing
section).
