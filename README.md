# MDART Paper — Analysis Code

This repository contains the analysis code for a study of how the brain represents
distinct facets of threat uncertainty — probability, risk (known-probability variance),
and ambiguity (unknown probability) — during anticipation of aversive outcomes. In the
human dataset (the MDART task), we combine fMRI, skin-conductance, and behavioral
ratings to characterize univariate, region-of-interest (central amygdala, BST), and
whole-brain neural-signature (e.g., negative affect, anticipatory anxiety) responses to
certain and uncertain threat. A complementary non-human primate dataset uses bilateral
BST lesions to test the region's causal role in threat persistence, social behavior, and
locomotion.

## Contents

- `scripts/` — analysis scripts, organized by pipeline stage
- `figure_notebooks/` — one notebook per figure/table, reproducing the reported results from
  intermediate outputs. Run them from inside `figure_notebooks/`; figures are saved to
  `outputs/figures/` (created automatically, not tracked by git)
- `data/` — small, figure-specific input files bundled so each notebook runs standalone
  (one subfolder per figure; see [`data/README.md`](data/README.md) for what each file is and
  where it originally came from in the pipeline)

## Analysis pipeline


| Stage | Script | Produces | Depends on |
|-------|--------|----------|------------|
| First-level GLM | [`scripts/first_level/first_level_fMRI_GLMs.py`](scripts/first_level/first_level_fMRI_GLMs.py) | Per-subject design matrix (`design_matrix.csv`) and per-regressor beta maps (`.nii.gz`) | Preprocessed fMRI + behavioral event files|
| Second-level GLM | [`scripts/second_level/run_second_level_full_pipeline.py`](scripts/second_level/run_second_level_full_pipeline.py) | Uncorrected + FDR-corrected (signed/pos-only/neg-only) group t-maps, 1mm-resampled versions, and `fdr_log.csv` | First-level GLM |
| ROI beta extraction (named regressors) | [`scripts/roi_extraction/extract_ce_bst_mean_betas.py`](scripts/roi_extraction/extract_ce_bst_mean_betas.py) | Ce/BST mean betas, long + wide CSV, one file pair per first-level GLM subdirectory (Probability/Risk/Ambiguity share one file since they come from the same combined GLM; each other contrast gets its own) | First-level GLM |
| ROI beta extraction (probability-bin contrasts) | [`scripts/roi_extraction/extract_ce_bst_mean_betas_for_contrasts.py`](scripts/roi_extraction/extract_ce_bst_mean_betas_for_contrasts.py) | Ce/BST mean betas per subject + group summary (mean/SEM/CI) for 4 adjacent probability-bin contrasts | First-level GLM |
| ROI beta extraction (cueType, per-level) | [`scripts/roi_extraction/extract_ce_bst_mean_betas_cue_based.py`](scripts/roi_extraction/extract_ce_bst_mean_betas_cue_based.py) | Ce/BST mean betas, long + wide CSV, one row per (subject, cue-level regressor, region); feeds Figure 2 Panels c-d | First-level GLM (cueType design) |
| Timecourse analysis (ROI) | [`scripts/timecourse/timecourse_analysis_BST_CE_mean_roi.py`](scripts/timecourse/timecourse_analysis_BST_CE_mean_roi.py) | Per-subject outcome-aligned epoch tables and per-timepoint regression betas (combined risky+ambiguous model), for the Ce and BST ROIs | Preprocessed fMRI + behavioral event files directly (independent of the GLM stages above) |
| Timecourse analysis (whole-brain signature expression) | [`scripts/timecourse/timecourse_analysis_signature_expression.py`](scripts/timecourse/timecourse_analysis_signature_expression.py) | Per-subject outcome-aligned epoch table and per-timepoint regression betas (combined risky+ambiguous model), for each whole-brain signature | Preprocessed fMRI + behavioral event files directly (independent of the GLM stages above) |
| Timecourse aggregation (ROI) | [`scripts/timecourse/averages_timecourse_CE_BST.py`](scripts/timecourse/averages_timecourse_CE_BST.py) | Subject-level long file + group-level (mean/SEM across subjects) file of Ce/BST timecourse betas | Timecourse analysis (ROI) |
| Timecourse aggregation (whole-brain signature expression) | [`scripts/timecourse/averages_timecourse_whole_brain_signature.py`](scripts/timecourse/averages_timecourse_whole_brain_signature.py) | Subject-level long file + group-level (mean/SEM across subjects) file of whole-brain signature timecourse betas; feeds Figure 1 Panel l | Timecourse analysis (whole-brain signature expression) |


Shared helpers used by stages 2–4 live in [`scripts/common/roi_utils.py`](scripts/common/roi_utils.py) (beta-map bookkeeping, mask loading, ROI averaging).

## Figures

| Figure | Notebook | Data |
|--------|----------|------|
| Figure 1 | [`figure_notebooks/Figure1.ipynb`](figure_notebooks/Figure1.ipynb) | [`data/figure1/`](data/figure1/) |
| Figure 2 | [`figure_notebooks/Figure2.ipynb`](figure_notebooks/Figure2.ipynb) | [`data/figure2/`](data/figure2/) |
| Figure 3 | [`figure_notebooks/Figure3.ipynb`](figure_notebooks/Figure3.ipynb) | [`data/figure3/`](data/figure3/) |
| Figure 4 (+ Supplementary Figure 7, NHP) | [`figure_notebooks/Figure4_and_SuppFig7(NHP).ipynb`](figure_notebooks/Figure4_and_SuppFig7(NHP).ipynb) | [`data/figure4/`](data/figure4/) |
| Supplementary Figure 1 | [`figure_notebooks/SuppFig1.ipynb`](figure_notebooks/SuppFig1.ipynb) | [`data/suppfig1/`](data/suppfig1/) (+ reuses `data/figure1/`) |
| Supplementary Figure 5 | [`figure_notebooks/SuppFig5.ipynb`](figure_notebooks/SuppFig5.ipynb) | [`data/suppfig5/`](data/suppfig5/) |
| Supplementary brain-slice figures (utility) | [`figure_notebooks/SuppBrainFigs_plot_brain_slices.ipynb`](figure_notebooks/SuppBrainFigs_plot_brain_slices.ipynb) | [`data/suppbrainfigs/`](data/suppbrainfigs/) (+ reuses `data/figure2/template_T1.nii.gz`) |

## Preprocessing

Preprocessing of the raw fMRI data is not included in this repository, yet.

## Requirements

Python 3.11 plus the packages listed in [`environment.yml`](environment.yml) — versions
there are pinned to what every script and notebook in this repository was actually
verified to run against. To create the environment:

```
conda env create -f environment.yml
conda activate mdart-analyses
```

## Data availability

This repository includes the analysis code together with small, derived data files
(`data/`; see [`data/README.md`](data/README.md) for what each file is and which script
produced it) sufficient to reproduce every figure and reported statistic in the
manuscript directly from `figure_notebooks/`. These are subject- and group-level summary
measures — behavioral ratings, extracted ROI/signature betas, timecourses, and
thresholded statistical maps — derived from the raw data, plus the non-human primate
behavioral dataset (`data/figure4/`). The underlying raw/preprocessed neuroimaging and
behavioral data are not included in this repository.


## Citation

See [`CITATION.cff`](CITATION.cff).

## License

MIT — see [LICENSE](LICENSE).
