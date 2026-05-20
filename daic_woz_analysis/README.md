# DAIC-WOZ Analysis

Exploratory data analysis utilities and notebooks for profiling DAIC-WOZ-derived datasets.

## What is here

- `eda/data_loader.py`
  - Dataset loading and validation helpers.
- `eda/statistical_analyzer.py`
  - Statistical comparison utilities.
- `eda/feature_profiler.py`
  - Feature extraction and summary profiling.
- `eda/visualizations.py`
  - Plotting functions for analysis notebooks.
- `config/paths.yaml`
  - Local path configuration.
- `notebooks/`
  - Narrative EDA workflows.

## Question Type Taxonomy

The question-type analysis uses a simplified taxonomy for downstream profiling and modeling. The original question labels were relatively fine-grained, but keeping them at that level introduced extra complexity and could create imbalanced category distributions, with some types appearing much less often than others.

To keep the feature more stable, the detailed labels are consolidated into five broader categories: `casual`, `background`, `emotional`, `clinical`, and `other`. This reduces sparsity while preserving the main functional differences between interview prompts.

## Quick Start

```bash
cd daic_woz_analysis
pip install -r requirements.txt
jupyter notebook notebooks/01_data_profiling.ipynb
```

## Notes

- Reports and figures are generated artifacts and should be ignored in Git.
- This folder is intended for inspection and analysis, not as a packaged Python library.
