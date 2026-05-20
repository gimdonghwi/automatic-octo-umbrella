# DAIC-WOZ Explainability

Explainability utilities for inspecting trained depression prediction models on DAIC-WOZ-derived features.

## What is here

- `config/explainability_config.yaml`
  - Analysis and visualization settings.
- `explainability/explainer.py`
  - Core extraction logic for attention and modality signals.
- `explainability/visualizations.py`
  - Plotting utilities.
- `explainability/report_generator.py`
  - Text and summary report helpers.

## Quick Start

```bash
cd daic_explainability
pip install -r requirements.txt
```

Then run the notebook or import the modules directly for programmatic analysis.

## Notes

- Generated reports and figures are local artifacts and should not be committed.
- This folder is intended for model inspection after preprocessing and training are already complete.
