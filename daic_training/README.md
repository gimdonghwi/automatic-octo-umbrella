# DAIC-WOZ Training

Training code for multimodal depression prediction using sequence features derived from DAIC-WOZ interviews.

## What is here

- `config/training_config.yaml`
  - Model and optimization settings.
- `models/`
  - Transformer model, fusion components, and loss functions.
- `training/core.py`
  - Dataset loading, training loop, and evaluation utilities.
- `notebooks/02_train_model_notebook.ipynb`
  - Interactive training workflow.
- `notebooks/02_train_model.py`
  - Script-style notebook export.

## Expected Inputs

The training code expects a preprocessed sequence dataset and metadata configured through `config/training_config.yaml`.

## Quick Start

```bash
cd daic_training
pip install -r requirements.txt
jupyter notebook notebooks/02_train_model_notebook.ipynb
```

## Notes

- Generated checkpoints and experiment outputs are intentionally not committed.
- This folder is organized as research infrastructure rather than a packaged training library.
