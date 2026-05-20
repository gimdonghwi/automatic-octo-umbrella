# DAIC-WOZ Preprocessing

Preprocessing pipeline for turning raw DAIC-WOZ interviews into utterance-level sequence data for downstream modeling.

## What is here

- `preprocessing/transcript_parser.py`
  - Reads participant transcript files and normalizes question-related metadata.
- `preprocessing/utterance_segmenter.py`
  - Merges or splits utterances into training-friendly segments.
- `preprocessing/audio_processor.py`
  - Loads interview audio and extracts acoustic features.
- `preprocessing/dataset_builder.py`
  - Orchestrates full dataset construction.
- `config/preprocessing_config.yaml`
  - Data and processing settings.
- `notebooks/01_run_preprocessing.ipynb`
  - Interactive preprocessing workflow.

## Expected Output

The pipeline builds a participant-indexed sequence dataset containing:

- acoustic embeddings
- utterance text
- question-type metadata
- utterance timing

## Quick Start

```bash
cd daic_preprocessing
pip install -r requirements.txt
jupyter notebook notebooks/01_run_preprocessing.ipynb
```

## Notes

- Local dataset paths are resolved from config and environment variables.
- Generated pickle files are not intended to be committed to the repository.
