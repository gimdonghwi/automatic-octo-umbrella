# DAIC-WOZ Workspace

This repository contains a set of small, organized projects built around the DAIC-WOZ interview dataset.

## Projects

- `daic_preprocessing`
  - Transcript parsing, utterance segmentation, audio feature extraction, and sequence dataset building.
- `daic_training`
  - Training code for PHQ score regression and threshold-based depression classification experiments.
- `daic_explainability`
  - Utilities for inspecting attention, modality contribution, and case-level model behavior.
- `daic_woz_analysis`
  - Exploratory analysis, profiling, and question-type analysis notebooks and modules.
- `daic_simulator`
  - Interview prototype and simulator work. This folder is more experimental than the others.

## Notes

- Large datasets and generated artifacts are intentionally not part of the cleaned codebase.
- Paths are driven by YAML config files and local environment setup.
- The repository is best understood as a research and prototyping workspace rather than a finished product or packaged library.

## Question Type Workflow

The question-type pipeline was built in a fairly practical way rather than from a single end-to-end classifier.

1. Ellie utterances were first collected from the transcript data.
2. A zero-shot classifier was used to separate likely `question` vs `statement` utterances.
3. Question utterances were merged and deduplicated into a smaller working set.
4. That reduced question set was then manually grouped into broader categories used later in preprocessing and analysis.

For reproducibility, a cleaned version of the zero-shot screening step is included at:
`daic_preprocessing/preprocessing/zero_shot_question_filter.py`

This worked reasonably well because Ellie tends to repeat or rephrase many prompts across interviews, so the unique question set was much smaller than the raw transcript volume.

At the same time, the final question-type grouping still contains subjective judgment. Some category boundaries were assigned manually for practical modeling purposes, so the taxonomy should be understood as a useful working simplification rather than a definitive annotation standard.

## Limitations

- Results are influenced by the interview structure and question-type distribution present in DAIC-WOZ, not just participant content alone.
- Some learned signals may partly reflect short, low-elaboration response patterns often associated with depressive speech, but those cues are not specific enough to support clinical claims by themselves.
- DAIC-WOZ is a relatively small research dataset, so model performance and observed patterns should be interpreted as exploratory results rather than broadly generalizable conclusions.
