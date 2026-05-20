# daic_experiments — Experimental Journey

This folder documents the full trial-and-error process that led to the final multimodal depression detection model in `daic_training/`.

Each subfolder represents a distinct research direction, in roughly chronological order. Results are summarised in [`results/ablation_summary.md`](results/ablation_summary.md).

---

## Why this folder exists

The final model did not appear fully formed.
It was preceded by failed attempts, collapsed training runs, surprising findings about which features actually matter, and incremental design decisions.
This folder preserves that process so the reasoning behind every architectural choice is traceable.

---

## Folder Structure

```
daic_experiments/
├── 01_eda_and_traditional_ml/      # Where everything started
├── 02_audio_only_wavlm/            # First deep learning baseline
├── 03_audio_only_hubert/           # Alternative audio encoder
├── 04_text_only_attempt/           # Text-only: an instructive failure
├── 05_multimodal_simple_concat/    # Fusion without attention
└── results/                        # Aggregated results table
```

---

## Experiment Log

### Stage 1 — EDA & Traditional ML  (`01_eda_and_traditional_ml/`)

**Starting question:** Can we detect depression from interview features at all?

- `01_eda_and_hypothesis.ipynb` — Initial data exploration. Formulated three hypotheses:
  1. Depressed participants use less lexically diverse language (lower TTR).
  2. Negative vocabulary frequency is higher in depressed responses.
  3. Prosodic features (pitch, energy) differ between groups.

  Excluded participants 451, 458, 480 (corrupted audio / missing transcripts).

- `02_traditional_ml_baselines.ipynb` — Tested handcrafted feature pipelines with classical ML (Random Forest, SVM) using 15 linguistic + acoustic features and 5-fold CV.

  **Key finding:** Random Forest achieved \~70% accuracy but only 25% F1 on the depression class — the model simply predicted "normal" for most samples. Class imbalance (70/30 split) was the main bottleneck.

  **Decision:** Move to deep learning to learn feature representations rather than engineering them manually.

---

### Stage 2 — Audio-Only: WavLM  (`02_audio_only_wavlm/`)

**Question:** How well does a pre-trained audio encoder alone perform?

- `wavlm_ttr_transformer.ipynb` — WavLM embeddings (768-d) + enhanced TTR features fed into a Transformer classifier. Optuna-tuned over balanced F1.

  **Test results:**
  - Accuracy: 67.4%
  - F1 (depression): 59.5%
  - Balanced F1: 65.4%

  WavLM clearly outperformed Random Forest on the depression class, validating the deep feature extraction approach. But 32.6% of the test set was still misclassified.

- `wavlm_variants_exploration.ipynb` — Explored different Transformer depths, loss functions (focal loss, label smoothing), and threshold strategies to improve sensitivity.

  **Key finding:** Optimal threshold via validation search (not fixed 0.5) significantly improved recall. Late-fusion attempts between audio and text models showed that text predictions were too unreliable to contribute positively.

---

### Stage 3 — Audio-Only: HuBERT  (`03_audio_only_hubert/`)

**Question:** Does a different self-supervised audio model (HuBERT) perform better?

- `hubert_baseline.ipynb` — Same Transformer architecture, HuBERT (768-d) replacing WavLM.

  **Test results:**
  - Accuracy: 58.7%
  - F1 (depression): 24.0%
  - Balanced F1: 36.0%

  **Key finding:** HuBERT showed strong validation performance (balanced F1 \~0.75) but collapsed on the test set — the model learned patterns specific to the validation split. The very narrow prediction probability range (0.438–0.456) suggested the model was not actually discriminating. WavLM was more stable.

  **Decision:** Keep WavLM as the primary audio encoder. HuBERT dropped.

---

### Stage 4 — Text-Only: MentalBERT  (`04_text_only_attempt/`)

**Question:** Can transcript text alone detect depression?

- `mentalbert_text_only.ipynb` — MentalBERT (domain-adapted BERT for mental health) on interview transcripts. Trained to predict binary labels from participant utterance sequences.

  **Result:** Complete training failure. The model converged to predicting the majority class (non-depressed) for every sample. F1 (depression) = 0.0 across all validation epochs.

  Root cause investigation:
  - The participant transcript segments are very short (often 1–5 words).
  - MentalBERT expects sentence-level context; utterance-level inputs are too sparse.
  - Class imbalance amplified the collapse.

  Late-fusion between saved audio probabilities and text probabilities also failed — the text model's output variance was essentially zero (std ≈ 0.005).

  **Key finding:** Raw transcript text at the utterance level does not carry enough signal for a standalone classifier. However, derived features (TTR, lexical density, question-type context) proved useful in the final model — the signal is in the *structure* of speech, not the token embeddings.

  **Decision:** Drop direct text encoder. Use engineered linguistic features (TTR + Q-type embedding) instead.

---

### Stage 5 — Multimodal Simple Concat  (`05_multimodal_simple_concat/`)

**Question:** Does naively concatenating audio + linguistic features help?

- `multimodal_feasibility_check.ipynb` — Verified that WavLM and TTR/Q-type features can be aligned at the utterance level. Confirmed data pipeline feasibility.

- `concat_fusion_kfold.ipynb` — Simple concatenation of [WavLM ‖ TTR ‖ Q-type] → Transformer, trained with 5-fold CV and ensemble inference.

  **Test results (ensemble average):**
  - Accuracy: 59%
  - F1 (depression): 49%
  - Balanced F1: 68.2%

  Balanced F1 improved over WavLM-only (65.4% → 68.2%), but overall accuracy dropped. The simple concatenation gave equal weight to all features regardless of their relevance per utterance.

  **Key finding:** Multimodal fusion helps, but naive concatenation is not sufficient. The model needs to learn *which modality to trust* for each utterance.

  **Decision:** Design an attention-based modality fusion mechanism (→ `ModalityFusion` in the final model).

---

## From Experiments to Final Model

The experimental findings directly shaped every architectural decision in `daic_training/`:

| Finding | Architectural Response |
|---|---|
| WavLM > HuBERT (stability) | WavLM as primary encoder |
| Wav2Vec adds complementary signal | Gated secondary encoder (`GatedFusion`) |
| Text tokens fail; TTR/Q-type work | `LinguisticEncoder` with TTR + Q-type embedding |
| Naive concat gives equal weight | Attention-based `ModalityFusion` |
| Per-utterance relevance varies | `UtteranceAttentionPooling` (also enables XAI) |
| PHQ regression + threshold > binary | Regression head + threshold-at-test-time |
| Class imbalance causes F1 collapse | Weighted Huber loss + balanced F1 as primary metric |

---

## Quantitative Summary

See [`results/ablation_summary.md`](results/ablation_summary.md) for the full metrics table.
