"""
Core explainability helpers for attention-based depression detection models.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ModelExplainer:
    """
    Extract attention and fusion signals from a trained model.

    Expected model contract:
    - `forward_with_attention(wavlm, wav2vec, ttr, q_types, mask)`
      returns `(prediction, explanation_dict)`
    - `explanation_dict` contains:
      - `utterance_attention`
      - `modality_weights`
      - `audio_gate`
    """

    def __init__(self, model: nn.Module, config: Dict, device: torch.device):
        self.model = model
        self.config = config
        self.device = device

        self.phq_max = config["data"]["phq_max"]
        self.threshold = config["data"]["depression_threshold"]
        self.qtype_names = config["data"]["qtype_names"]

        self.model.eval()

    @staticmethod
    def _to_float(value) -> float:
        """Convert scalar-like torch/numpy/python values to float."""
        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu().item())
        if isinstance(value, np.ndarray):
            return float(value.reshape(-1)[0])
        return float(value)

    def explain_single_sample(
        self,
        wavlm: torch.Tensor,
        wav2vec: torch.Tensor,
        ttr: torch.Tensor,
        q_types: torch.Tensor,
        mask: torch.Tensor,
        texts: List[str],
        true_score: float,
        pid: str,
    ) -> Dict:
        """Generate explainability output for a single participant."""
        with torch.no_grad():
            pred, explanations = self.model.forward_with_attention(wavlm, wav2vec, ttr, q_types, mask)

        pred_phq = self._to_float(pred) * self.phq_max
        actual_len = int(mask.sum().item())

        pred_label = 1 if pred_phq >= self.threshold else 0
        true_label = 1 if true_score >= self.threshold else 0

        return {
            "pid": pid,
            "pred_phq": float(pred_phq),
            "true_phq": float(true_score),
            "pred_label": pred_label,
            "true_label": true_label,
            "correct": pred_label == true_label,
            "utterance_attention": explanations["utterance_attention"].detach().cpu().numpy()[0][:actual_len],
            "modality_weights": explanations["modality_weights"].detach().cpu().numpy()[0][:actual_len],
            "audio_gate": explanations["audio_gate"].detach().cpu().numpy()[0][:actual_len].flatten(),
            "q_types": q_types.detach().cpu().numpy()[0][:actual_len],
            "texts": texts[:actual_len],
            "num_utterances": actual_len,
        }

    def explain_dataset(self, dataset) -> List[Dict]:
        """Generate explainability output for an entire dataset."""
        all_results: List[Dict] = []
        logger.info("Generating explanations for %d participants...", len(dataset))

        for idx in range(len(dataset)):
            sample = dataset[idx]

            wavlm = sample["wavlm"].unsqueeze(0).to(self.device)
            wav2vec = sample["wav2vec"].unsqueeze(0).to(self.device)
            ttr = sample["ttr"].unsqueeze(0).to(self.device)
            q_types = sample["q_types"].unsqueeze(0).to(self.device)
            mask = sample["mask"].unsqueeze(0).to(self.device)

            if "raw_score" in sample:
                true_score = self._to_float(sample["raw_score"])
            else:
                true_score = self._to_float(sample["score"]) * self.phq_max

            result = self.explain_single_sample(
                wavlm,
                wav2vec,
                ttr,
                q_types,
                mask,
                sample["texts"],
                true_score,
                sample["pid"],
            )
            all_results.append(result)

        logger.info("Generated explanations for %d participants", len(all_results))
        return all_results

    def categorize_results(self, results: List[Dict]) -> Dict[str, List[Dict]]:
        """Categorize participant results into TP/TN/FP/FN buckets."""
        categorized = {
            "all": results,
            "correct": [r for r in results if r["correct"]],
            "incorrect": [r for r in results if not r["correct"]],
            "tp": [r for r in results if r["pred_label"] == 1 and r["true_label"] == 1],
            "tn": [r for r in results if r["pred_label"] == 0 and r["true_label"] == 0],
            "fp": [r for r in results if r["pred_label"] == 1 and r["true_label"] == 0],
            "fn": [r for r in results if r["pred_label"] == 0 and r["true_label"] == 1],
        }

        logger.info("Classification breakdown:")
        logger.info("  Correct: %d / %d", len(categorized["correct"]), len(results))
        logger.info("  TP: %d, TN: %d", len(categorized["tp"]), len(categorized["tn"]))
        logger.info("  FP: %d, FN: %d", len(categorized["fp"]), len(categorized["fn"]))
        return categorized

    def select_case_studies(self, categorized: Dict[str, List[Dict]]) -> Dict[str, Optional[Dict]]:
        """Pick representative cases for deeper review."""
        cases: Dict[str, Optional[Dict]] = {}

        cases["best_true_positive"] = (
            min(categorized["tp"], key=lambda x: abs(x["pred_phq"] - x["true_phq"]))
            if categorized["tp"]
            else None
        )
        cases["best_true_negative"] = (
            min(categorized["tn"], key=lambda x: abs(x["pred_phq"] - x["true_phq"]))
            if categorized["tn"]
            else None
        )
        cases["worst_false_positive"] = (
            max(categorized["fp"], key=lambda x: x["pred_phq"] - x["true_phq"])
            if categorized["fp"]
            else None
        )
        cases["worst_false_negative"] = (
            min(categorized["fn"], key=lambda x: x["pred_phq"] - x["true_phq"])
            if categorized["fn"]
            else None
        )
        return cases

    def compute_aggregate_statistics(self, results: List[Dict]) -> Dict:
        """Compute aggregate regression and explainability statistics."""
        if not results:
            return {
                "accuracy": 0.0,
                "mean_error": 0.0,
                "rmse": 0.0,
                "mae": 0.0,
                "qtype_importance": {"normal": {}, "depression": {}},
                "modality_contribution": {
                    "normal": {"audio": {"mean": 0.0, "std": 0.0}, "linguistic": {"mean": 0.0, "std": 0.0}},
                    "depression": {"audio": {"mean": 0.0, "std": 0.0}, "linguistic": {"mean": 0.0, "std": 0.0}},
                },
            }

        stats: Dict = {}
        correct = sum(1 for r in results if r["correct"])
        stats["accuracy"] = correct / len(results)

        errors = [r["pred_phq"] - r["true_phq"] for r in results]
        stats["mean_error"] = float(np.mean(errors))
        stats["rmse"] = float(np.sqrt(np.mean([e ** 2 for e in errors])))
        stats["mae"] = float(np.mean([abs(e) for e in errors]))

        normal_results = [r for r in results if r["true_phq"] < self.threshold]
        dep_results = [r for r in results if r["true_phq"] >= self.threshold]

        stats["qtype_importance"] = {
            "normal": self._aggregate_qtype_attention(normal_results),
            "depression": self._aggregate_qtype_attention(dep_results),
        }
        stats["modality_contribution"] = {
            "normal": self._aggregate_modality_weights(normal_results),
            "depression": self._aggregate_modality_weights(dep_results),
        }
        return stats

    def _aggregate_qtype_attention(self, results: List[Dict]) -> Dict:
        qtype_attn = {}
        for result in results:
            for attn, qt in zip(result["utterance_attention"], result["q_types"]):
                qtype_attn.setdefault(int(qt), []).append(float(attn))

        return {
            self.qtype_names[qt] if 0 <= qt < len(self.qtype_names) else f"Q{qt}": {
                "mean": float(np.mean(attns)),
                "std": float(np.std(attns)),
                "count": len(attns),
            }
            for qt, attns in qtype_attn.items()
            if attns
        }

    def _aggregate_modality_weights(self, results: List[Dict]) -> Dict:
        audio_weights: List[float] = []
        ling_weights: List[float] = []

        for result in results:
            audio_weights.extend(np.asarray(result["modality_weights"])[:, 0].tolist())
            ling_weights.extend(np.asarray(result["modality_weights"])[:, 1].tolist())

        return {
            "audio": {
                "mean": float(np.mean(audio_weights)) if audio_weights else 0.0,
                "std": float(np.std(audio_weights)) if audio_weights else 0.0,
            },
            "linguistic": {
                "mean": float(np.mean(ling_weights)) if ling_weights else 0.0,
                "std": float(np.std(ling_weights)) if ling_weights else 0.0,
            },
        }

    def save_results_summary(self, results: List[Dict], categorized: Dict, stats: Dict, output_path: Path):
        """Save a compact JSON summary for downstream reporting."""
        summary = {
            "total_participants": len(results),
            "classification": {
                "correct": len(categorized["correct"]),
                "incorrect": len(categorized["incorrect"]),
                "tp": len(categorized["tp"]),
                "tn": len(categorized["tn"]),
                "fp": len(categorized["fp"]),
                "fn": len(categorized["fn"]),
                "accuracy": stats["accuracy"],
            },
            "regression_metrics": {
                "mean_error": stats["mean_error"],
                "rmse": stats["rmse"],
                "mae": stats["mae"],
            },
            "aggregate_statistics": {
                "qtype_importance": stats["qtype_importance"],
                "modality_contribution": stats["modality_contribution"],
            },
            "individual_predictions": [
                {
                    "pid": r["pid"],
                    "pred_phq": r["pred_phq"],
                    "true_phq": r["true_phq"],
                    "correct": r["correct"],
                }
                for r in results
            ],
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info("Results summary saved: %s", output_path)
