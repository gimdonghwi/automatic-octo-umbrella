"""
Report generation helpers for explainability outputs.

Creates participant-level reports and dataset-level summary artifacts.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generate text and visual explainability reports."""

    def __init__(self, config: Dict, visualizer):
        self.config = config
        self.visualizer = visualizer
        self.qtype_names = config["data"]["qtype_names"]
        self.top_k = config["analysis"].get("top_k_utterances", 5)

    def create_individual_report(self, result: Dict, output_dir: Path) -> Path:
        """
        Generate a participant-level explainability package.

        Creates:
        - utterance_attention.png
        - modality_contribution.png
        - audio_gate.png
        - report.txt
        """
        pid = result["pid"]
        pid_dir = output_dir / f"PID_{pid}"
        pid_dir.mkdir(parents=True, exist_ok=True)

        self.visualizer.plot_utterance_attention(result, pid_dir / "utterance_attention.png")
        self.visualizer.plot_modality_contribution(result, pid_dir / "modality_contribution.png")
        self.visualizer.plot_audio_gate(result, pid_dir / "audio_gate.png")
        self._write_text_report(result, pid_dir / "report.txt")

        logger.info("Report saved for PID %s: %s", pid, pid_dir)
        return pid_dir

    def create_summary_report(
        self,
        results: List[Dict],
        categorized: Dict[str, List[Dict]],
        stats: Dict,
        output_dir: Path,
        case_studies: Optional[Dict[str, Optional[Dict]]] = None,
    ) -> Path:
        """
        Generate dataset-level summary artifacts.

        Creates:
        - prediction_scatter.png
        - qtype_importance_by_group.png
        - summary.txt
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        self.visualizer.plot_prediction_scatter(results, output_dir / "prediction_scatter.png")
        self.visualizer.plot_qtype_importance_by_group(results, output_dir / "qtype_importance_by_group.png")
        self._write_summary_text(results, categorized, stats, output_dir / "summary.txt", case_studies or {})

        logger.info("Summary report saved: %s", output_dir)
        return output_dir

    def _write_text_report(self, result: Dict, output_path: Path):
        """Write a detailed participant-level text report."""
        lines: List[str] = []
        lines.append("=" * 64)
        lines.append(f"EXPLAINABILITY REPORT - PID {result['pid']}")
        lines.append("=" * 64)
        lines.append("")

        lines.append("[PREDICTION]")
        lines.append(f"  Predicted PHQ:   {result['pred_phq']:.2f}")
        lines.append(f"  Actual PHQ:      {result['true_phq']:.2f}")
        lines.append(f"  Error:           {result['pred_phq'] - result['true_phq']:.2f}")
        lines.append(f"  Classification:  {'Depression' if result['pred_label'] == 1 else 'Normal'}")
        lines.append(f"  Correct:         {'Yes' if result['correct'] else 'No'}")
        lines.append("")

        lines.append("[MODALITY CONTRIBUTION]")
        audio_mean = float(np.mean(result["modality_weights"][:, 0]))
        ling_mean = float(np.mean(result["modality_weights"][:, 1]))
        lines.append(f"  Audio:           {audio_mean:.3f} ({audio_mean * 100:.1f}%)")
        lines.append(f"  Linguistic:      {ling_mean:.3f} ({ling_mean * 100:.1f}%)")
        if "audio_gate" in result and len(result["audio_gate"]) > 0:
            gate_mean = float(np.mean(result["audio_gate"]))
            lines.append(f"  WavLM Gate Mean: {gate_mean:.3f}")
        lines.append("")

        lines.append("[Q-TYPE ATTENTION]")
        qtype_attn: Dict[int, List[float]] = {}
        for attn, qt in zip(result["utterance_attention"], result["q_types"]):
            qtype_attn.setdefault(int(qt), []).append(float(attn))

        for qt in sorted(qtype_attn.keys()):
            mean_attn = np.mean(qtype_attn[qt])
            qtype_name = self.qtype_names[qt] if 0 <= qt < len(self.qtype_names) else f"Q{qt}"
            lines.append(f"  {qtype_name:12s}: {mean_attn:.4f} (n={len(qtype_attn[qt])})")
        lines.append("")

        lines.append(f"[TOP {self.top_k} ATTENDED UTTERANCES]")
        sorted_idx = np.argsort(result["utterance_attention"])[::-1][: self.top_k]
        for rank, idx in enumerate(sorted_idx, 1):
            text = str(result["texts"][idx]).strip()
            qtype_idx = int(result["q_types"][idx])
            qtype = self.qtype_names[qtype_idx] if 0 <= qtype_idx < len(self.qtype_names) else f"Q{qtype_idx}"
            attn = float(result["utterance_attention"][idx])
            preview = text[:120] + ("..." if len(text) > 120 else "")
            lines.append(f"  {rank}. [{qtype}] (attn={attn:.4f})")
            lines.append(f'     "{preview}"')

        output_path.write_text("\n".join(lines), encoding="utf-8")

    def _write_summary_text(
        self,
        results: List[Dict],
        categorized: Dict[str, List[Dict]],
        stats: Dict,
        output_path: Path,
        case_studies: Dict[str, Optional[Dict]],
    ):
        """Write a dataset-level summary text report."""
        lines: List[str] = []
        lines.append("=" * 64)
        lines.append("EXPLAINABILITY SUMMARY REPORT")
        lines.append("=" * 64)
        lines.append("")

        lines.append("[CLASSIFICATION]")
        lines.append(f"  Total Participants: {len(results)}")
        lines.append(f"  Correct:            {len(categorized['correct'])}")
        lines.append(f"  Incorrect:          {len(categorized['incorrect'])}")
        lines.append(f"  TP / TN:            {len(categorized['tp'])} / {len(categorized['tn'])}")
        lines.append(f"  FP / FN:            {len(categorized['fp'])} / {len(categorized['fn'])}")
        lines.append(f"  Accuracy:           {stats['accuracy']:.4f}")
        lines.append("")

        lines.append("[REGRESSION]")
        lines.append(f"  Mean Error:         {stats['mean_error']:.4f}")
        lines.append(f"  RMSE:               {stats['rmse']:.4f}")
        lines.append(f"  MAE:                {stats['mae']:.4f}")
        lines.append("")

        lines.append("[AGGREGATE MODALITY CONTRIBUTION]")
        for group_name, values in stats["modality_contribution"].items():
            lines.append(f"  {group_name.capitalize()}:")
            lines.append(f"    Audio Mean:       {values['audio']['mean']:.4f}")
            lines.append(f"    Linguistic Mean:  {values['linguistic']['mean']:.4f}")
        lines.append("")

        lines.append("[CASE STUDIES]")
        for label, case in case_studies.items():
            if case is None:
                lines.append(f"  {label}: None")
                continue
            lines.append(
                f"  {label}: PID {case['pid']} | pred={case['pred_phq']:.2f} | true={case['true_phq']:.2f}"
            )

        output_path.write_text("\n".join(lines), encoding="utf-8")

    def save_case_studies_json(self, case_studies: Dict[str, Optional[Dict]], output_path: Path):
        """Persist lightweight case study metadata."""
        payload = {}
        for key, value in case_studies.items():
            if value is None:
                payload[key] = None
                continue
            payload[key] = {
                "pid": value["pid"],
                "pred_phq": float(value["pred_phq"]),
                "true_phq": float(value["true_phq"]),
                "pred_label": int(value["pred_label"]),
                "true_label": int(value["true_label"]),
                "correct": bool(value["correct"]),
            }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
