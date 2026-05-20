"""
Visualization utilities for explainability reports.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.style.use("seaborn-v0_8-whitegrid")


class ExplainabilityVisualizer:
    """Create interpretable visualizations for participant- and dataset-level review."""

    def __init__(self, config: Dict):
        self.config = config
        self.colors = config["visualization"]["colors"]
        self.qtype_colors = config["visualization"]["qtype_colors"]
        self.qtype_names = config["data"]["qtype_names"]
        self.dpi = config["visualization"]["dpi"]
        self.threshold = config["data"]["depression_threshold"]

    def _qtype_name(self, qt: int) -> str:
        return self.qtype_names[int(qt)] if 0 <= int(qt) < len(self.qtype_names) else f"Q{qt}"

    def _qtype_color(self, qt: int) -> str:
        qtype_name = self._qtype_name(qt)
        return self.qtype_colors.get(qtype_name, self.colors["audio"])

    def plot_utterance_attention(self, result: Dict, save_path: Path):
        """Visualize utterance-level attention weights."""
        attn = np.asarray(result["utterance_attention"])
        texts = result["texts"]
        q_types = result["q_types"]
        pid = result["pid"]

        n_utt = len(attn)
        fig, ax = plt.subplots(figsize=(14, max(6, n_utt * 0.3)))

        qtype_color_list = [self._qtype_color(qt) for qt in q_types]
        y_pos = np.arange(n_utt)
        ax.barh(y_pos, attn, color=qtype_color_list, alpha=0.85, edgecolor="white", linewidth=0.5)

        for i, (weight, text) in enumerate(zip(attn, texts)):
            display_text = text[:70] + "..." if len(text) > 70 else text
            ax.text(float(weight) + 0.001, i, f" {display_text}", va="center", fontsize=8)

        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"Utt {i + 1}" for i in range(n_utt)], fontsize=9)
        ax.set_xlabel("Attention Weight", fontsize=11, fontweight="bold")
        ax.set_title(
            f'Utterance Attention - PID {pid}\nPred: {result["pred_phq"]:.1f} | Actual: {result["true_phq"]:.1f}',
            fontsize=13,
            fontweight="bold",
        )

        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, facecolor=self.qtype_colors[qtype], label=qtype)
            for qtype in self.qtype_names
            if qtype in self.qtype_colors
        ]
        ax.legend(handles=legend_elements, loc="lower right", title="Question Type")

        ax.invert_yaxis()
        ax.set_xlim(0, max(float(np.max(attn)) * 1.5, 0.05))

        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight", facecolor="white")
        plt.close()

    def plot_modality_contribution(self, result: Dict, save_path: Path):
        """Visualize audio vs linguistic contribution per utterance."""
        modality_weights = np.asarray(result["modality_weights"])
        pid = result["pid"]

        n_utt = len(modality_weights)
        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(n_utt)
        audio_w = modality_weights[:, 0]
        ling_w = modality_weights[:, 1]

        ax.bar(x, audio_w, label="Audio", color=self.colors["audio"], alpha=0.85)
        ax.bar(x, ling_w, bottom=audio_w, label="Linguistic", color=self.colors["linguistic"], alpha=0.85)

        ax.set_xlabel("Utterance Index", fontsize=11, fontweight="bold")
        ax.set_ylabel("Contribution Weight", fontsize=11, fontweight="bold")
        ax.set_title(f"Modality Contribution - PID {pid}", fontsize=13, fontweight="bold")
        ax.legend()
        ax.set_ylim(0, 1.1)

        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight", facecolor="white")
        plt.close()

    def plot_audio_gate(self, result: Dict, save_path: Path):
        """Visualize gate values that balance WavLM and Wav2Vec."""
        gate = np.asarray(result.get("audio_gate", []))
        if gate.size == 0:
            return

        fig, ax = plt.subplots(figsize=(12, 4))
        x = np.arange(len(gate))
        ax.plot(x, gate, marker="o", color=self.colors["audio"], linewidth=2)
        ax.fill_between(x, gate, alpha=0.2, color=self.colors["audio"])
        ax.axhline(float(np.mean(gate)), linestyle="--", color=self.colors["warning"], alpha=0.8, label="Mean Gate")

        ax.set_xlabel("Utterance Index", fontsize=11, fontweight="bold")
        ax.set_ylabel("Gate Value", fontsize=11, fontweight="bold")
        ax.set_title(f'Audio Gate (WavLM Preference) - PID {result["pid"]}', fontsize=13, fontweight="bold")
        ax.set_ylim(0, 1)
        ax.legend()

        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight", facecolor="white")
        plt.close()

    def plot_prediction_scatter(self, results: List[Dict], save_path: Path):
        """Scatter plot of predicted vs actual PHQ scores."""
        fig, ax = plt.subplots(figsize=(10, 10))

        for result in results:
            color = self.colors["success"] if result["correct"] else self.colors["danger"]
            marker = "o" if result["correct"] else "x"
            ax.scatter(
                result["true_phq"],
                result["pred_phq"],
                c=color,
                marker=marker,
                s=100,
                alpha=0.75,
                edgecolors="white",
                linewidths=1,
            )

        ax.plot([0, 24], [0, 24], "k--", alpha=0.5, label="Perfect")
        ax.axhline(y=self.threshold, color=self.colors["warning"], linestyle=":", alpha=0.7)
        ax.axvline(x=self.threshold, color=self.colors["warning"], linestyle=":", alpha=0.7)

        ax.set_xlabel("Actual PHQ", fontsize=12, fontweight="bold")
        ax.set_ylabel("Predicted PHQ", fontsize=12, fontweight="bold")
        ax.set_title("Prediction vs Actual", fontsize=14, fontweight="bold")
        ax.set_xlim(-1, 25)
        ax.set_ylim(-1, 25)
        ax.legend()
        ax.set_aspect("equal")

        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight", facecolor="white")
        plt.close()

    def plot_qtype_importance_by_group(self, results: List[Dict], save_path: Path):
        """Aggregate Q-type importance by normal/depression groups."""
        normal_qtype = defaultdict(list)
        dep_qtype = defaultdict(list)

        for result in results:
            is_dep = result["true_phq"] >= self.threshold
            for attn, qt in zip(result["utterance_attention"], result["q_types"]):
                target = dep_qtype if is_dep else normal_qtype
                target[int(qt)].append(float(attn))

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        qtype_color_list = [self.qtype_colors.get(qtype, self.colors["audio"]) for qtype in self.qtype_names]

        means_normal = [np.mean(normal_qtype[i]) if normal_qtype[i] else 0 for i in range(len(self.qtype_names))]
        axes[0].bar(self.qtype_names, means_normal, color=qtype_color_list, alpha=0.85)
        axes[0].set_ylabel("Mean Attention", fontsize=11, fontweight="bold")
        axes[0].set_title("Normal (PHQ < 10)", fontsize=12, fontweight="bold")

        means_dep = [np.mean(dep_qtype[i]) if dep_qtype[i] else 0 for i in range(len(self.qtype_names))]
        axes[1].bar(self.qtype_names, means_dep, color=qtype_color_list, alpha=0.85)
        axes[1].set_ylabel("Mean Attention", fontsize=11, fontweight="bold")
        axes[1].set_title("Depression (PHQ >= 10)", fontsize=12, fontweight="bold")

        plt.suptitle("Question Type Importance by Group", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight", facecolor="white")
        plt.close()
