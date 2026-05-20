"""
Zero-shot screening utility for Ellie utterances.

This script keeps the lightweight workflow that was used during question-type
data preparation:

1. Load Ellie-only utterance CSV files.
2. Use a zero-shot classifier to separate likely questions from statements.
3. Save per-file annotations.
4. Optionally merge the predicted question rows into a single review file.

The output is intended as a practical curation step, not as a gold-standard
annotation pipeline.
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
from pathlib import Path
from typing import Iterable, List

import pandas as pd
import torch
from transformers import pipeline

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "facebook/bart-large-mnli"
DEFAULT_CANDIDATE_LABELS = ["question", "statement"]


def iter_target_files(root_path: str, pattern: str) -> List[str]:
    return sorted(glob.glob(os.path.join(root_path, pattern)))


def build_classifier(model_name: str):
    device = 0 if torch.cuda.is_available() else -1
    logger.info("Loading zero-shot model: %s", model_name)
    return pipeline("zero-shot-classification", model=model_name, device=device)


def annotate_file(
    file_path: str,
    classifier,
    candidate_labels: Iterable[str],
    text_column: str = "value",
    label_column: str = "label",
    score_column: str = "score",
) -> bool:
    try:
        df = pd.read_csv(file_path)
    except Exception as exc:
        logger.warning("Failed to read %s: %s", file_path, exc)
        return False

    if df.empty or text_column not in df.columns:
        return False

    texts = df[text_column].fillna("").astype(str).tolist()
    if not texts:
        return False

    results = classifier(texts, list(candidate_labels))
    predicted_labels = []
    confidence_scores = []

    for res in results:
        predicted_labels.append(res["labels"][0])
        confidence_scores.append(float(res["scores"][0]))

    df[label_column] = predicted_labels
    df[score_column] = confidence_scores
    df.to_csv(file_path, index=False, encoding="utf-8-sig")
    logger.info("Annotated %s", file_path)
    return True


def annotate_directory(
    root_path: str,
    pattern: str,
    model_name: str = DEFAULT_MODEL_NAME,
    candidate_labels: Iterable[str] = DEFAULT_CANDIDATE_LABELS,
) -> int:
    classifier = build_classifier(model_name)
    target_files = iter_target_files(root_path, pattern)
    updated = 0

    for file_path in target_files:
        if annotate_file(file_path, classifier, candidate_labels):
            updated += 1

    logger.info("Annotated %d files", updated)
    return updated


def merge_predicted_questions(
    root_path: str,
    pattern: str,
    output_path: str,
    text_column: str = "value",
    label_column: str = "label",
    positive_label: str = "question",
) -> int:
    target_files = iter_target_files(root_path, pattern)
    frames = []

    for file_path in target_files:
        try:
            df = pd.read_csv(file_path)
        except Exception as exc:
            logger.warning("Failed to read %s: %s", file_path, exc)
            continue

        if df.empty or text_column not in df.columns or label_column not in df.columns:
            continue

        subset = df[df[label_column].astype(str).str.lower() == positive_label.lower()].copy()
        if subset.empty:
            continue

        participant_id = Path(file_path).name.split("_")[0]
        subset.insert(0, "Participant_ID", participant_id)
        frames.append(subset)

    if not frames:
        logger.warning("No predicted question rows found.")
        return 0

    merged = pd.concat(frames, ignore_index=True)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output, index=False, encoding="utf-8-sig")
    logger.info("Saved merged file to %s", output)
    return len(merged)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Zero-shot screen Ellie utterances into question vs statement.")
    parser.add_argument("--root-path", required=True, help="Root directory containing participant folders.")
    parser.add_argument(
        "--pattern",
        default=os.path.join("*_P", "*_Ellie_comments.csv"),
        help="Glob pattern relative to root path.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Hugging Face zero-shot model name.",
    )
    parser.add_argument(
        "--merge-output",
        default="",
        help="Optional output CSV path for merging rows predicted as questions.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    annotate_directory(
        root_path=args.root_path,
        pattern=args.pattern,
        model_name=args.model_name,
    )

    if args.merge_output:
        merge_predicted_questions(
            root_path=args.root_path,
            pattern=args.pattern,
            output_path=args.merge_output,
        )


if __name__ == "__main__":
    main()
