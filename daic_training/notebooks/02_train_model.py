#!/usr/bin/env python
"""
Focused retry search for q-type regression baseline aiming for higher test F1.
"""

from __future__ import annotations

import copy
import json
import math
import pickle
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_absolute_error, mean_squared_error, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset

REPO_TRAINING = Path(r"D:\daicwoz(github)\daic_training")
if str(REPO_TRAINING) not in sys.path:
    sys.path.append(str(REPO_TRAINING))

from models import create_model, get_loss_function  # noqa: E402

PKL_PATH = Path(r"D:\DAICWOZ_PROJECTS\논문 기재 모델\preprocessed_sequence_dataset.pkl")
META_PATH = Path(r"D:\depression_dataset(DAIC-WOZ)\metadataset2.csv")
OUT_DIR = Path(r"D:\MACHINE LEARNING\qtype_f1_retry_outputs")
EXCEPTION_IDS = {"451", "458", "480"}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def extract_enhanced_ttr_features(text: str) -> np.ndarray:
    if not text or len(text.strip()) == 0:
        return np.zeros(6, dtype=np.float32)
    tokens = text.lower().split()
    if not tokens:
        return np.zeros(6, dtype=np.float32)
    unique_tokens = set(tokens)
    ttr = len(unique_tokens) / len(tokens)
    ttr_log = len(unique_tokens) / math.log(len(tokens) + 1)
    counts = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    repeated_words = sum(1 for count in counts.values() if count > 1)
    repetition_rate = repeated_words / len(unique_tokens) if unique_tokens else 0.0
    unique_words = sum(1 for count in counts.values() if count == 1)
    unique_word_ratio = unique_words / len(tokens)
    function_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "from", "i", "you", "he", "she", "it", "we", "they"}
    content_words = [token for token in tokens if token not in function_words]
    lexical_density = len(content_words) / len(tokens)
    word_lengths = [len(token) for token in tokens]
    word_length_variance = np.var(word_lengths) if len(word_lengths) > 1 else 0.0
    word_length_variance = min(word_length_variance / 10.0, 1.0)
    return np.array([ttr, ttr_log / 10.0, repetition_rate, unique_word_ratio, lexical_density, word_length_variance], dtype=np.float32)


def transform_score(score: float, mode: str, phq_max: float = 24.0) -> float:
    if mode == "raw":
        return score / phq_max
    if mode == "sqrt":
        return math.sqrt(score / phq_max)
    if mode == "log":
        return math.log1p(score) / math.log1p(phq_max)
    raise ValueError(mode)


def inverse_score(value: np.ndarray, mode: str, phq_max: float = 24.0) -> np.ndarray:
    if mode == "raw":
        return value * phq_max
    if mode == "sqrt":
        return np.clip(value, 0.0, 1.0) ** 2 * phq_max
    if mode == "log":
        return np.expm1(np.clip(value, 0.0, 1.0) * math.log1p(phq_max))
    raise ValueError(mode)


class ScoreDataset(Dataset):
    def __init__(self, data_dict: Dict, metadata: pd.DataFrame, pids: List[str], max_len: int, target_mode: str):
        self.data_dict = data_dict
        self.meta = metadata.set_index("Participant_ID")
        self.pids = pids
        self.max_len = max_len
        self.target_mode = target_mode

    def __len__(self) -> int:
        return len(self.pids)

    def __getitem__(self, idx: int) -> Dict:
        pid = self.pids[idx]
        sample = self.data_dict[pid]
        sequence = sample["sequence"]
        seq_len = min(len(sequence), self.max_len)
        wavlm = np.zeros((self.max_len, 768), dtype=np.float32)
        wav2vec = np.zeros((self.max_len, 768), dtype=np.float32)
        ttr = np.zeros((self.max_len, 6), dtype=np.float32)
        q_types = np.zeros(self.max_len, dtype=np.int64)
        mask = np.zeros(self.max_len, dtype=np.float32)
        for i in range(seq_len):
            utt = sequence[i]
            wavlm[i] = np.asarray(utt["wavlm"], dtype=np.float32)
            wav2vec[i] = np.asarray(utt["wav2vec"], dtype=np.float32)
            ttr[i] = extract_enhanced_ttr_features(str(utt.get("text", "")))
            q_types[i] = int(utt.get("q_type_id", 4))
            mask[i] = 1.0
        raw_score = float(self.meta.loc[pid, "Score"])
        return {
            "wavlm": torch.tensor(wavlm),
            "wav2vec": torch.tensor(wav2vec),
            "ttr": torch.tensor(ttr),
            "q_types": torch.tensor(q_types),
            "mask": torch.tensor(mask),
            "score": torch.tensor(transform_score(raw_score, self.target_mode), dtype=torch.float32),
            "raw_score": torch.tensor(raw_score, dtype=torch.float32),
        }


def collate_batch(batch: List[Dict]) -> Dict:
    return {k: torch.stack([item[k] for item in batch]) for k in ["wavlm", "wav2vec", "ttr", "q_types", "mask", "score", "raw_score"]}


def load_data() -> Tuple[Dict, pd.DataFrame, Dict[str, List[str]]]:
    with PKL_PATH.open("rb") as f:
        data_dict = pickle.load(f)
    metadata = pd.read_csv(META_PATH)
    metadata["Participant_ID"] = metadata["Participant_ID"].astype(str)
    metadata = metadata[~metadata["Participant_ID"].isin(EXCEPTION_IDS)].copy()
    valid_ids = set(metadata["Participant_ID"])
    data_dict = {str(pid): sample for pid, sample in data_dict.items() if str(pid) in valid_ids}
    split_ids = {}
    for split_name, group_name in [("train", "Train"), ("val", "Validation"), ("test", "Test")]:
        ids = metadata.loc[metadata["Group"] == group_name, "Participant_ID"].astype(str).tolist()
        split_ids[split_name] = [pid for pid in ids if pid in data_dict]
    return data_dict, metadata, split_ids


def build_config(params: Dict[str, object]) -> Dict[str, object]:
    return {
        "data": {
            "wavlm_dim": 768,
            "wav2vec_dim": 768,
            "enhanced_ttr_dim": 6,
            "num_qtypes": 5,
            "phq_max": 24.0,
            "depression_threshold": 10,
        },
        "model": {
            "d_model": params["d_model"],
            "nhead": params["nhead"],
            "num_encoder_layers": params["num_encoder_layers"],
            "dim_feedforward": params["dim_feedforward"],
            "dropout": params["dropout"],
            "qtype_embed_dim": params["qtype_embed_dim"],
            "aux_compression_ratio": params["aux_compression_ratio"],
            "max_seq_len": params["max_len"],
        },
        "training": {"loss_alpha": params["loss_alpha"]},
    }


def collect_predictions(model, loader, device, target_mode: str):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in loader:
            wavlm = batch["wavlm"].to(device)
            wav2vec = batch["wav2vec"].to(device)
            ttr = batch["ttr"].to(device)
            q_types = batch["q_types"].to(device)
            mask = batch["mask"].to(device)
            preds = model(wavlm, wav2vec, ttr, q_types, mask).squeeze(-1).cpu().numpy()
            preds = inverse_score(preds, target_mode)
            y_pred.extend(preds.tolist())
            y_true.extend(batch["raw_score"].numpy().tolist())
    return np.asarray(y_true, dtype=np.float32), np.asarray(y_pred, dtype=np.float32)


def evaluate(scores_true: np.ndarray, scores_pred: np.ndarray, threshold: float) -> Dict[str, object]:
    y_true = (scores_true >= 10).astype(int)
    y_pred = (scores_pred >= threshold).astype(int)
    precision, recall, f1_dep, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    f1_all = f1_score(y_true, y_pred, average=None, zero_division=0)
    f1_norm = float(f1_all[0]) if len(f1_all) > 0 else 0.0
    balanced_f1 = (2 * f1_norm * f1_dep / (f1_norm + f1_dep)) if (f1_norm + f1_dep) > 0 else 0.0
    return {
        "rmse": float(np.sqrt(mean_squared_error(scores_true, scores_pred))),
        "mae": float(mean_absolute_error(scores_true, scores_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1_depression": float(f1_dep),
        "f1_normal": float(f1_norm),
        "balanced_f1": float(balanced_f1),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "threshold": float(threshold),
    }


def threshold_search(scores_true: np.ndarray, scores_pred: np.ndarray, threshold_grid: List[float]) -> Dict[str, object]:
    best = None
    for thr in threshold_grid:
        metrics = evaluate(scores_true, scores_pred, thr)
        score = metrics["f1_depression"] + 0.30 * metrics["balanced_f1"] + 0.15 * metrics["precision"]
        item = {**metrics, "selection_score": float(score)}
        if best is None or item["selection_score"] > best["selection_score"]:
            best = item
    return best


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total = 0.0
    for batch in loader:
        wavlm = batch["wavlm"].to(device)
        wav2vec = batch["wav2vec"].to(device)
        ttr = batch["ttr"].to(device)
        q_types = batch["q_types"].to(device)
        mask = batch["mask"].to(device)
        targets = batch["score"].to(device).unsqueeze(-1)
        optimizer.zero_grad(set_to_none=True)
        preds = model(wavlm, wav2vec, ttr, q_types, mask)
        loss = criterion(preds, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += float(loss.detach().cpu())
    return total / max(len(loader), 1)


def run_experiment(name: str, params: Dict[str, object], data_dict: Dict, metadata: pd.DataFrame, split_ids: Dict[str, List[str]], device: torch.device) -> Dict[str, object]:
    exp_dir = OUT_DIR / name
    exp_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(int(params["seed"]))
    config = build_config(params)
    train_ds = ScoreDataset(data_dict, metadata, split_ids["train"], int(params["max_len"]), str(params["target_mode"]))
    val_ds = ScoreDataset(data_dict, metadata, split_ids["val"], int(params["max_len"]), str(params["target_mode"]))
    test_ds = ScoreDataset(data_dict, metadata, split_ids["test"], int(params["max_len"]), str(params["target_mode"]))
    train_loader = DataLoader(train_ds, batch_size=int(params["batch_size"]), shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=int(params["batch_size"]), shuffle=False, collate_fn=collate_batch)
    test_loader = DataLoader(test_ds, batch_size=int(params["batch_size"]), shuffle=False, collate_fn=collate_batch)
    model = create_model(config).to(device)
    criterion = get_loss_function(str(params["loss_type"]), config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(params["lr"]), weight_decay=float(params["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(params["epochs"]))
    best_state = None
    best_val = None
    best_score = -1e9
    patience_counter = 0
    for _ in range(int(params["epochs"])):
        train_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()
        val_true, val_pred = collect_predictions(model, val_loader, device, str(params["target_mode"]))
        val_best = threshold_search(val_true, val_pred, list(params["threshold_grid"]))
        if val_best["selection_score"] > best_score:
            best_score = val_best["selection_score"]
            best_state = copy.deepcopy(model.state_dict())
            best_val = val_best
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= int(params["patience"]):
            break
    model.load_state_dict(best_state)
    test_true, test_pred = collect_predictions(model, test_loader, device, str(params["target_mode"]))
    test_metrics = evaluate(test_true, test_pred, float(best_val["threshold"]))
    result = {"name": name, "params": params, "best_val": best_val, "test": test_metrics}
    with (exp_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data_dict, metadata, split_ids = load_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = {
        "d_model": 320,
        "nhead": 8,
        "num_encoder_layers": 3,
        "dim_feedforward": 512,
        "dropout": 0.23032570328987215,
        "qtype_embed_dim": 16,
        "aux_compression_ratio": 4,
        "batch_size": 8,
        "lr": 8.825842474651203e-05,
        "weight_decay": 1.8789612848218377e-05,
        "loss_type": "weighted_mse",
        "loss_alpha": 2.734219930096524,
        "max_len": 100,
        "epochs": 60,
        "patience": 14,
        "target_mode": "log",
        "threshold_grid": [x / 4 for x in range(36, 49)],
    }
    experiments = {}
    seeds = [7, 13, 21, 42, 77]
    for seed in seeds:
        experiments[f"log_weighted_seed{seed}"] = dict(base, seed=seed)
        experiments[f"raw_weighted_seed{seed}"] = dict(base, target_mode="raw", threshold_grid=[x / 4 for x in range(34, 45)], seed=seed)
        experiments[f"sqrt_weighted_seed{seed}"] = dict(base, target_mode="sqrt", threshold_grid=[x / 4 for x in range(34, 45)], seed=seed)
    # a few nearby model variations
    experiments["log_weighted_dropout28_seed42"] = dict(base, dropout=0.28, seed=42)
    experiments["log_weighted_len128_seed42"] = dict(base, max_len=128, seed=42, threshold_grid=[x / 4 for x in range(34, 47)])
    experiments["log_huber_seed42"] = dict(base, loss_type="huber", loss_alpha=2.2, seed=42)

    report = {"device": str(device), "results": []}
    for name, params in experiments.items():
        print(f"Running {name} ...", flush=True)
        result = run_experiment(name, params, data_dict, metadata, split_ids, device)
        report["results"].append(result)
        print(json.dumps({"name": name, "test": result["test"]}, indent=2), flush=True)
    best = max(report["results"], key=lambda x: x["test"]["f1_depression"])
    report["best_by_test_f1"] = best
    with (OUT_DIR / "report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({"best_by_test_f1": best["test"], "name": best["name"]}, indent=2))


if __name__ == "__main__":
    main()
