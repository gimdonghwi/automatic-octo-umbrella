"""
Training Infrastructure
Combines Dataset, Trainer, and Evaluator for depression detection.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import pickle
import math
import os
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from tqdm import tqdm
import logging
from pathlib import Path

from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    f1_score, precision_score, recall_score,
    confusion_matrix, accuracy_score,
    precision_recall_fscore_support
)
from scipy.stats import pearsonr, spearmanr

logger = logging.getLogger(__name__)


# =============================================================================
# Enhanced TTR Feature Extraction
# =============================================================================
def extract_enhanced_ttr_features(text: str) -> np.ndarray:
    """Extract 6-dimensional enhanced TTR features from text"""
    if not text or len(text.strip()) == 0:
        return np.zeros(6, dtype=np.float32)
    
    tokens = text.lower().split()
    if len(tokens) == 0:
        return np.zeros(6, dtype=np.float32)
    
    unique_tokens = set(tokens)
    
    # 1. TTR
    ttr = len(unique_tokens) / len(tokens)
    
    # 2. Log TTR
    ttr_log = len(unique_tokens) / math.log(len(tokens) + 1)
    
    # 3. Repetition rate
    word_counts = {}
    for token in tokens:
        word_counts[token] = word_counts.get(token, 0) + 1
    
    repeated_words = sum(1 for count in word_counts.values() if count > 1)
    repetition_rate = repeated_words / len(unique_tokens) if len(unique_tokens) > 0 else 0.0
    
    # 4. Unique word ratio
    unique_words = sum(1 for count in word_counts.values() if count == 1)
    unique_word_ratio = unique_words / len(tokens)
    
    # 5. Lexical density
    function_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
    }
    content_words = [token for token in tokens if token not in function_words]
    lexical_density = len(content_words) / len(tokens) if len(tokens) > 0 else 0.0
    
    # 6. Word length variance
    word_lengths = [len(token) for token in tokens]
    word_length_variance = np.var(word_lengths) if len(word_lengths) > 1 else 0.0
    word_length_variance = min(word_length_variance / 10.0, 1.0)
    
    return np.array([
        ttr, ttr_log / 10.0,
        repetition_rate, unique_word_ratio,
        lexical_density, word_length_variance
    ], dtype=np.float32)


# =============================================================================
# Dataset
# =============================================================================
class DepressionDataset(Dataset):
    """Dataset for depression detection with sequence-based features"""
    
    def __init__(
        self,
        data_dict: Dict,
        participant_ids: List[str],
        config: Dict,
        max_len: int = 100
    ):
        self.data_dict = data_dict
        self.pids = participant_ids
        self.config = config
        self.max_len = max_len
        self.phq_max = config['data']['phq_max']
    
    def __len__(self) -> int:
        return len(self.pids)
    
    def __getitem__(self, idx: int) -> Dict:
        pid = self.pids[idx]
        sample = self.data_dict[pid]
        
        sequence = sample['sequence']
        seq_len = min(len(sequence), self.max_len)
        
        # Initialize tensors
        wavlm = np.zeros((self.max_len, 768), dtype=np.float32)
        wav2vec = np.zeros((self.max_len, 768), dtype=np.float32)
        ttr = np.zeros((self.max_len, 6), dtype=np.float32)
        q_types = np.zeros(self.max_len, dtype=np.int64)
        mask = np.zeros(self.max_len, dtype=np.float32)
        
        # Fill with actual data
        for i in range(seq_len):
            utt = sequence[i]
            wavlm[i] = utt['wavlm']
            wav2vec[i] = utt['wav2vec']
            ttr[i] = extract_enhanced_ttr_features(utt['text'])
            q_types[i] = utt['q_type_id']
            mask[i] = 1.0
        
        # Normalize PHQ score to [0, 1]
        phq_score = sample['label']
        normalized_score = phq_score / self.phq_max
        
        return {
            'wavlm': torch.FloatTensor(wavlm),
            'wav2vec': torch.FloatTensor(wav2vec),
            'ttr': torch.FloatTensor(ttr),
            'q_types': torch.LongTensor(q_types),
            'mask': torch.FloatTensor(mask),
            'score': torch.FloatTensor([normalized_score]),
            'pid': pid
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """Custom collate function"""
    return {
        'wavlm': torch.stack([item['wavlm'] for item in batch]),
        'wav2vec': torch.stack([item['wav2vec'] for item in batch]),
        'ttr': torch.stack([item['ttr'] for item in batch]),
        'q_types': torch.stack([item['q_types'] for item in batch]),
        'mask': torch.stack([item['mask'] for item in batch]),
        'score': torch.stack([item['score'] for item in batch]),
        'pids': [item['pid'] for item in batch]
    }


# =============================================================================
# Data Loading
# =============================================================================
def load_data(config: Dict) -> Tuple[Dict, pd.DataFrame]:
    """Load preprocessed dataset and metadata"""
    base_path = Path(config['data']['base_path'])
    
    # Load preprocessed data
    data_file = base_path / config['data']['preprocessed_file']
    with open(data_file, 'rb') as f:
        data_dict = pickle.load(f)
    
    # Load metadata
    meta_file = base_path / config['data']['metadata_file']
    metadata = pd.read_csv(meta_file)
    
    logger.info(f"Loaded {len(data_dict)} participants")
    
    return data_dict, metadata


# =============================================================================
# Evaluator
# =============================================================================
class Evaluator:
    """Evaluation metrics for depression detection"""
    
    def __init__(self, config: Dict, device: torch.device):
        self.config = config
        self.device = device
        self.phq_max = config['data']['phq_max']
        self.threshold = config['data']['depression_threshold']
    
    def evaluate(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module
    ) -> Tuple[np.ndarray, np.ndarray, List[str], float]:
        """Run evaluation"""
        model.eval()
        all_preds, all_targets, all_pids = [], [], []
        total_loss = 0
        
        with torch.no_grad():
            for batch in dataloader:
                wavlm = batch['wavlm'].to(self.device)
                wav2vec = batch['wav2vec'].to(self.device)
                ttr = batch['ttr'].to(self.device)
                q_types = batch['q_types'].to(self.device)
                mask = batch['mask'].to(self.device)
                targets = batch['score'].to(self.device)
                
                preds = model(wavlm, wav2vec, ttr, q_types, mask)
                loss = criterion(preds, targets)
                total_loss += loss.item()
                
                all_preds.extend(preds.cpu().numpy().flatten())
                all_targets.extend(targets.cpu().numpy().flatten())
                all_pids.extend(batch['pids'])
        
        # De-normalize to PHQ scale
        preds_phq = np.array(all_preds) * self.phq_max
        targets_phq = np.array(all_targets) * self.phq_max
        
        return targets_phq, preds_phq, all_pids, total_loss / len(dataloader)
    
    def compute_metrics(
        self,
        targets_phq: np.ndarray,
        preds_phq: np.ndarray
    ) -> Dict:
        """Compute regression and classification metrics"""
        
        # Regression metrics
        rmse = np.sqrt(mean_squared_error(targets_phq, preds_phq))
        mae = mean_absolute_error(targets_phq, preds_phq)
        r2 = r2_score(targets_phq, preds_phq)
        pearson_r, _ = pearsonr(targets_phq, preds_phq)
        spearman_r, _ = spearmanr(targets_phq, preds_phq)
        
        # Classification (threshold-based)
        pred_labels = (preds_phq >= self.threshold).astype(int)
        true_labels = (targets_phq >= self.threshold).astype(int)
        
        cm = confusion_matrix(true_labels, pred_labels)
        accuracy = accuracy_score(true_labels, pred_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, pred_labels, average='binary', zero_division=0
        )
        
        # F1 per class
        f1_per_class = f1_score(true_labels, pred_labels, average=None, zero_division=0)
        f1_normal = f1_per_class[0] if len(f1_per_class) > 0 else 0
        f1_depression = f1_per_class[1] if len(f1_per_class) > 1 else 0
        
        # Balanced F1
        if f1_normal > 0 and f1_depression > 0:
            balanced_f1 = 2 * f1_normal * f1_depression / (f1_normal + f1_depression)
        else:
            balanced_f1 = 0.0
        
        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'pearson': pearson_r,
            'spearman': spearman_r,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'f1_normal': f1_normal,
            'f1_depression': f1_depression,
            'balanced_f1': balanced_f1,
            'confusion_matrix': cm
        }


# =============================================================================
# Trainer
# =============================================================================
class Trainer:
    """Training loop with early stopping"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        evaluator: Evaluator,
        config: Dict,
        device: torch.device
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.evaluator = evaluator
        self.config = config
        self.device = device
        
        self.best_metric = 0.0
        self.patience_counter = 0
        self.history = defaultdict(list)
    
    def train_epoch(self) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch in self.train_loader:
            wavlm = batch['wavlm'].to(self.device)
            wav2vec = batch['wav2vec'].to(self.device)
            ttr = batch['ttr'].to(self.device)
            q_types = batch['q_types'].to(self.device)
            mask = batch['mask'].to(self.device)
            targets = batch['score'].to(self.device)
            
            self.optimizer.zero_grad()
            
            preds = self.model(wavlm, wav2vec, ttr, q_types, mask)
            loss = self.criterion(preds, targets)
            
            loss.backward()
            
            # Gradient clipping
            if 'gradient_clip' in self.config['training']:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['gradient_clip']
                )
            
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)
    
    def train(self) -> Dict:
        """Full training loop with early stopping"""
        max_epochs = self.config['training']['max_epochs']
        patience = self.config['training']['patience']
        primary_metric = self.config['evaluation']['primary_metric']
        
        for epoch in range(max_epochs):
            train_loss = self.train_epoch()
            
            # Validation
            targets_phq, preds_phq, _, val_loss = self.evaluator.evaluate(
                self.model, self.val_loader, self.criterion
            )
            metrics = self.evaluator.compute_metrics(targets_phq, preds_phq)
            
            # Scheduler step
            self.scheduler.step()
            
            # Logging
            current_metric = metrics[primary_metric]
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history[primary_metric].append(current_metric)
            
            # Early stopping check
            if current_metric > self.best_metric:
                self.best_metric = current_metric
                self.patience_counter = 0
                logger.info(
                    f"Epoch {epoch+1}: {primary_metric}={current_metric:.3f} "
                    f"RMSE={metrics['rmse']:.3f} ⭐"
                )
            else:
                self.patience_counter += 1
            
            if self.patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        return self.history
