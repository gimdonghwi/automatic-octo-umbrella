"""
Feature Profiling Module for DAIC-WOZ Dataset

Extracts and analyzes:
- Linguistic features (TTR, word count, etc.)
- Audio features (WavLM embeddings statistics)
- Question type distributions
- Temporal patterns
"""

import pandas as pd
import numpy as np
import torch
from typing import Dict, List, Optional
import logging
from collections import Counter

logger = logging.getLogger(__name__)


class LinguisticFeatureExtractor:
    """
    Extract linguistic features from participant transcripts.
    
    Features:
    - TTR (Type-Token Ratio): Vocabulary diversity
    - Word count: Response verbosity
    - Average word length: Linguistic complexity
    - Filler word ratio: Speech disfluency
    """
    
    FILLER_WORDS = {'uh', 'um', 'like', 'you know', 'i mean', 'well', 'so'}
    
    @staticmethod
    def calculate_ttr(text: str) -> float:
        """
        Type-Token Ratio: Unique words / Total words
        
        Clinical significance:
        Lower TTR correlates with reduced cognitive flexibility
        and limited vocabulary usage in depression.
        """
        if not text or text.strip() == "":
            return 0.0
        
        tokens = text.lower().split()
        
        if len(tokens) == 0:
            return 0.0
        
        unique_tokens = set(tokens)
        return len(unique_tokens) / len(tokens)
    
    @staticmethod
    def calculate_word_count(text: str) -> int:
        """Total word count in participant responses"""
        if not text or text.strip() == "":
            return 0
        return len(text.split())
    
    @staticmethod
    def calculate_avg_word_length(text: str) -> float:
        """Average word length (characters)"""
        if not text or text.strip() == "":
            return 0.0
        
        words = text.split()
        if len(words) == 0:
            return 0.0
        
        return np.mean([len(word) for word in words])
    
    @classmethod
    def extract_all_features(cls, text: str) -> Dict[str, float]:
        """Extract all linguistic features from text"""
        return {
            'ttr': cls.calculate_ttr(text),
            'word_count': cls.calculate_word_count(text),
            'avg_word_length': cls.calculate_avg_word_length(text)
        }


class AudioFeatureProfiler:
    """
    Profile audio embedding characteristics.
    
    Analyzes:
    - Embedding statistics (mean, std, min, max)
    - Temporal patterns
    - Dimensionality
    """
    
    @staticmethod
    def profile_embedding(embedding: torch.Tensor) -> Dict[str, float]:
        """
        Extract statistical profile from audio embeddings.
        
        Args:
            embedding: Tensor of shape (seq_len, embedding_dim)
            
        Returns:
            {
                'seq_len': int,
                'embedding_dim': int,
                'mean': float,
                'std': float,
                'min': float,
                'max': float,
                'l2_norm': float
            }
        """
        if embedding.numel() == 0:
            return {
                'seq_len': 0,
                'embedding_dim': 0,
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'l2_norm': 0.0
            }
        
        return {
            'seq_len': embedding.shape[0],
            'embedding_dim': embedding.shape[1] if len(embedding.shape) > 1 else 1,
            'mean': embedding.mean().item(),
            'std': embedding.std().item(),
            'min': embedding.min().item(),
            'max': embedding.max().item(),
            'l2_norm': torch.norm(embedding).item()
        }
    
    @staticmethod
    def calculate_temporal_statistics(embedding: torch.Tensor) -> Dict[str, float]:
        """
        Analyze temporal patterns in audio embeddings.
        
        Returns:
            {
                'temporal_variance': float,  # Variance across time
                'temporal_range': float       # Max - Min across time
            }
        """
        if embedding.numel() == 0 or len(embedding.shape) < 2:
            return {'temporal_variance': 0.0, 'temporal_range': 0.0}
        
        temporal_mean = embedding.mean(dim=1)
        
        return {
            'temporal_variance': temporal_mean.var().item(),
            'temporal_range': (temporal_mean.max() - temporal_mean.min()).item()
        }


class QuestionTypeAnalyzer:
    """
    Analyze question type distributions and their relationship to depression.
    
    Question types in DAIC-WOZ:
    0: Scripted Introduction
    1: Open Questions  
    2: Rapport Building
    3: Follow-up Questions
    4: Depression Direct (PHQ-9 related)
    ...
    """
    
    QUESTION_TYPE_LABELS = {
        0: "Scripted Introduction",
        1: "Open Questions",
        2: "Rapport Building", 
        3: "Follow-up Questions",
        4: "Depression Direct",
        5: "PTSD Questions",
        6: "Positive/Negative Topics",
        7: "Life Questions",
        8: "Closing",
        9: "Other"
    }
    
    @staticmethod
    def get_question_distribution(
        transcript_df: pd.DataFrame, 
        participant_id: int
    ) -> pd.Series:
        """
        Get distribution of question types for a participant.
        
        Returns:
            Series with question_type as index and counts as values
        """
        mask = transcript_df['participant_id'] == participant_id
        return transcript_df[mask]['question_type'].value_counts()
    
    @staticmethod
    def calculate_question_ratio(
        transcript_df: pd.DataFrame,
        participant_id: int,
        target_qtype: int
    ) -> float:
        """
        Calculate ratio of a specific question type.
        
        Returns:
            Ratio of target question type to total questions
        """
        distribution = QuestionTypeAnalyzer.get_question_distribution(
            transcript_df, 
            participant_id
        )
        
        total = distribution.sum()
        if total == 0:
            return 0.0
        
        return distribution.get(target_qtype, 0) / total
    
    @classmethod
    def analyze_participant_qtypes(
        cls,
        transcript_df: pd.DataFrame,
        participant_id: int
    ) -> Dict[str, float]:
        """
        Comprehensive question type analysis for a participant.
        
        Returns:
            {
                'total_questions': int,
                'qtype_0_ratio': float,
                'qtype_1_ratio': float,
                ...
                'qtype_diversity': float  # Entropy of distribution
            }
        """
        distribution = cls.get_question_distribution(transcript_df, participant_id)
        
        total = distribution.sum()
        
        result = {'total_questions': int(total)}
        
        for qtype in range(10):
            result[f'qtype_{qtype}_ratio'] = distribution.get(qtype, 0) / total if total > 0 else 0.0
        
        if total > 0:
            proportions = distribution.values / total
            proportions = proportions[proportions > 0]
            entropy = -np.sum(proportions * np.log2(proportions))
            result['qtype_diversity'] = entropy
        else:
            result['qtype_diversity'] = 0.0
        
        return result


class ComprehensiveFeatureProfiler:
    """
    Orchestrates all feature extraction for dataset profiling.
    
    Usage:
        profiler = ComprehensiveFeatureProfiler(data_loader)
        features_df = profiler.profile_dataset('train')
    """
    
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.linguistic_extractor = LinguisticFeatureExtractor()
        self.audio_profiler = AudioFeatureProfiler()
        self.qtype_analyzer = QuestionTypeAnalyzer()
    
    def profile_participant(
        self, 
        participant_id: int,
        include_audio: bool = True,
        include_text: bool = True
    ) -> Dict:
        """
        Extract all features for a single participant.
        
        Returns:
            Dictionary with all extracted features
        """
        features = {'participant_id': participant_id}
        
        if include_text:
            try:
                transcript_df = self.data_loader.load_transcript()
                from .data_loader import TranscriptProcessor
                
                processor = TranscriptProcessor(transcript_df)
                participant_text = processor.get_participant_text(participant_id)
                
                linguistic_features = self.linguistic_extractor.extract_all_features(participant_text)
                features.update(linguistic_features)
                
                qtype_features = self.qtype_analyzer.analyze_participant_qtypes(
                    transcript_df, 
                    participant_id
                )
                features.update(qtype_features)
                
            except Exception as e:
                logger.warning(f"Failed to extract text features for P{participant_id}: {e}")
        
        if include_audio:
            try:
                audio_emb = self.data_loader.load_embeddings(participant_id, 'audio')
                audio_features = self.audio_profiler.profile_embedding(audio_emb)
                
                for key, value in audio_features.items():
                    features[f'audio_{key}'] = value
                
            except Exception as e:
                logger.warning(f"Failed to extract audio features for P{participant_id}: {e}")
        
        return features
    
    def profile_dataset(
        self, 
        split: str = 'train',
        include_audio: bool = True,
        include_text: bool = True
    ) -> pd.DataFrame:
        """
        Profile entire dataset split.
        
        Args:
            split: 'train', 'dev', or 'test'
            include_audio: Extract audio features
            include_text: Extract linguistic features
            
        Returns:
            DataFrame with all features and labels
        """
        split_df = self.data_loader.load_split(split)
        participant_ids = split_df['Participant_ID'].tolist()
        
        all_features = []
        
        for pid in participant_ids:
            try:
                features = self.profile_participant(
                    pid, 
                    include_audio=include_audio,
                    include_text=include_text
                )
                all_features.append(features)
                
            except Exception as e:
                logger.error(f"Failed to profile P{pid}: {e}")
        
        features_df = pd.DataFrame(all_features)
        
        result_df = split_df.merge(
            features_df, 
            left_on='Participant_ID', 
            right_on='participant_id',
            how='left'
        )
        
        logger.info(
            f"Profiled {len(result_df)} participants from {split} split "
            f"with {len(result_df.columns)} features"
        )
        
        return result_df
