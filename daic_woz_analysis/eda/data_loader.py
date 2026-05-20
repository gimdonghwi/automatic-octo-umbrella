"""
Data Loading Infrastructure for DAIC-WOZ Depression Detection

Handles:
- Multi-split loading (train/dev/test)
- Embedding retrieval with validation
- Transcript processing
- Missing data detection
"""

import pandas as pd
import torch
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import yaml
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DAICDataLoader:
    """
    Production data loader for DAIC-WOZ dataset.
    
    Usage:
        loader = DAICDataLoader.from_config('config/paths.yaml')
        train_df = loader.load_split('train')
        embeddings = loader.load_embeddings(participant_id=300, modality='audio')
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.root_dir = Path(self._resolve_env_vars(config['data']['root_dir']))
        
        if not self.root_dir.exists():
            raise FileNotFoundError(
                f"Data root not found: {self.root_dir}\n"
                f"Set DAIC_ROOT environment variable or update config."
            )
        
        self._cache = {}
        logger.info(f"Initialized data loader: {self.root_dir}")
    
    @classmethod
    def from_config(cls, config_path: str = 'config/paths.yaml'):
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return cls(config)
    
    def load_split(self, split: str) -> pd.DataFrame:
        """
        Load train/dev/test split with validation.
        
        Args:
            split: 'train', 'dev', or 'test'
            
        Returns:
            DataFrame with columns: Participant_ID, PHQ8_Binary, PHQ8_Score, Gender
        """
        cache_key = f"split_{split}"
        if cache_key in self._cache:
            logger.debug(f"Returning cached {split} split")
            return self._cache[cache_key]
        
        split_file = self.config['data']['splits'].get(split)
        if not split_file:
            raise ValueError(f"Unknown split: {split}. Use 'train', 'dev', or 'test'")
        
        csv_path = self.root_dir / split_file
        
        if not csv_path.exists():
            raise FileNotFoundError(f"Split file not found: {csv_path}")
        
        df = pd.read_csv(csv_path)
        
        required_cols = ['Participant_ID', 'PHQ8_Binary']
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing columns in {split}: {missing_cols}")
        
        logger.info(
            f"Loaded {split} split: {len(df)} participants "
            f"(Depressed: {df['PHQ8_Binary'].sum()}, "
            f"Non-depressed: {(df['PHQ8_Binary']==0).sum()})"
        )
        
        self._cache[cache_key] = df
        return df
    
    def load_embeddings(
        self, 
        participant_id: int, 
        modality: str = 'audio'
    ) -> torch.Tensor:
        """
        Load pre-computed embeddings for a participant.
        
        Args:
            participant_id: DAIC participant ID
            modality: 'audio' or 'text'
            
        Returns:
            Tensor of shape (seq_len, embedding_dim)
        """
        if modality not in ['audio', 'text']:
            raise ValueError(f"Invalid modality: {modality}")
        
        emb_dir = self.root_dir / self.config['data']['embeddings'][f'{modality}_dir']
        emb_path = emb_dir / f"{participant_id}_embeddings.pt"
        
        if not emb_path.exists():
            raise FileNotFoundError(
                f"Embedding not found: {emb_path}\n"
                f"Participant {participant_id} may be missing {modality} data"
            )
        
        embedding = torch.load(emb_path, map_location='cpu')
        return embedding
    
    def load_transcript(self) -> pd.DataFrame:
        """
        Load full transcript data.
        
        Returns:
            DataFrame with columns: participant_id, speaker, value, question_type
        """
        cache_key = "transcript"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        transcript_path = self.root_dir / self.config['data']['raw']['transcript']
        
        if not transcript_path.exists():
            raise FileNotFoundError(f"Transcript not found: {transcript_path}")
        
        df = pd.read_csv(transcript_path)
        logger.info(f"Loaded transcript: {len(df)} utterances")
        
        self._cache[cache_key] = df
        return df
    
    def get_participant_ids(self, split: str) -> List[int]:
        """Get list of participant IDs for a given split"""
        df = self.load_split(split)
        return df['Participant_ID'].tolist()
    
    def validate_data_availability(self, split: str = 'train') -> Dict:
        """
        Check data availability for a split.
        
        Returns:
            {
                'total': int,
                'missing_audio': List[int],
                'missing_text': List[int],
                'complete': int
            }
        """
        pids = self.get_participant_ids(split)
        
        missing_audio = []
        missing_text = []
        
        for pid in pids:
            try:
                self.load_embeddings(pid, 'audio')
            except FileNotFoundError:
                missing_audio.append(pid)
            
            try:
                self.load_embeddings(pid, 'text')
            except FileNotFoundError:
                missing_text.append(pid)
        
        return {
            'total': len(pids),
            'missing_audio': missing_audio,
            'missing_text': missing_text,
            'complete': len(pids) - len(set(missing_audio) | set(missing_text))
        }
    
    def _resolve_env_vars(self, path_str: str) -> str:
        """Resolve ${VAR:default} syntax in paths"""
        import re
        
        pattern = r'\$\{(\w+):([^}]+)\}'
        
        def replacer(match):
            var_name = match.group(1)
            default_value = match.group(2)
            return os.getenv(var_name, default_value)
        
        return re.sub(pattern, replacer, path_str)


class TranscriptProcessor:
    """
    Process DAIC-WOZ transcripts for feature extraction.
    
    Handles:
    - Speaker separation (Ellie vs Participant)
    - Question type extraction
    - Text normalization
    """
    
    def __init__(self, transcript_df: pd.DataFrame):
        self.df = transcript_df
    
    def get_participant_text(self, participant_id: int) -> str:
        """Extract all participant utterances as single string"""
        mask = (
            (self.df['participant_id'] == participant_id) &
            (self.df['speaker'] == 'Participant')
        )
        
        utterances = self.df[mask]['value'].fillna("").tolist()
        return " ".join(utterances)
    
    def get_question_type_distribution(self, participant_id: int) -> pd.Series:
        """Get distribution of question types for a participant"""
        mask = self.df['participant_id'] == participant_id
        return self.df[mask]['question_type'].value_counts()
    
    def filter_by_question_type(
        self, 
        participant_id: int, 
        question_type: int
    ) -> pd.DataFrame:
        """Get all exchanges for a specific question type"""
        mask = (
            (self.df['participant_id'] == participant_id) &
            (self.df['question_type'] == question_type)
        )
        return self.df[mask]
