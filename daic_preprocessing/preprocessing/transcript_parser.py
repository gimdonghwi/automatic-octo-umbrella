"""
Transcript Parsing Module

Handles:
- TRANSCRIPT.csv file parsing
- Question type normalization
- Speaker identification
- Utterance extraction
"""

import pandas as pd
import os
from typing import List, Dict, Optional
import logging
import re

logger = logging.getLogger(__name__)


class TranscriptParser:
    """
    Parses DAIC-WOZ transcript files and extracts participant utterances.
    
    Usage:
        parser = TranscriptParser(config, base_path)
        utterances = parser.parse_participant(participant_id)
    """
    
    def __init__(self, config: dict, base_path: str):
        self.config = config
        self.base_path = base_path
        self.q_type_mapping = config['question_types']['categories']
        self.q_type_simplification = config['question_types']['mapping']
    
    def _normalize_question_type(self, q_type: str) -> str:
        """
        Normalize and simplify question type string.
        
        Args:
            q_type: Original question type string
            
        Returns:
            Simplified question type category
        """
        if pd.isna(q_type) or not q_type or q_type.strip() == '':
            return 'other'
        
        q_type_clean = str(q_type).strip().lower()
        q_type_clean = re.sub(r'[^a-z0-9/\s-]', '', q_type_clean)
        
        return self.q_type_simplification.get(q_type_clean, 'other')
    
    def parse_participant(self, participant_id: str) -> List[Dict]:
        """
        Parse transcript file for a participant.
        
        Args:
            participant_id: Participant ID (e.g., "300")
            
        Returns:
            List of utterance dictionaries with keys:
            - speaker: 'Ellie' or 'Participant'
            - text: Utterance text
            - start: Start time (seconds)
            - end: End time (seconds)
            - duration: Duration (seconds)
            - q_type: Simplified question type
        """
        transcript_path = os.path.join(
            self.base_path,
            f"{participant_id}_P",
            f"{participant_id}_TRANSCRIPT.csv"
        )
        
        if not os.path.exists(transcript_path):
            logger.warning(f"Transcript not found: {transcript_path}")
            return []
        
        try:
            df = pd.read_csv(transcript_path, delimiter='\t')
            
            required_cols = ['speaker', 'value', 'start_time', 'stop_time']
            missing_cols = set(required_cols) - set(df.columns)
            if missing_cols:
                logger.error(f"Missing columns in transcript: {missing_cols}")
                return []
            
            utterances = []
            
            for _, row in df.iterrows():
                if pd.isna(row['value']) or str(row['value']).strip() == '':
                    continue
                
                if row['speaker'] != 'Participant':
                    continue
                
                start = float(row['start_time'])
                end = float(row['stop_time'])
                duration = end - start
                
                if duration < self.config['audio']['min_utterance_duration']:
                    continue
                
                q_type_raw = row.get('question_type', 'other')
                q_type = self._normalize_question_type(q_type_raw)
                
                utterances.append({
                    'speaker': row['speaker'],
                    'text': str(row['value']).strip(),
                    'start': start,
                    'end': end,
                    'duration': duration,
                    'q_type': q_type,
                    'q_type_id': self.q_type_mapping.get(q_type, 4)
                })
            
            logger.debug(f"Parsed {len(utterances)} utterances for P{participant_id}")
            return utterances
            
        except Exception as e:
            logger.error(f"Failed to parse transcript for P{participant_id}: {e}")
            return []
    
    def validate_transcript(self, participant_id: str) -> bool:
        """
        Check if transcript file exists and is readable.
        
        Args:
            participant_id: Participant ID
            
        Returns:
            True if transcript is valid, False otherwise
        """
        transcript_path = os.path.join(
            self.base_path,
            f"{participant_id}_P",
            f"{participant_id}_TRANSCRIPT.csv"
        )
        
        if not os.path.exists(transcript_path):
            return False
        
        try:
            df = pd.read_csv(transcript_path, delimiter='\t')
            required_cols = ['speaker', 'value', 'start_time', 'stop_time']
            return all(col in df.columns for col in required_cols)
        except:
            return False


class QuestionTypeAnalyzer:
    """
    Analyzes question type distributions and statistics.
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.q_type_mapping = config['question_types']['categories']
    
    def get_distribution(self, utterances: List[Dict]) -> Dict[str, int]:
        """
        Get question type distribution from utterances.
        
        Args:
            utterances: List of utterance dictionaries
            
        Returns:
            Dictionary mapping q_type to count
        """
        distribution = {q_type: 0 for q_type in self.q_type_mapping.keys()}
        
        for utt in utterances:
            q_type = utt['q_type']
            distribution[q_type] = distribution.get(q_type, 0) + 1
        
        return distribution
    
    def get_statistics(self, all_utterances: List[List[Dict]]) -> Dict:
        """
        Calculate overall question type statistics.
        
        Args:
            all_utterances: List of utterance lists (one per participant)
            
        Returns:
            Statistics dictionary
        """
        total_dist = {q_type: 0 for q_type in self.q_type_mapping.keys()}
        total_utterances = 0
        
        for utterances in all_utterances:
            dist = self.get_distribution(utterances)
            for q_type, count in dist.items():
                total_dist[q_type] += count
                total_utterances += count
        
        percentages = {
            q_type: (count / total_utterances * 100) if total_utterances > 0 else 0
            for q_type, count in total_dist.items()
        }
        
        return {
            'distribution': total_dist,
            'percentages': percentages,
            'total_utterances': total_utterances
        }
