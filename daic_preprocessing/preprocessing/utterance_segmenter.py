"""
Utterance Segmentation Module

Handles:
- Merging consecutive participant utterances
- Splitting overly long utterances
- Utterance validation
"""

import numpy as np
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class UtteranceSegmenter:
    """
    Segments participant utterances by merging close ones and splitting long ones.
    
    Strategy:
    1. Merge consecutive participant utterances if pause < threshold
    2. Split utterances exceeding max duration
    3. Filter out utterances below minimum duration
    
    Usage:
        segmenter = UtteranceSegmenter(config)
        processed = segmenter.process_utterances(raw_utterances)
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.max_pause = config['audio']['max_pause_for_merge']
        self.max_duration = config['audio']['max_utterance_duration']
        self.min_duration = config['audio']['min_utterance_duration']
    
    def merge_consecutive_utterances(self, utterances: List[Dict]) -> List[Dict]:
        """
        Merge consecutive participant utterances with short pauses.
        
        Args:
            utterances: List of raw utterances (must be sorted by start time)
            
        Returns:
            List of merged utterances
        """
        if not utterances:
            return []
        
        utterances = sorted(utterances, key=lambda x: x['start'])
        
        merged = []
        current = utterances[0].copy()
        
        for next_utt in utterances[1:]:
            pause = next_utt['start'] - current['end']
            
            if pause <= self.max_pause:
                current['text'] = f"{current['text']} {next_utt['text']}"
                current['end'] = next_utt['end']
                current['duration'] = current['end'] - current['start']
                
                if next_utt['q_type'] == 'clinical':
                    current['q_type'] = 'clinical'
                    current['q_type_id'] = next_utt['q_type_id']
            else:
                merged.append(current)
                current = next_utt.copy()
        
        merged.append(current)
        
        logger.debug(
            f"Merged {len(utterances)} utterances into {len(merged)} "
            f"(reduction: {len(utterances) - len(merged)})"
        )
        
        return merged
    
    def split_long_utterances(self, utterances: List[Dict]) -> List[Dict]:
        """
        Split utterances exceeding maximum duration.
        
        Args:
            utterances: List of utterances
            
        Returns:
            List with long utterances split into chunks
        """
        final_utterances = []
        num_splits = 0
        
        for utt in utterances:
            if utt['duration'] <= self.max_duration:
                final_utterances.append(utt)
            else:
                num_chunks = int(np.ceil(utt['duration'] / self.max_duration))
                chunk_duration = utt['duration'] / num_chunks
                
                for i in range(num_chunks):
                    chunk_start = utt['start'] + i * chunk_duration
                    chunk_end = min(chunk_start + chunk_duration, utt['end'])
                    
                    chunk = utt.copy()
                    chunk.update({
                        'start': chunk_start,
                        'end': chunk_end,
                        'duration': chunk_end - chunk_start
                    })
                    
                    final_utterances.append(chunk)
                
                num_splits += (num_chunks - 1)
        
        if num_splits > 0:
            logger.debug(f"Split {num_splits} long utterances")
        
        return final_utterances
    
    def filter_short_utterances(self, utterances: List[Dict]) -> List[Dict]:
        """
        Remove utterances below minimum duration threshold.
        
        Args:
            utterances: List of utterances
            
        Returns:
            Filtered list
        """
        filtered = [
            utt for utt in utterances 
            if utt['duration'] >= self.min_duration
        ]
        
        removed = len(utterances) - len(filtered)
        if removed > 0:
            logger.debug(f"Filtered out {removed} short utterances")
        
        return filtered
    
    def process_utterances(self, utterances: List[Dict]) -> List[Dict]:
        """
        Full utterance processing pipeline.
        
        Steps:
        1. Merge consecutive utterances with short pauses
        2. Split overly long utterances
        3. Filter short utterances
        
        Args:
            utterances: Raw utterances from transcript
            
        Returns:
            Processed utterances ready for feature extraction
        """
        if not utterances:
            return []
        
        utterances = sorted(utterances, key=lambda x: x['start'])
        
        merged = self.merge_consecutive_utterances(utterances)
        
        split = self.split_long_utterances(merged)
        
        filtered = self.filter_short_utterances(split)
        
        logger.debug(
            f"Utterance processing: {len(utterances)} → "
            f"{len(merged)} (merged) → "
            f"{len(split)} (split) → "
            f"{len(filtered)} (filtered)"
        )
        
        return filtered
    
    def validate_utterances(self, utterances: List[Dict]) -> bool:
        """
        Validate utterances for consistency.
        
        Checks:
        - All have required fields
        - Timestamps are valid
        - No overlapping utterances
        
        Args:
            utterances: List of utterances
            
        Returns:
            True if valid, False otherwise
        """
        if not utterances:
            return False
        
        required_fields = ['start', 'end', 'duration', 'text', 'q_type']
        
        for utt in utterances:
            if not all(field in utt for field in required_fields):
                logger.error("Utterance missing required fields")
                return False
            
            if utt['start'] >= utt['end']:
                logger.error(f"Invalid timestamp: start={utt['start']}, end={utt['end']}")
                return False
            
            if abs(utt['duration'] - (utt['end'] - utt['start'])) > 0.01:
                logger.error("Duration mismatch")
                return False
        
        utterances = sorted(utterances, key=lambda x: x['start'])
        for i in range(len(utterances) - 1):
            if utterances[i]['end'] > utterances[i+1]['start']:
                logger.warning("Overlapping utterances detected")
        
        return True
    
    def get_statistics(self, utterances: List[Dict]) -> Dict:
        """
        Calculate statistics for utterances.
        
        Args:
            utterances: List of utterances
            
        Returns:
            Statistics dictionary
        """
        if not utterances:
            return {
                'num_utterances': 0,
                'total_duration': 0,
                'mean_duration': 0,
                'median_duration': 0,
                'min_duration': 0,
                'max_duration': 0
            }
        
        durations = [utt['duration'] for utt in utterances]
        
        return {
            'num_utterances': len(utterances),
            'total_duration': sum(durations),
            'mean_duration': np.mean(durations),
            'median_duration': np.median(durations),
            'min_duration': min(durations),
            'max_duration': max(durations)
        }
