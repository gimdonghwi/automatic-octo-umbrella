"""
Dataset Builder - Main Preprocessing Pipeline

Orchestrates:
- Transcript parsing
- Utterance segmentation
- Audio feature extraction
- Dataset assembly
"""

import os
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict
from tqdm import tqdm
import logging

from .transcript_parser import TranscriptParser, QuestionTypeAnalyzer
from .utterance_segmenter import UtteranceSegmenter
from .audio_processor import AudioFeatureExtractor, AudioLoader

logger = logging.getLogger(__name__)


class DatasetBuilder:
    """
    Main preprocessing pipeline for DAIC-WOZ dataset.
    
    Workflow:
    1. Load metadata
    2. For each participant:
       a. Parse transcript
       b. Segment utterances
       c. Load and denoise audio
       d. Extract features
       e. Assemble sequence
    3. Save dataset
    
    Usage:
        builder = DatasetBuilder(config)
        dataset = builder.build_dataset()
        builder.save_dataset(dataset, output_path)
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.base_path = self._resolve_base_path()
        
        self.transcript_parser = TranscriptParser(config, self.base_path)
        self.segmenter = UtteranceSegmenter(config)
        self.audio_extractor = AudioFeatureExtractor(config)
        self.audio_loader = AudioLoader(config)
        self.qtype_analyzer = QuestionTypeAnalyzer(config)
        
        self.stats = {
            'processed': 0,
            'failed': 0,
            'total_utterances': 0,
            'q_type_counts': defaultdict(int)
        }
        self.failure_log = defaultdict(int)
    
    def _resolve_base_path(self) -> Path:
        """Resolve base path with environment variable support"""
        import re
        
        path_str = self.config['data']['base_path']
        
        pattern = r'\$\{(\w+):([^}]+)\}'
        
        def replacer(match):
            var_name = match.group(1)
            default_value = match.group(2)
            return os.getenv(var_name, default_value)
        
        resolved = re.sub(pattern, replacer, path_str)
        path = Path(resolved)
        
        if not path.exists():
            raise FileNotFoundError(
                f"Data path not found: {path}\n"
                f"Set DAIC_ROOT environment variable or update config."
            )
        
        logger.info(f"Data root: {path}")
        return path
    
    def load_metadata(self) -> pd.DataFrame:
        """
        Load and filter participant metadata.
        
        Returns:
            DataFrame with Participant_ID and Binary label
        """
        meta_path = self.base_path / "metadataset.csv"
        
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata not found: {meta_path}")
        
        meta = pd.read_csv(meta_path)
        meta['Participant_ID'] = meta['Participant_ID'].astype(str)
        
        exception_ids = self.config['data']['exception_ids']
        meta = meta[~meta['Participant_ID'].isin(exception_ids)].reset_index(drop=True)
        
        logger.info(f"Loaded metadata for {len(meta)} participants")
        return meta
    
    def process_participant(
        self, 
        participant_id: str, 
        label: int
    ) -> Optional[Dict]:
        """
        Process a single participant.
        
        Args:
            participant_id: Participant ID
            label: Binary depression label (0 or 1)
            
        Returns:
            Participant data dictionary or None if processing fails
        """
        try:
            utterances = self.transcript_parser.parse_participant(participant_id)
            if not utterances:
                self.failure_log['no_utterances'] += 1
                return None
            
            utterances = self.segmenter.process_utterances(utterances)
            if not utterances:
                self.failure_log['empty_after_segmentation'] += 1
                return None
            
            audio_path = self.base_path / f"{participant_id}_P" / f"{participant_id}_AUDIO.wav"
            audio = self.audio_loader.load_and_denoise(str(audio_path))
            if audio is None:
                self.failure_log['audio_load_error'] += 1
                return None
            
            sequence = []
            
            for utt in utterances:
                audio_segment = self.audio_loader.extract_segment(
                    audio, 
                    utt['start'], 
                    utt['end']
                )
                
                if audio_segment is None:
                    continue
                
                wavlm_feat, wav2vec_feat = self.audio_extractor.extract_features(audio_segment)
                
                if wavlm_feat is None or wav2vec_feat is None:
                    continue
                
                sequence.append({
                    'wavlm': wavlm_feat,
                    'wav2vec': wav2vec_feat,
                    'text': utt['text'],
                    'q_type': utt['q_type'],
                    'q_type_id': utt['q_type_id'],
                    'timestamp': utt['start'],
                    'duration': utt['duration']
                })
                
                self.stats['q_type_counts'][utt['q_type']] += 1
            
            if not sequence:
                self.failure_log['empty_sequence'] += 1
                return None
            
            self.stats['total_utterances'] += len(sequence)
            
            return {
                'label': label,
                'sequence': sequence,
                'num_utterances': len(sequence),
                'total_duration': sequence[-1]['timestamp'] + sequence[-1]['duration'],
                'speaking_time': sum([u['duration'] for u in sequence])
            }
            
        except Exception as e:
            logger.error(f"Failed to process P{participant_id}: {e}")
            self.failure_log['unexpected_error'] += 1
            return None
    
    def build_dataset(self, checkpoint_interval: Optional[int] = None) -> Dict:
        """
        Build complete dataset from all participants.
        
        Args:
            checkpoint_interval: Save checkpoint every N participants (optional)
            
        Returns:
            Dictionary mapping participant_id to data
        """
        meta = self.load_metadata()
        dataset = {}
        
        show_progress = self.config['processing'].get('show_progress', True)
        
        iterator = tqdm(
            meta.iterrows(), 
            total=len(meta), 
            desc="Processing Participants",
            disable=not show_progress
        )
        
        for idx, row in iterator:
            pid = str(row['Participant_ID'])
            label = int(row['Binary'])
            
            participant_data = self.process_participant(pid, label)
            
            if participant_data:
                dataset[pid] = participant_data
                self.stats['processed'] += 1
            else:
                self.stats['failed'] += 1
            
            if checkpoint_interval and (idx + 1) % checkpoint_interval == 0:
                self._save_checkpoint(dataset, idx + 1)
        
        return dataset
    
    def _save_checkpoint(self, dataset: Dict, iteration: int):
        """Save intermediate checkpoint"""
        checkpoint_path = self.base_path / f"checkpoint_{iteration}.pkl"
        
        try:
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Checkpoint saved: {checkpoint_path}")
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")
    
    def save_dataset(self, dataset: Dict, output_path: Optional[str] = None):
        """
        Save dataset to pickle file.
        
        Args:
            dataset: Processed dataset dictionary
            output_path: Output file path (optional, uses config if None)
        """
        if output_path is None:
            output_filename = self.config['data']['output_filename']
            output_path = self.base_path / output_filename
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(output_path, 'wb') as f:
                pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            size_mb = output_path.stat().st_size / (1024 * 1024)
            
            logger.info(f"✓ Dataset saved: {output_path}")
            logger.info(f"  File size: {size_mb:.2f} MB")
            
        except Exception as e:
            logger.error(f"Failed to save dataset: {e}")
            raise
    
    def print_statistics(self):
        print("\n" + "="*70)
        print("✅ Preprocessing Complete!")
        print("="*70)
        print(f"Successfully processed: {self.stats['processed']} participants")
        print(f"Failed to process: {self.stats['failed']} participants")
        
        if self.failure_log:
            print("\nFailure reasons:")
            for reason, count in sorted(self.failure_log.items(), key=lambda x: -x[1]):
                print(f"  - {reason}: {count}")
        
        print(f"\nTotal valid utterances: {self.stats['total_utterances']}")
        
        if self.stats['total_utterances'] > 0:
            print("\nQuestion Type Distribution:")
            sorted_qtypes = sorted(
                self.stats['q_type_counts'].items(), 
                key=lambda x: -x[1]
            )
            
            for q_type, count in sorted_qtypes:
                percentage = (count / self.stats['total_utterances']) * 100
                print(f"  - {q_type:15s}: {count:5d} utterances ({percentage:5.1f}%)")
    
    def get_sample_data(self, dataset: Dict) -> Dict:
        """
        Get sample data for verification.
        
        Args:
            dataset: Processed dataset
            
        Returns:
            Sample participant data
        """
        if not dataset:
            return {}
        
        sample_pid = list(dataset.keys())[0]
        return {
            'participant_id': sample_pid,
            'data': dataset[sample_pid]
        }
