"""
Audio Feature Extraction Module

Handles:
- Wav2Vec 2.0 feature extraction
- WavLM feature extraction  
- Audio loading and denoising
- Model management
"""

import torch
import numpy as np
import librosa
import noisereduce as nr
from transformers import Wav2Vec2Processor, Wav2Vec2Model, WavLMModel
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class AudioFeatureExtractor:
    #Extract audio features using Wav2Vec and WavLM
    
    def __init__(self, config: dict):
        self.config = config
        self.sr = config['audio']['sample_rate']
        self.device = self._setup_device()
        
        self.processor = None
        self.wav2vec_model = None
        self.wavlm_model = None
        
        self._load_models()
    
    def _setup_device(self) -> torch.device:
        """Auto-detect or use configured device"""
        device_name = self.config['models'].get('device', 'cuda')
        
        if device_name == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA is not available")
            return torch.device('cpu')
        
        device = torch.device(device_name if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        return device
    
    def _load_models(self):
        logger.info("Loading audio feature extraction models...")
        
        wav2vec_name = self.config['models']['wav2vec']['name']
        wavlm_name = self.config['models']['wavlm']['name']
        
        try:
            self.processor = Wav2Vec2Processor.from_pretrained(wav2vec_name)
            
            self.wav2vec_model = Wav2Vec2Model.from_pretrained(wav2vec_name).to(self.device)
            self.wav2vec_model.eval()
            logger.info(f"✓ Wav2Vec 2.0 loaded: {wav2vec_name}")
            
            self.wavlm_model = WavLMModel.from_pretrained(wavlm_name).to(self.device)
            self.wavlm_model.eval()
            logger.info(f"✓ WavLM loaded: {wavlm_name}")
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise
    
    def extract_wav2vec_features(self, audio: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract Wav2Vec 2.0 features from audio segment.
        
        Args:
            audio: Audio waveform (numpy array)
            
        Returns:
            768-dimensional embedding or None if extraction fails
        """
        try:
            if len(audio) < self.sr * 0.3:
                logger.debug("Audio too short for Wav2Vec extraction")
                return None
            
            audio_normalized = audio / (np.max(np.abs(audio)) + 1e-8)
            
            inputs = self.processor(
                audio_normalized,
                sampling_rate=self.sr,
                return_tensors="pt",
                padding=True
            )
            
            input_values = inputs.input_values.to(self.device)
            
            with torch.no_grad():
                outputs = self.wav2vec_model(input_values, output_hidden_states=True)
                
                hidden_layer = self.config['models']['wav2vec']['hidden_layer']
                hidden_states = outputs.hidden_states[hidden_layer].squeeze(0)
            
            embedding = torch.mean(hidden_states, dim=0).cpu().numpy()
            
            if not np.isfinite(embedding).all():
                logger.warning("Non-finite values in Wav2Vec embedding")
                return None
            
            return embedding
            
        except Exception as e:
            logger.debug(f"Wav2Vec extraction failed: {e}")
            return None
    
    def extract_wavlm_features(self, audio: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract WavLM features from audio segment.
        
        Args:
            audio: Audio waveform (numpy array)
            
        Returns:
            768-dimensional embedding or None if extraction fails
        """
        try:
            if len(audio) < self.sr * 0.3:
                logger.debug("Audio too short for WavLM extraction")
                return None
            
            audio_normalized = audio / (np.max(np.abs(audio)) + 1e-8)
            
            inputs = self.processor(
                audio_normalized,
                sampling_rate=self.sr,
                return_tensors="pt",
                padding=True
            )
            
            input_values = inputs.input_values.to(self.device)
            
            with torch.no_grad():
                outputs = self.wavlm_model(input_values, output_hidden_states=True)
                
                hidden_layer = self.config['models']['wavlm']['hidden_layer']
                hidden_states = outputs.hidden_states[hidden_layer].squeeze(0)
            
            embedding = torch.mean(hidden_states, dim=0).cpu().numpy()
            
            if not np.isfinite(embedding).all():
                logger.warning("Non-finite values in WavLM embedding")
                return None
            
            return embedding
            
        except Exception as e:
            logger.debug(f"WavLM extraction failed: {e}")
            return None
    
    def extract_features(
        self, 
        audio: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Extract both WavLM and Wav2Vec features.
        
        Returns:
            (wavlm_embedding, wav2vec_embedding) or (None, None) if both fail
        """
        wavlm_feat = self.extract_wavlm_features(audio)
        wav2vec_feat = self.extract_wav2vec_features(audio)
        
        return wavlm_feat, wav2vec_feat


class AudioLoader:
    """
    Handles audio file loading and preprocessing.
    
    Includes:
    - File loading with librosa
    - Noise reduction
    - Resampling
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.sr = config['audio']['sample_rate']
        self.noise_config = config['audio']['noise_reduction']
    
    def load_and_denoise(self, audio_path: str) -> Optional[np.ndarray]:
        """
        Load audio file and apply noise reduction.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Denoised audio waveform or None if loading fails
        """
        try:
            audio, _ = librosa.load(audio_path, sr=self.sr)
            
            if self.noise_config.get('enabled', True):
                audio = nr.reduce_noise(
                    y=audio,
                    sr=self.sr,
                    stationary=self.noise_config['stationary'],
                    prop_decrease=self.noise_config['prop_decrease']
                )
            
            return audio
            
        except Exception as e:
            logger.error(f"Failed to load audio from {audio_path}: {e}")
            return None
    
    def extract_segment(
        self, 
        audio: np.ndarray, 
        start_sec: float, 
        end_sec: float
    ) -> Optional[np.ndarray]:
        """
        Extract audio segment from full waveform.
        
        Args:
            audio: Full audio waveform
            start_sec: Start time in seconds
            end_sec: End time in seconds
            
        Returns:
            Audio segment or None if invalid range
        """
        start_sample = int(start_sec * self.sr)
        end_sample = int(end_sec * self.sr)
        
        if not (0 <= start_sample < end_sample <= len(audio)):
            logger.warning(f"Invalid segment range: [{start_sec:.2f}, {end_sec:.2f}]")
            return None
        
        return audio[start_sample:end_sample]
