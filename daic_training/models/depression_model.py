"""
Depression Detection Model

Multimodal Transformer for PHQ score regression.

Architecture:
1. Audio Fusion (WavLM + Wav2Vec) → Gated features
2. Linguistic Encoding (TTR + Q-Type) → Linguistic features  
3. Modality Fusion → Combined features
4. Transformer Encoder → Temporal modeling
5. Attention Pooling → Sequence aggregation
6. Regression Head → PHQ score prediction
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple

from .components import (
    PositionalEncoding,
    GatedFusion,
    LinguisticEncoder,
    ModalityFusion,
    UtteranceAttentionPooling
)


class DepressionDetectionModel(nn.Module):
    """
    End-to-end model for depression detection from multimodal interview data.
    
    Args:
        config: Configuration dictionary containing model hyperparameters
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # Extract config
        data_cfg = config['data']
        model_cfg = config['model']
        
        wavlm_dim = data_cfg['wavlm_dim']
        wav2vec_dim = data_cfg['wav2vec_dim']
        ttr_dim = data_cfg['enhanced_ttr_dim']
        num_qtypes = data_cfg['num_qtypes']
        
        d_model = model_cfg['d_model']
        nhead = model_cfg['nhead']
        num_layers = model_cfg['num_encoder_layers']
        dim_feedforward = model_cfg['dim_feedforward']
        dropout = model_cfg['dropout']
        qtype_embed_dim = model_cfg['qtype_embed_dim']
        compression_ratio = model_cfg['aux_compression_ratio']
        max_len = model_cfg.get('max_seq_len', 500)
        
        # Components
        self.audio_fusion = GatedFusion(
            wavlm_dim=wavlm_dim,
            wav2vec_dim=wav2vec_dim,
            d_model=d_model,
            compression_ratio=compression_ratio,
            dropout=dropout
        )
        
        self.ling_encoder = LinguisticEncoder(
            ttr_dim=ttr_dim,
            qtype_vocab_size=num_qtypes,
            qtype_embed_dim=qtype_embed_dim,
            d_model=d_model,
            dropout=dropout
        )
        
        self.modality_fusion = ModalityFusion(
            d_model=d_model,
            dropout=dropout
        )
        
        self.pos_encoder = PositionalEncoding(
            d_model=d_model,
            max_len=max_len,
            dropout=dropout
        )
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Utterance pooling
        self.utterance_pooling = UtteranceAttentionPooling(
            d_model=d_model,
            dropout=dropout
        )
        
        # Regression head
        self.regressor = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()  # Output in [0, 1] range (normalized PHQ score)
        )
    
    def forward(
        self,
        wavlm: torch.Tensor,
        wav2vec: torch.Tensor,
        ttr: torch.Tensor,
        q_types: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            wavlm: WavLM embeddings (batch, seq_len, 768)
            wav2vec: Wav2Vec embeddings (batch, seq_len, 768)
            ttr: Enhanced TTR features (batch, seq_len, 6)
            q_types: Question type IDs (batch, seq_len)
            mask: Padding mask (batch, seq_len) - 1 for valid, 0 for padding
            
        Returns:
            Predicted normalized PHQ scores (batch, 1)
        """
        # 1. Audio fusion
        audio_feat, _ = self.audio_fusion(wavlm, wav2vec)
        
        # 2. Linguistic encoding
        ling_feat = self.ling_encoder(ttr, q_types)
        
        # 3. Modality fusion
        fused, _ = self.modality_fusion(audio_feat, ling_feat)
        
        # 4. Positional encoding
        fused = self.pos_encoder(fused)
        
        # 5. Transformer encoding
        padding_mask = (mask == 0)  # True for padding positions
        encoded = self.transformer(fused, src_key_padding_mask=padding_mask)
        
        # 6. Attention pooling
        pooled, _ = self.utterance_pooling(encoded, mask)
        
        # 7. Regression
        output = self.regressor(pooled)
        
        return output
    
    def forward_with_attention(
        self,
        wavlm: torch.Tensor,
        wav2vec: torch.Tensor,
        ttr: torch.Tensor,
        q_types: torch.Tensor,
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with attention weights for interpretability.
        
        Returns:
            (predictions, attention_dict)
            
        attention_dict contains:
        - audio_gate: Gate values for audio fusion
        - modality_weights: Weights for audio vs linguistic
        - utterance_attention: Importance of each utterance
        """
        # 1. Audio fusion
        audio_feat, audio_gate = self.audio_fusion(wavlm, wav2vec)
        
        # 2. Linguistic encoding
        ling_feat = self.ling_encoder(ttr, q_types)
        
        # 3. Modality fusion
        fused, modality_weights = self.modality_fusion(audio_feat, ling_feat)
        
        # 4. Positional encoding
        fused = self.pos_encoder(fused)
        
        # 5. Transformer encoding
        padding_mask = (mask == 0)
        encoded = self.transformer(fused, src_key_padding_mask=padding_mask)
        
        # 6. Attention pooling
        pooled, utterance_attention = self.utterance_pooling(encoded, mask)
        
        # 7. Regression
        output = self.regressor(pooled)
        
        attention_dict = {
            'audio_gate': audio_gate,
            'modality_weights': modality_weights,
            'utterance_attention': utterance_attention
        }
        
        return output, attention_dict
    
    def count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_summary(self) -> str:
        """Get model architecture summary"""
        total_params = self.count_parameters()
        
        summary = [
            "=" * 60,
            "Depression Detection Model Summary",
            "=" * 60,
            f"Total Parameters: {total_params:,}",
            "",
            "Architecture:",
            f"  - Audio Fusion: {self.audio_fusion.__class__.__name__}",
            f"  - Linguistic Encoder: {self.ling_encoder.__class__.__name__}",
            f"  - Modality Fusion: {self.modality_fusion.__class__.__name__}",
            f"  - Transformer Layers: {len(self.transformer.layers)}",
            f"  - Model Dimension: {self.transformer.layers[0].self_attn.embed_dim}",
            f"  - Attention Heads: {self.transformer.layers[0].self_attn.num_heads}",
            "=" * 60
        ]
        
        return "\n".join(summary)


def create_model(config: Dict) -> DepressionDetectionModel:
    """
    Factory function to create model from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Initialized model
    """
    model = DepressionDetectionModel(config)
    
    # Initialize weights (Xavier uniform for linear layers)
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0, std=0.02)
    
    model.apply(init_weights)
    
    return model
