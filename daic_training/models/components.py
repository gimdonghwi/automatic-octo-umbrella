"""
Neural Network Components for Depression Detection

Modular components:
- Positional Encoding
- Gated Fusion (audio)
- Linguistic Encoder
- Modality Fusion
- Utterance Attention Pooling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for Transformer.
    
    Encodes temporal position in the interview sequence.
    """
    
    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq_len, d_model)
            
        Returns:
            Position-encoded tensor
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class GatedFusion(nn.Module):
    """
    Gated fusion for audio features (WavLM + Wav2Vec).
    
    Strategy:
    - WavLM: Primary features (full dimension)
    - Wav2Vec: Auxiliary features (compressed)
    - Gate: Learned modulation of auxiliary features
    
    Args:
        wavlm_dim: WavLM embedding dimension (768)
        wav2vec_dim: Wav2Vec embedding dimension (768)
        d_model: Target model dimension
        compression_ratio: Wav2Vec compression factor (4 or 8)
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        wavlm_dim: int,
        wav2vec_dim: int,
        d_model: int,
        compression_ratio: int = 4,
        dropout: float = 0.3
    ):
        super().__init__()
        self.d_model = d_model
        aux_dim = d_model // compression_ratio
        
        # WavLM projection (primary)
        self.wavlm_proj = nn.Sequential(
            nn.Linear(wavlm_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Wav2Vec compression (auxiliary)
        self.wav2vec_compress = nn.Sequential(
            nn.Linear(wav2vec_dim, aux_dim * 2),
            nn.LayerNorm(aux_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(aux_dim * 2, aux_dim)
        )
        
        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(d_model + aux_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Auxiliary projection
        self.aux_proj = nn.Linear(aux_dim, d_model)
        
        # Final combination
        self.combine = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    
    def forward(
        self, 
        wavlm: torch.Tensor, 
        wav2vec: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            wavlm: WavLM features (batch, seq_len, 768)
            wav2vec: Wav2Vec features (batch, seq_len, 768)
            
        Returns:
            (fused_features, gate_values)
        """
        # Project WavLM to d_model
        wavlm_main = self.wavlm_proj(wavlm)
        
        # Compress Wav2Vec
        wav2vec_aux = self.wav2vec_compress(wav2vec)
        
        # Compute gate values
        gate_input = torch.cat([wavlm_main, wav2vec_aux], dim=-1)
        gate_values = self.gate(gate_input)
        
        # Gate-modulated auxiliary features
        wav2vec_proj = self.aux_proj(wav2vec_aux)
        wav2vec_gated = wav2vec_proj * gate_values
        
        # Combine
        combined = torch.cat([wavlm_main, wav2vec_gated], dim=-1)
        output = self.combine(combined)
        
        return output, gate_values


class LinguisticEncoder(nn.Module):
    """
    Encodes linguistic features (TTR + Question Type).
    
    Args:
        ttr_dim: Enhanced TTR feature dimension (6)
        qtype_vocab_size: Number of question types (5)
        qtype_embed_dim: Question type embedding dimension
        d_model: Target model dimension
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        ttr_dim: int,
        qtype_vocab_size: int,
        qtype_embed_dim: int,
        d_model: int,
        dropout: float = 0.3
    ):
        super().__init__()
        
        # TTR projection
        self.ttr_proj = nn.Sequential(
            nn.Linear(ttr_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, d_model // 2)
        )
        
        # Question type embedding
        self.qtype_embed = nn.Embedding(qtype_vocab_size, qtype_embed_dim)
        self.qtype_proj = nn.Sequential(
            nn.Linear(qtype_embed_dim, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU()
        )
        
        # Combination
        self.combine = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    
    def forward(
        self, 
        ttr: torch.Tensor, 
        q_types: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            ttr: Enhanced TTR features (batch, seq_len, 6)
            q_types: Question type IDs (batch, seq_len)
            
        Returns:
            Encoded linguistic features (batch, seq_len, d_model)
        """
        ttr_feat = self.ttr_proj(ttr)
        
        qtype_emb = self.qtype_embed(q_types)
        qtype_feat = self.qtype_proj(qtype_emb)
        
        combined = torch.cat([ttr_feat, qtype_feat], dim=-1)
        output = self.combine(combined)
        
        return output


class ModalityFusion(nn.Module):
    """
    Attention-based fusion of audio and linguistic modalities.
    
    Learns adaptive weights for each modality per timestep.
    
    Args:
        d_model: Model dimension
        dropout: Dropout probability
    """
    
    def __init__(self, d_model: int, dropout: float = 0.3):
        super().__init__()
        
        # Modality attention
        self.modality_attention = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 2),
            nn.Softmax(dim=-1)
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    
    def forward(
        self, 
        audio_feat: torch.Tensor, 
        ling_feat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            audio_feat: Audio features (batch, seq_len, d_model)
            ling_feat: Linguistic features (batch, seq_len, d_model)
            
        Returns:
            (fused_features, modality_weights)
        """
        combined = torch.cat([audio_feat, ling_feat], dim=-1)
        
        # Compute modality weights (batch, seq_len, 2)
        modality_weights = self.modality_attention(combined)
        
        # Weighted sum
        fused = (
            modality_weights[:, :, 0:1] * audio_feat + 
            modality_weights[:, :, 1:2] * ling_feat
        )
        
        output = self.output_proj(fused)
        
        return output, modality_weights


class UtteranceAttentionPooling(nn.Module):
    """
    Attention-based pooling across utterances.
    
    Learns to weight utterances by their importance for depression detection.
    
    Args:
        d_model: Model dimension
        dropout: Dropout probability
    """
    
    def __init__(self, d_model: int, dropout: float = 0.3):
        super().__init__()
        
        self.attention = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input features (batch, seq_len, d_model)
            mask: Padding mask (batch, seq_len) - 1 for valid, 0 for padding
            
        Returns:
            (pooled_output, attention_weights)
        """
        # Compute attention scores
        attn_scores = self.attention(x).squeeze(-1)  # (batch, seq_len)
        
        # Mask padding positions
        attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        # Softmax
        attn_weights = F.softmax(attn_scores, dim=-1).unsqueeze(-1)  # (batch, seq_len, 1)
        
        # Weighted sum
        pooled = (attn_weights * x).sum(dim=1)  # (batch, d_model)
        
        return pooled, attn_weights.squeeze(-1)
