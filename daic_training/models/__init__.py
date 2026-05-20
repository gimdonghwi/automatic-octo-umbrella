"""
Models Package

Neural network architectures for depression detection.
"""

from .depression_model import DepressionDetectionModel, create_model
from .losses import (
    PHQWeightedMSE,
    HuberLossWeighted,
    FocalMSELoss,
    get_loss_function
)
from .components import (
    PositionalEncoding,
    GatedFusion,
    LinguisticEncoder,
    ModalityFusion,
    UtteranceAttentionPooling
)

__all__ = [
    'DepressionDetectionModel',
    'create_model',
    'PHQWeightedMSE',
    'HuberLossWeighted',
    'FocalMSELoss',
    'get_loss_function',
    'PositionalEncoding',
    'GatedFusion',
    'LinguisticEncoder',
    'ModalityFusion',
    'UtteranceAttentionPooling'
]
