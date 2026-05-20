"""
Custom Loss Functions for Depression Detection

Implements PHQ-weighted losses that prioritize high-score (depressed) samples.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PHQWeightedMSE(nn.Module):
    """
    Weighted MSE loss with higher weights for high PHQ scores.
    
    Clinical rationale:
    - False negatives (missing depression) are more critical than false positives
    - Higher PHQ scores should have higher loss penalties
    
    Args:
        phq_max: Maximum PHQ score (default: 24.0)
        alpha: Weight exponent (higher = more emphasis on high scores)
    """
    
    def __init__(self, phq_max: float = 24.0, alpha: float = 2.0):
        super().__init__()
        self.phq_max = phq_max
        self.alpha = alpha
    
    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            preds: Predicted normalized scores [0, 1]
            targets: Target normalized scores [0, 1]
            
        Returns:
            Weighted MSE loss
        """
        # De-normalize to PHQ scale for weight calculation
        phq = targets * self.phq_max
        
        # Weight increases exponentially with PHQ score
        # Example: PHQ=0 → weight=1.0, PHQ=24 → weight=2.0 (alpha=2)
        weight = 1.0 + (phq / self.phq_max) ** self.alpha
        
        # Weighted squared error
        squared_error = (preds - targets) ** 2
        weighted_loss = (weight * squared_error).mean()
        
        return weighted_loss


class HuberLossWeighted(nn.Module):
    """
    Weighted Huber loss - robust to outliers while emphasizing high PHQ scores.
    
    Combines:
    - L2 loss for small errors (smooth gradients)
    - L1 loss for large errors (outlier robustness)
    - PHQ-based weighting (clinical priority)
    
    Args:
        delta: Threshold for L1/L2 transition (default: 1.0)
        phq_max: Maximum PHQ score (default: 24.0)
        alpha: Weight exponent (default: 1.5)
    """
    
    def __init__(self, delta: float = 1.0, phq_max: float = 24.0, alpha: float = 1.5):
        super().__init__()
        self.delta = delta
        self.phq_max = phq_max
        self.alpha = alpha
    
    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            preds: Predicted normalized scores [0, 1]
            targets: Target normalized scores [0, 1]
            
        Returns:
            Weighted Huber loss
        """
        phq = targets * self.phq_max
        weight = 1.0 + (phq / self.phq_max) ** self.alpha
        
        diff = torch.abs(preds - targets)
        
        # Huber loss: quadratic for small errors, linear for large
        huber = torch.where(
            diff < self.delta,
            0.5 * diff ** 2,
            self.delta * (diff - 0.5 * self.delta)
        )
        
        weighted_loss = (weight * huber).mean()
        
        return weighted_loss


class FocalMSELoss(nn.Module):
    """
    Focal MSE loss - automatically focuses on hard-to-predict samples.
    
    Inspired by Focal Loss for object detection, adapted for regression.
    
    Args:
        gamma: Focusing parameter (higher = more focus on hard samples)
        phq_max: Maximum PHQ score
    """
    
    def __init__(self, gamma: float = 2.0, phq_max: float = 24.0):
        super().__init__()
        self.gamma = gamma
        self.phq_max = phq_max
    
    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            preds: Predicted normalized scores
            targets: Target normalized scores
            
        Returns:
            Focal MSE loss
        """
        squared_error = (preds - targets) ** 2
        
        # Focal weight: higher for larger errors
        focal_weight = (squared_error + 1e-6) ** self.gamma
        
        loss = (focal_weight * squared_error).mean()
        
        return loss


def get_loss_function(loss_type: str, config: dict) -> nn.Module:
    """
    Factory function to create loss function from config.
    
    Args:
        loss_type: Loss function type ('mse', 'weighted_mse', 'huber', 'focal')
        config: Configuration dictionary
        
    Returns:
        Loss function module
        
    Raises:
        ValueError: If loss_type is not recognized
    """
    phq_max = config['data']['phq_max']
    alpha = config['training'].get('loss_alpha', 2.0)
    
    if loss_type == 'mse':
        return nn.MSELoss()
    
    elif loss_type == 'weighted_mse':
        return PHQWeightedMSE(phq_max=phq_max, alpha=alpha)
    
    elif loss_type == 'huber':
        return HuberLossWeighted(phq_max=phq_max, alpha=alpha)
    
    elif loss_type == 'focal':
        return FocalMSELoss(gamma=alpha, phq_max=phq_max)
    
    else:
        raise ValueError(
            f"Unknown loss type: {loss_type}. "
            f"Choose from: mse, weighted_mse, huber, focal"
        )
