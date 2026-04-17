import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple

class TweedieLoss(nn.Module):
    """
    Log-Space Tweedie Negative Log-Likelihood.
    Optimized for FP16 training on A100 to prevent gradient overflow.
    
    The Tweedie distribution is a member of the exponential dispersion models
    and is particularly effective for modeling data with a cluster of 
    zeros and a continuous positive tail (1 < rho < 2).
    """
    def __init__(self, rho: float = 1.5, eps: float = 1e-10):
        super().__init__()
        if not (1 < rho < 2):
            raise ValueError("Tweedie rho must be between 1 and 2.")
        self.rho = rho
        self.eps = eps

    def forward(self, log_y_pred: Tensor, y_true: Tensor) -> Tensor:
        """
        Args:
            log_y_pred: Raw model output (log-space predictions).
            y_true: Ground truth sales values.
        """
        # deviance = -y * exp(log_pred * (1-rho)) / (1-rho) + exp(log_pred * (2-rho)) / (2-rho)
        rho = self.rho
        
        # Part 1: -y * exp(μ^(1-ρ)) / (1-ρ)
        part_1 = -y_true * torch.exp(log_y_pred * (1 - rho)) / (1 - rho)
        
        # Part 2: exp(μ^(2-ρ)) / (2-ρ)
        part_2 = torch.exp(log_y_pred * (2 - rho)) / (2 - rho)
        
        loss = part_1 + part_2
        return torch.mean(loss)

class M5SupremeLoss(nn.Module):
    """
    Research-Grade Composite Loss for M5 Forecasting.
    
    This loss function bridges the gap between point estimation (volume)
    and structural forecasting (sparsity) while maintaining alignment 
    with the WRMSSE metric.
    """
    def __init__(self, 
                 rho: float = 1.5, 
                 zi_weight: float = 0.2, 
                 wrmse_weight: float = 0.5):
        super().__init__()
        self.tweedie = TweedieLoss(rho=rho)
        self.zi_weight = zi_weight
        self.wrmse_weight = wrmse_weight
        self.eps = 1e-10

    def forward(self, 
                log_y_pred: Tensor, 
                y_true: Tensor, 
                prob_zero_logits: Optional[Tensor] = None,
                series_weights: Optional[Tensor] = None,
                series_scale: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            log_y_pred: Shape [N, 28] - Predicted sales in log-space.
            y_true: Shape [N, 28] - Actual sales.
            prob_zero_logits: Shape [N, 28] - Logits from ZI-GNN head.
            series_weights: M5 series weights for WRMSSE alignment.
            series_scale: Historical scaling factor for WRMSSE.
        """
        # 1. Volume Loss: Tweedie (Primary driver for point accuracy)
        # We use log-space Tweedie to handle high-variance demand spikes.
        t_loss = self.tweedie(log_y_pred, y_true)
        
        total_loss = t_loss
        
        # 2. Sparsity Loss: Zero-Inflation (Handles the 'Intermittent' problem)
        if prob_zero_logits is not None:
            # target is 1 if actual sale is 0, else 0
            is_zero_target = (y_true < self.eps).float()
            zi_loss = F.binary_cross_entropy_with_logits(prob_zero_logits, is_zero_target)
            total_loss += self.zi_weight * zi_loss
            
        # 3. Structural Loss: Weighted RMSE (Differentiable Proxy for WRMSSE)
        if series_weights is not None and series_scale is not None:
            # exp(log_y_pred) to get back to original scale for RMSE
            y_hat = torch.exp(log_y_pred)
            
            # Squared Error per time step: [N, 28]
            se = (y_true - y_hat) ** 2
            
            # Weighted Root Mean Squared Scaled Error approximation:
            # We calculate RMSE first, then scale and weight.
            # Denominator is the pre-computed historical scale per series.
            mse_per_series = torch.mean(se, dim=1) # [N]
            rmse_scaled = torch.sqrt(mse_per_series / (series_scale + self.eps))
            
            # Weighted average across all series (N)
            w_rmse_loss = torch.sum(series_weights * rmse_scaled)
            
            total_loss += self.wrmse_weight * w_rmse_loss
            
        return total_loss

    @staticmethod
    def get_log_forecast(log_y_pred: Tensor, prob_zero_logits: Optional[Tensor] = None) -> Tensor:
        """
        Utility to generate the final expected forecast during inference.
        Final = exp(log_volume) * (1 - prob_zero)
        """
        volume = torch.exp(log_y_pred)
        if prob_zero_logits is not None:
            prob_not_zero = 1.0 - torch.sigmoid(prob_zero_logits)
            return volume * prob_not_zero
        return volume