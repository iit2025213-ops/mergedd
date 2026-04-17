import torch
import torch.nn as nn
from torch import Tensor
import numpy as np

class WRMSSEMetric:
    """
    Research-Grade WRMSSE Evaluation Engine.
    
    This engine implements the M5 competition's primary metric. 
    It computes the Root Mean Squared Scaled Error across the 
    hierarchy and applies dollar-weighted importance.
    
    Optimized for: Vectorized execution on A100 GPU.
    """
    def __init__(self, 
                 weights: Tensor, 
                 scales: Tensor, 
                 device: torch.device):
        """
        Args:
            weights: The M5 dollar-weight for each series [N]
            scales: The historical scaling factor for each series [N]
                    (Average squared difference of consecutive days)
            device: The device for computation (cuda:0)
        """
        self.weights = weights.to(device)
        self.scales = scales.to(device)
        self.device = device
        self.eps = 1e-10

    @torch.no_grad()
    def compute(self, y_pred: Tensor, y_true: Tensor) -> float:
        """
        Calculates the WRMSSE score.
        
        Args:
            y_pred: Predicted sales [N, 28]
            y_true: Actual sales [N, 28]
        """
        # 1. Calculate Squared Error per time step
        # [N, 28]
        squared_error = (y_true - y_pred) ** 2
        
        # 2. Compute Mean Squared Error across the 28-day horizon
        # [N]
        mse_per_series = torch.mean(squared_error, dim=1)
        
        # 3. Calculate RMSSE (Root Mean Squared Scaled Error)
        # Scaled Error = MSE / Scale
        # RMSSE = sqrt(Scaled Error)
        rmsse = torch.sqrt(mse_per_series / (self.scales + self.eps))
        
        # 4. Apply Weights and Aggregate
        # WRMSSE = Sum(Weight_i * RMSSE_i)
        wrmsse = torch.sum(self.weights * rmsse)
        
        return wrmsse.item()

class HierarchicalAggregator:
    """
    Advanced utility to aggregate Level 12 (Item) predictions 
    up to Level 1 (Total).
    
    Essential for 'Supreme' validation as WRMSSE is technically 
    an average across all 12 hierarchical levels.
    """
    def __init__(self, aggregation_matrix: torch.sparse.FloatTensor):
        """
        Args:
            aggregation_matrix: A sparse matrix S [Total_Nodes, Item_Nodes]
                                where S * y_item = y_aggregate
        """
        self.S = aggregation_matrix

    def aggregate(self, y_item: Tensor) -> Tensor:
        """
        Vectorized summation up the Walmart hierarchy.
        """
        # If y_item is [30490, 28], returns [Total_Nodes, 28]
        return torch.sparse.mm(self.S, y_item)

def get_m5_weights_and_scales(sales_train_val, prices, calendar):
    """
    Research Utility: Pre-computes the scaling factors and weights.
    
    The Scale (denominator) is the mean squared difference of 
    all consecutive days in the training period.
    """
    # ... logic for M5 specific scale calculation ...
    # This is usually done in the GraphBuilder/Preprocessing stage
    pass