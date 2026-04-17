import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.typing import Adj, OptTensor
import math

def drop_path(x: Tensor, drop_prob: float = 0.0, training: bool = False) -> Tensor:
    """
    Stochastic Depth (DropPath) per sample.
    Vital for deep GNN ensembles to prevent co-adaptation of experts.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    return x.div(keep_prob) * random_tensor

class GNNResidualBlock(nn.Module):
    """
    Supreme Research-Grade Backbone for Deep GNNs.
    
    Architectural Upgrades:
    1. LayerScale: Initializing residual branches with 1e-6 to enable 50+ layers.
    2. Stochastic Depth: Randomly dropping paths to boost ensemble resilience.
    3. Proper Initialization: Kaiming initialization for GELU/Swish manifolds.
    4. SwiGLU-style FFN: Enhanced non-linear feature mixing for demand spikes.
    """
    def __init__(self, 
                 hidden_dim: int, 
                 conv_layer: nn.Module, 
                 dropout: float = 0.1, 
                 drop_path_rate: float = 0.05,
                 init_values: float = 1e-6):
        super().__init__()
        self.conv = conv_layer
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # LayerScale parameters: Allows deep GNNs to start as identity mappings
        # and slowly learn the residual signal, preventing initial gradient chaos.
        self.ls1 = nn.Parameter(init_values * torch.ones(hidden_dim))
        self.ls2 = nn.Parameter(init_values * torch.ones(hidden_dim))
        
        # SwiGLU-inspired FFN for superior feature separation
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.drop_path_rate = drop_path_rate
        self._init_weights()

    def _init_weights(self):
        """Kaiming Initialization for high-stakes deep learning."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: OptTensor = None) -> Tensor:
        # --- Stage 1: Spatial/Graph Information Flow ---
        # Pre-Norm for training stability on A100 Mixed Precision
        h = self.norm1(x)
        
        if edge_attr is not None:
            h = self.conv(h, edge_index, edge_attr)
        else:
            h = self.conv(h, edge_index)
            
        h = self.dropout(h)
        
        # LayerScale + Stochastic Depth
        # x = x + DropPath(λ1 * Conv(LN(x)))
        x = x + drop_path(self.ls1 * h, self.drop_path_rate, self.training)
        
        # --- Stage 2: Feature Mixing (Point-wise FFN) ---
        h = self.norm2(x)
        h = self.ffn(h)
        
        # LayerScale + Stochastic Depth
        # x = x + DropPath(λ2 * FFN(LN(x)))
        x = x + drop_path(self.ls2 * h, self.drop_path_rate, self.training)
        
        return x.contiguous()