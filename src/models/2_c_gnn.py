import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GATv2Conv
from torch_geometric.typing import Adj
from src.models.base_expert import GNNResidualBlock

class BehavioralAttentionHead(nn.Module):
    """
    Research-grade Attention Head. 
    Implements multi-head attention over behavioral correlations 
    with specialized dropouts to prevent 'Attention Collapse'.
    """
    def __init__(self, dim: int, heads: int = 8, dropout: float = 0.2):
        super().__init__()
        self.conv = GATv2Conv(
            in_channels=dim,
            out_channels=dim // heads,
            heads=heads,
            dropout=dropout,
            concat=True,
            edge_dim=1  # We use the correlation coefficient as an edge feature
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor, edge_index: Adj, edge_weight: Tensor) -> Tensor:
        identity = x
        # Dynamic attention weights based on node features AND correlation strength
        x = self.conv(x, edge_index, edge_weight)
        return self.norm(x + identity)

class CGNNExpert(nn.Module):
    """
    2. Cross-Correlation Behavioral GNN (The Market Intelligence Expert)
    
    Research Strategy: 
    Nodes are connected based on historical Pearson Correlation of sales.
    This GNN uses dynamic attention to identify which 'neighbors' are 
    actual predictors vs. mere statistical noise.
    """
    def __init__(self, 
                 in_dim: int, 
                 hidden_dim: int, 
                 num_layers: int = 3, 
                 heads: int = 16, 
                 dropout: float = 0.2):
        super().__init__()
        
        # High-dimensional embedding space projection
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # Expert Layers: Each layer allows 'information' to jump across the store
        # between correlated items (e.g., Beer -> Charcoal).
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            # Using GATv2 for 'universal' attention ranking
            conv = GATv2Conv(
                in_channels=hidden_dim, 
                out_channels=hidden_dim // heads, 
                heads=heads, 
                edge_dim=1, # Uses correlation strength
                concat=True
            )
            self.layers.append(GNNResidualBlock(hidden_dim, conv, dropout=dropout))
            
        # Global Behavioral Context
        # Captures the 'average' behavior of the correlated cluster
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.context_gate = nn.Linear(hidden_dim, hidden_dim)
        
        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: Tensor, edge_index: Adj, edge_weight: Tensor) -> Tensor:
        """
        Args:
            x: Node features [N, in_dim]
            edge_index: Correlation-based adjacency [2, E]
            edge_weight: The raw Pearson correlation values [E, 1]
        """
        # 1. Project to Behavioral Latent Space
        x = self.input_proj(x)
        
        # 2. Dynamic Attention Message Passing
        # In this phase, nodes attend to 'Behavioral Neighbors' 
        # weighted by their correlation strength.
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr=edge_weight)
            
        # 3. Behavioral Context Injection
        # We calculate a 'Cluster Context' to help individual items 
        # understand if the entire 'basket' is trending upward.
        context = torch.mean(x, dim=0, keepdim=True) # Global Store Trend
        gate = torch.sigmoid(self.context_gate(context))
        x = x + (gate * context)
        
        return self.final_norm(x).contiguous()