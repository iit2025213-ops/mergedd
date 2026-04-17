import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import GATv2Conv
from torch_geometric.typing import Adj
from src.models.base_expert import GNNResidualBlock

# Research Note: In production, you'd use 'signatory' or 'iisignature'. 
# This implementation assumes the signatures are pre-calculated 
# in the GraphBuilder as edge/node features.

class SigGNNExpert(nn.Module):
    """
    4. Path Signature GNN (The Volatility & Momentum Expert)
    
    Research Strategy: 
    Utilizes iterated integrals (Signatures) to capture non-linear 
    temporal patterns. This expert is the 'HFT-grade' defense 
    against demand spikes and sudden market shocks.
    """
    def __init__(self, 
                 in_dim: int, 
                 hidden_dim: int, 
                 sig_dim: int, 
                 num_layers: int = 3, 
                 heads: int = 8, 
                 dropout: float = 0.1):
        super().__init__()
        
        # Projection for node features
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        
        # Specialized Projection for Path Signatures
        # Signatures are high-dimensional; we compress them while 
        # preserving the geometric 'shape' information.
        self.sig_compressor = nn.Sequential(
            nn.Linear(sig_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            # We use GATv2 because the attention must be conditioned 
            # on the 'Signature' (the path's momentum) on the edge.
            conv = GATv2Conv(
                in_channels=hidden_dim, 
                out_channels=hidden_dim // heads, 
                heads=heads, 
                edge_dim=hidden_dim, # Compressed Signature as edge attribute
                concat=True
            )
            self.layers.append(GNNResidualBlock(hidden_dim, conv, dropout=dropout))
            
        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: Tensor, edge_index: Adj, edge_sig: Tensor) -> Tensor:
        """
        Args:
            x: Node features [N, in_dim]
            edge_index: Adjacency [2, E]
            edge_sig: Pre-calculated Path Signatures [E, sig_dim]
                      (Represents the joint path of node i and j)
        """
        # 1. Compress the high-dimensional Signature features
        # This allows the A100 to process complex iterated integrals in parallel.
        sig_feat = self.sig_compressor(edge_sig)
        
        # 2. Project node features
        x = self.input_proj(x)
        
        # 3. Path-Conditioned Message Passing
        # The GATv2 attention mechanism now asks: 
        # "Based on the 'momentum' (signature) between these two products, 
        # how much should Item A influence Item B's forecast?"
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr=sig_feat)
            
        return self.final_norm(x).contiguous()

    @staticmethod
    def get_signature_formula():
        """
        The mathematical foundation of this layer.
        The signature S(X) of a path X is the collection of all iterated integrals:
        """
        return r"S(X)_{s,t} = \left( 1, \int dX, \iint dX^2, \dots, \int \dots \int dX^k \right)"