import torch
import torch.nn as nn
from typing import Dict, List, Tuple

# Importing your expert modules
from .expert_1_h_gnn import HGNNExpert
from .expert_2_c_gnn import CGNNExpert
from .expert_3_graphormer import GraphormerExpert
from .expert_4_sig_gnn import SigGNNExpert
from .expert_5_zi_gnn import ZIGNNExpert
from .expert_6_e_gnn import EGNNExpert
from .expert_7_cal_gnn import CalGNNExpert
from .expert_8_flow_gnn import FlowGNNExpert
# Note: Ensure these file names match your actual files exactly
from .expert_9_vat_gnn import VATGNNExpert
from .expert_10_blender import MetaBlender

class DecagonEnsemble(nn.Module):
    """
    Decagon Ensemble (GNN-10 Orchestrator).
    
    Research Strategy: 
    This class wraps all 9 GNN experts into a unified manifold. 
    It is designed for the A100 80GB to maximize parallel expert 
    execution via the Meta-Blender's attention gate.
    """
    def __init__(self, in_dim: int, hidden_dim: int, sig_edge_dim: int = 16):
        super().__init__()
        
        # 1. Initialize the 9 Specialized GNN Experts
        # These are the 'Opinions' that the Blender will evaluate.
        self.h_gnn = HGNNExpert(in_dim, hidden_dim)
        self.c_gnn = CGNNExpert(in_dim, hidden_dim)
        self.graphormer = GraphormerExpert(in_dim, hidden_dim)
        self.sig_gnn = SigGNNExpert(in_dim, hidden_dim, sig_edge_dim)
        self.zi_gnn = ZIGNNExpert(in_dim, hidden_dim)
        self.e_gnn = EGNNExpert(in_dim, hidden_dim)
        self.cal_gnn = CalGNNExpert(in_dim, hidden_dim)
        self.flow_gnn = FlowGNNExpert(in_dim, hidden_dim)
        self.vat_gnn = VATGNNExpert(in_dim, hidden_dim)
        
        # 2. Initialize the Meta-Blender (The 10th Brain)
        self.blender = MetaBlender(hidden_dim, num_experts=9)

    def forward(self, x: torch.Tensor, adj_dict: Dict[str, torch.Tensor], 
                time_idx: torch.Tensor = None, edge_attr_dict: Dict = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Raw node features [N, in_dim]
            adj_dict: Dict containing the 10 graph views
            time_idx: Current day index (required for CalGNN)
            edge_attr_dict: Optional edge attributes (for Econ/Corr GNNs)
            
        Returns:
            final_forecast: [N, 28] point forecasts
            prob_zero_logits: Sparsity mask from ZI-GNN
            trust_weights: [N, 9] expert attention weights
        """
        # Execute all 9 experts in parallel (optimized for A100)
        h1 = self.h_gnn(x, adj_dict['hierarchical'])
        h2 = self.c_gnn(x, adj_dict['behavioral'], edge_weight=edge_attr_dict.get('behavioral'))
        h3 = self.graphormer(x, adj_dict['global_transformer'])
        h4 = self.sig_gnn(x, adj_dict['path_signature'])
        
        # Expert 5 (ZI-GNN) is unique: it provides both volume latent and sparsity mask
        h5_vol, zi_logits = self.zi_gnn(x, adj_dict['zero_inflation'])
        
        h6 = self.e_gnn(x, adj_dict['economic'], edge_attr=edge_attr_dict.get('economic'))
        h7 = self.cal_gnn(x, adj_dict['temporal_sync'], edge_attr=edge_attr_dict.get('temporal_sync'), time_idx=time_idx)
        h8 = self.flow_gnn(x, adj_dict['logistics_flow'])
        h9 = self.vat_gnn(x, adj_dict['adversarial'])

        # Create the Latent Expert Stack [N, 9, Hidden_Dim]
        expert_embeddings = [h1, h2, h3, h4, h5_vol, h6, h7, h8, h9]
        
        # The Final Fusion Logic
        final_forecast, trust_weights = self.blender(expert_embeddings, zi_logits)
        
        return final_forecast, zi_logits, trust_weights

    @torch.no_grad()
    def predict_all(self, x, adj_dict, time_idx=None, edge_attr_dict=None):
        """High-performance inference method for the final ensemble stage."""
        self.eval()
        return self.forward(x, adj_dict, time_idx, edge_attr_dict)[0]