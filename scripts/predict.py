import os
import torch
import pandas as pd
import numpy as np
import yaml
import logging
from pathlib import Path
from src.models.decagon_ensemble import DecagonEnsemble
from src.engine.pipeline import M5SupremeDataset
from src.boosting.lgbm_expert import SupremeLGBMExpert
from src.boosting.xgb_expert import XGBExpert

# Setup Research-Grade Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Prediction_Auditor")

class M5Auditor:
    """
    Research-Grade Inference & Ensemble Fusion Engine.
    
    Features:
    - EMA Weight Loading: Pulls the smoothed 'Shadow' weights for stability.
    - Expert Trust Blending: Implements the 0.6/0.2/0.2 Hybrid Fusion.
    - Zero-Clamping & Post-Processing: Ensures physical validity of forecasts.
    """
    def __init__(self, config_path: str, model_path: str):
        with open(config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        
        # 1. Initialize the 10-GNN Engine
        self.gnn_engine = DecagonEnsemble(
            in_dim=self.cfg['model']['in_dim'],
            hidden_dim=self.cfg['model']['hidden_dim'],
            sig_edge_dim=self.cfg['model']['sig_dim']
        ).to(self.device)

    def load_ema_weights(self):
        """Loads the Exponential Moving Average weights from the checkpoint."""
        logger.info(f"Loading Supreme EMA weights from {self.model_path}...")
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # We prefer 'ema_state' over 'model_state' for sub-0.5 WRMSSE
        if 'ema_state' in checkpoint and checkpoint['ema_state'] is not None:
            # Note: This usually requires the torch-ema library logic 
            # but we assume the state_dict is extracted for simplicity
            self.gnn_engine.load_state_dict(checkpoint['ema_state'])
        else:
            logger.warning("EMA weights not found. Falling back to raw model state.")
            self.gnn_engine.load_state_dict(checkpoint['model_state'])
        
        self.gnn_engine.eval()

    @torch.no_grad()
    def run_inference(self, test_dataset: M5SupremeDataset):
        """Executes the Decagon Ensemble forward pass."""
        logger.info("Executing 10-GNN Expert Consensus...")
        
        # Graph-level inference
        forecast, _, trust_scores = self.gnn_engine(
            test_dataset.x.to(self.device), 
            test_dataset.adj_matrices
        )
        
        # Save trust scores for research audit (which expert was most 'trusted'?)
        torch.save(trust_scores, "outputs/logs/inference_trust_analysis.pt")
        
        return forecast.cpu().numpy()

    def ensemble_fusion(self, gnn_preds: np.ndarray, lgbm_path: str, xgb_path: str):
        """
        Calculates the final Hybrid Blend.
        Aim: Leverage GNN relational depth + Boosting tabular precision.
        """
        logger.info("Fusing Hybrid Experts (LGBM + XGB)...")
        
        # Load Boosting Experts (Pre-trained)
        lgbm = SupremeLGBMExpert().load_model(lgbm_path)
        xgb = XGBExpert().load_model(xgb_path)
        
        # Generate Boosting predictions
        lgbm_preds = lgbm.predict(self.cfg['boosting']['x_test'])
        xgb_preds = xgb.predict(self.cfg['boosting']['x_test'])
        
        # Weights from boosting_config.yaml
        w_gnn = self.cfg['fusion']['weights']['gnn_ensemble']
        w_lgbm = self.cfg['fusion']['weights']['lgbm_expert']
        w_xgb = self.cfg['fusion']['weights']['xgb_expert']
        
        # Final Weighted Average Blend
        final_forecast = (w_gnn * gnn_preds) + (w_lgbm * lgbm_preds) + (w_xgb * xgb_preds)
        
        # Post-Processing: Physical Constraints
        if self.cfg['fusion']['post_process']['floor_at_zero']:
            final_forecast = np.maximum(0, final_forecast)
            
        return final_forecast

    def export_submission(self, predictions: np.ndarray, sample_sub_path: str):
        """Formats the predictions into the M5 submission CSV structure."""
        logger.info("Formatting submission file...")
        sub = pd.read_csv(sample_sub_path)
        
        # M5 requires 28 columns (F1-F28) for 30,490 items (Validation + Evaluation)
        col_names = [f"F{i}" for i in range(1, 29)]
        
        # Assuming predictions are [30490, 28]
        # We split them across Validation and Evaluation rows as per competition rules
        sub.iloc[:len(predictions), 1:] = predictions
        sub.iloc[len(predictions):, 1:] = predictions # Evaluation placeholder
        
        out_path = Path(self.cfg['paths']['submission_dir']) / "supreme_submission.csv"
        sub.to_csv(out_path, index=False)
        logger.info(f"Submission archived at {out_path}. WRMSSE < 0.5 targeted.")

if __name__ == "__main__":
    auditor = M5Auditor(
        config_path="configs/supreme_config.yaml", 
        model_path="checkpoints/decagon_v1/best_model.pt"
    )
    auditor.load_ema_weights()
    
    # Load Inference Dataset
    test_ds = M5SupremeDataset(mode='eval') # Simplified for brevity
    
    gnn_out = auditor.run_inference(test_ds)
    final_out = auditor.ensemble_fusion(gnn_out, "models/lgbm.txt", "models/xgb.json")
    
    auditor.export_submission(final_out, "data/raw/sample_submission.csv")