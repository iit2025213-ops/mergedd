import os
import torch
import pandas as pd
import numpy as np
import yaml
import logging
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List

# Setup Research-Grade Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Preprocess_Engine")

class M5Preprocessor:
    """
    Supreme Preprocessing Engine.
    
    Converts raw M5 CSVs into research-ready artifacts:
    1. .parquet: For Boosting Experts (LGBM/XGB).
    2. .pt: For GNN Experts (Node features and Targets).
    3. .pt (Meta): WRMSSE Weights and Scales.
    """
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)
        
        self.raw_dir = Path(self.cfg['paths']['raw_data_dir'])
        self.proc_dir = Path(self.cfg['paths']['processed_dir'])
        self.proc_dir.mkdir(parents=True, exist_ok=True)

    def reduce_mem_usage(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standard research practice to fit M5 into system RAM."""
        for col in df.columns:
            if df[col].dtype == 'float64':
                df[col] = df[col].astype('float32')
            if df[col].dtype == 'int64':
                df[col] = df[col].astype('int32')
        return df

    def compute_wrmsse_scales(self, sales: pd.DataFrame) -> torch.Tensor:
        """
        Pre-computes the 'Scale' for each of the 30,490 series.
        Scale = Mean Squared Difference of consecutive days (t vs t-1).
        """
        logger.info("Computing Hierarchical Scales for WRMSSE...")
        # Get only the 'd_' sales columns
        sales_data = sales.filter(like='d_').values
        
        # Calculate (y_t - y_{t-1})^2
        diffs = np.diff(sales_data, axis=1) ** 2
        scales = np.mean(diffs, axis=1)
        
        # Add epsilon to prevent division by zero in the loss function
        return torch.tensor(scales, dtype=torch.float32) + 1e-8

    def process_tabular_data(self):
        """Prepares the 'Long' format data for LGBM/XGB Experts."""
        logger.info("Generating Tabular Features (Long Format)...")
        
        sales = pd.read_csv(self.raw_dir / "sales_train_evaluation.csv")
        calendar = pd.read_csv(self.raw_dir / "calendar.csv")
        prices = pd.read_csv(self.raw_dir / "sell_prices.csv")

        # 1. Melt Sales to Long Format
        # This turns 1,941 columns into rows for GBDT compatibility
        df = pd.melt(sales, 
                     id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], 
                     var_name='d', value_name='sales')
        
        # 2. Merge Calendar & Prices
        df = df.merge(calendar, on='d', how='left')
        df = df.merge(prices, on=['store_id', 'item_id', 'wm_yr_wk'], how='left')

        # 3. Research-Grade Label Encoding
        # Preserves hierarchical relationships for the GNNs
        cat_cols = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'event_name_1', 'event_type_1']
        for col in cat_cols:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

        # 4. Feature Engineering: Lags & Windows
        # Ingesting hyperparams from data_config.yaml
        for lag in self.cfg['features']['lags']:
            df[f'lag_{lag}'] = df.groupby('id')['sales'].transform(lambda x: x.shift(lag))
            
        for window in self.cfg['features']['rolling_windows']:
            df[f'roll_mean_{window}'] = df.groupby('id')['sales'].transform(lambda x: x.shift(1).rolling(window).mean())

        # Save as Parquet for fast IO
        df.to_parquet(self.proc_dir / "lgbm_features.parquet", index=False)
        logger.info("Tabular Parquet exported.")

    def process_gnn_tensors(self):
        """Prepares the 'Wide' format tensors for the 10-GNN Engine."""
        logger.info("Generating Node Tensors (Wide Format)...")
        
        sales = pd.read_csv(self.raw_dir / "sales_train_evaluation.csv")
        
        # 1. Targets [N, 28] - The last 28 days for validation
        targets = torch.tensor(sales.filter(like='d_').values[:, -28:], dtype=torch.float32)
        
        # 2. Node Features [N, In_Dim]
        # We take a snapshot of the most recent historical features
        # e.g., rolling means and static categorical embeddings
        node_features = torch.tensor(sales.filter(like='d_').values[:, -100:-28], dtype=torch.float32)
        
        # 3. Meta Data (Weights and Scales)
        scales = self.compute_wrmsse_scales(sales)
        # Dummy weights for initialization; actual weights should come from 'weights_validation.csv'
        weights = torch.ones_like(scales) / len(scales) 

        torch.save(targets, self.proc_dir / "targets_train.pt")
        torch.save(node_features, self.proc_dir / "features_train.pt")
        torch.save({'weights': weights, 'scales': scales}, self.proc_dir / "m5_meta.pt")
        
        logger.info("GNN Tensors and Metadata exported.")

if __name__ == "__main__":
    preprocessor = M5Preprocessor(config_path="configs/data_config.yaml")
    preprocessor.process_tabular_data()
    preprocessor.process_gnn_tensors()