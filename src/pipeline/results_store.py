"""
M5 Chaos Forecasting — Results Store (Hawkes-Extended)
========================================================

DataFrame-based storage for experiment results with full Hawkes
process intensity trace logging.

Schema:
    experiment_id | model | failure_type | intensity | seed |
    mu | alpha | beta |
    wrmsse | rmse | mae |
    robustness_wrmsse | robustness_rmse | robustness_mae |
    lambda_mean | lambda_max | n_hawkes_events |
    lambda_trace_path |
    runtime_sec | n_test_samples | timestamp

The λ(t) intensity trace for each experiment is stored as a separate
.npz file under experiments/results/intensity_traces/ to keep the
main CSV lean while retaining full traces for deep analysis.

Robustness is now parameterised as:
    R(μ, α, β) = Metric_baseline / Metric_chaos(Hawkes(μ, α, β))
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.chaos.chaos_config import RESULTS_DIR, INTENSITY_TRACES_DIR


class ResultsStore:
    """
    Stores and manages experiment results with Hawkes metadata.

    Results are stored in a pandas DataFrame and saved/loaded from CSV.
    Intensity traces are stored as separate .npz files.
    """

    COLUMNS = [
        'experiment_id', 'model', 'failure_type', 'intensity', 'seed',
        'mu', 'alpha', 'beta',
        'wrmsse', 'rmse', 'mae',
        'robustness_wrmsse', 'robustness_rmse', 'robustness_mae',
        'lambda_mean', 'lambda_max', 'n_hawkes_events',
        'lambda_trace_path',
        'n_test_samples', 'runtime_sec', 'timestamp',
    ]

    def __init__(self, name: str = 'experiment_results'):
        self.name = name
        self.results = pd.DataFrame(columns=self.COLUMNS)
        self.csv_path = RESULTS_DIR / f"{name}.csv"

    def add_result(
        self,
        experiment: dict,
        metrics: dict,
        runtime_sec: float = 0.0,
        n_test_samples: int = 0,
        hawkes_stats: Optional[dict] = None,
        intensity_trace: Optional[np.ndarray] = None,
    ):
        """
        Add a single experiment result.

        Parameters
        ----------
        experiment : dict
            Experiment spec with: experiment_id, model, failure_type,
            intensity, seed, mu, alpha, beta
        metrics : dict
            Metrics with: wrmsse, rmse, mae
        runtime_sec : float
            Experiment runtime in seconds.
        n_test_samples : int
            Number of test samples.
        hawkes_stats : dict, optional
            Hawkes summary stats with: lambda_mean, lambda_max, n_events
        intensity_trace : np.ndarray, optional
            Full λ(t) trace to save as .npz file.
        """
        # Save intensity trace if provided
        trace_path = ''
        if intensity_trace is not None and len(intensity_trace) > 0:
            exp_id = experiment.get('experiment_id', len(self.results))
            trace_filename = f"trace_exp{exp_id}.npz"
            trace_path = str(INTENSITY_TRACES_DIR / trace_filename)
            np.savez_compressed(
                trace_path,
                intensity_trace=intensity_trace,
                mu=experiment.get('mu', 0.0),
                alpha=experiment.get('alpha', 0.0),
                beta=experiment.get('beta', 1.0),
            )

        # Extract Hawkes stats
        if hawkes_stats is None:
            hawkes_stats = {}

        row = {
            'experiment_id': experiment.get('experiment_id', len(self.results)),
            'model': experiment['model'],
            'failure_type': experiment['failure_type'],
            'intensity': experiment['intensity'],
            'seed': experiment['seed'],
            'mu': experiment.get('mu', 0.0),
            'alpha': experiment.get('alpha', 0.0),
            'beta': experiment.get('beta', 1.0),
            'wrmsse': metrics.get('wrmsse', np.nan),
            'rmse': metrics.get('rmse', np.nan),
            'mae': metrics.get('mae', np.nan),
            'robustness_wrmsse': np.nan,  # computed after baselines
            'robustness_rmse': np.nan,
            'robustness_mae': np.nan,
            'lambda_mean': hawkes_stats.get('lambda_mean', 0.0),
            'lambda_max': hawkes_stats.get('lambda_max', 0.0),
            'n_hawkes_events': hawkes_stats.get('n_events', 0),
            'lambda_trace_path': trace_path,
            'n_test_samples': n_test_samples,
            'runtime_sec': runtime_sec,
            'timestamp': datetime.now().isoformat(),
        }

        self.results = pd.concat(
            [self.results, pd.DataFrame([row])],
            ignore_index=True
        )

    def compute_robustness(self):
        """
        Compute robustness metric R for each experiment:

        R = Performance_baseline / Performance_chaos

        For metrics where lower is better (WRMSSE, RMSE, MAE):
            R = baseline_metric / chaos_metric

        R = 1.0 means no degradation
        R < 1.0 means degradation (lower = worse)

        Robustness is now parameterised: R(μ, α, β).
        """
        for model in self.results['model'].unique():
            model_mask = self.results['model'] == model

            # Get baseline metrics (average across seeds)
            baseline_mask = model_mask & (self.results['failure_type'] == 'baseline')

            if baseline_mask.sum() == 0:
                continue

            baseline_wrmsse = self.results.loc[baseline_mask, 'wrmsse'].mean()
            baseline_rmse = self.results.loc[baseline_mask, 'rmse'].mean()
            baseline_mae = self.results.loc[baseline_mask, 'mae'].mean()

            # Compute robustness for chaos experiments
            chaos_mask = model_mask & (self.results['failure_type'] != 'baseline')

            for idx in self.results[chaos_mask].index:
                w = self.results.loc[idx, 'wrmsse']
                r = self.results.loc[idx, 'rmse']
                m = self.results.loc[idx, 'mae']

                self.results.loc[idx, 'robustness_wrmsse'] = (
                    baseline_wrmsse / w if w > 0 else 0.0
                )
                self.results.loc[idx, 'robustness_rmse'] = (
                    baseline_rmse / r if r > 0 else 0.0
                )
                self.results.loc[idx, 'robustness_mae'] = (
                    baseline_mae / m if m > 0 else 0.0
                )

            # Baselines have robustness = 1.0
            for idx in self.results[baseline_mask].index:
                self.results.loc[idx, 'robustness_wrmsse'] = 1.0
                self.results.loc[idx, 'robustness_rmse'] = 1.0
                self.results.loc[idx, 'robustness_mae'] = 1.0

    def save(self):
        """Save results to CSV."""
        self.results.to_csv(self.csv_path, index=False)
        print(f"[ResultsStore] Saved {len(self.results)} results to {self.csv_path}")

    def load(self) -> pd.DataFrame:
        """Load results from CSV."""
        if self.csv_path.exists():
            self.results = pd.read_csv(self.csv_path)
            print(f"[ResultsStore] Loaded {len(self.results)} results from {self.csv_path}")
        else:
            print(f"[ResultsStore] No saved results found at {self.csv_path}")
        return self.results

    def get_summary(self) -> pd.DataFrame:
        """
        Get summary statistics grouped by (model, failure_type, intensity,
        mu, alpha, beta).

        Returns DataFrame with mean and std of metrics across seeds.
        """
        group_cols = ['model', 'failure_type', 'intensity', 'mu', 'alpha', 'beta']
        metric_cols = ['wrmsse', 'rmse', 'mae',
                       'robustness_wrmsse', 'robustness_rmse', 'robustness_mae',
                       'lambda_mean', 'lambda_max', 'n_hawkes_events']

        summary = self.results.groupby(group_cols)[metric_cols].agg(
            ['mean', 'std']
        ).reset_index()

        # Flatten column names
        summary.columns = [
            '_'.join(col).strip('_') if isinstance(col, tuple) else col
            for col in summary.columns
        ]

        return summary

    def get_baseline_metrics(self, model: str = 'lgbm') -> dict:
        """Get average baseline metrics for a model."""
        mask = (self.results['model'] == model) & \
               (self.results['failure_type'] == 'baseline')

        if mask.sum() == 0:
            return {'wrmsse': np.nan, 'rmse': np.nan, 'mae': np.nan}

        return {
            'wrmsse': self.results.loc[mask, 'wrmsse'].mean(),
            'rmse': self.results.loc[mask, 'rmse'].mean(),
            'mae': self.results.loc[mask, 'mae'].mean(),
        }

    def get_hawkes_robustness_surface(
        self,
        model: str = 'lgbm',
        failure_type: str = None,
        metric: str = 'rmse',
    ) -> pd.DataFrame:
        """
        Get robustness R(μ, α, β) surface for a given model/failure type.

        Returns pivot table suitable for heatmap plotting.
        """
        mask = self.results['model'] == model
        if failure_type:
            mask = mask & (self.results['failure_type'] == failure_type)
        mask = mask & (self.results['failure_type'] != 'baseline')

        df = self.results[mask].copy()

        rob_col = f'robustness_{metric}'
        if rob_col not in df.columns:
            return pd.DataFrame()

        return df.groupby(['alpha', 'beta'])[rob_col].mean().reset_index()
