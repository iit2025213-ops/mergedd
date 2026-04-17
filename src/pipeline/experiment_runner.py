"""
M5 Chaos Forecasting — Experiment Runner (Hawkes-Extended)
============================================================

Orchestrates the full experimentation pipeline with Hawkes process integration:

1. Load preprocessed data + features
2. Train baseline model(s)
3. For each experiment E_k = (model, failure_type, intensity, seed, μ, α, β):
   a. Instantiate HawkesProcess with shared event history
   b. Apply chaos perturbation (Hawkes-driven or Bernoulli)
   c. Evaluate model on perturbed test data
   d. Record metrics + Hawkes intensity trace
4. Compute robustness: R(μ, α, β) = Performance_baseline / Performance_chaos
5. MLE-fit Hawkes parameters from generated failure data
6. Save results + intensity traces to CSV/npz

The pipeline maintains a shared event_history across chaos stages
so excitation carries over continuously through the pipeline.
"""

import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import gc

from src.chaos.chaos_config import (
    set_global_seed, DEFAULT_SEED, SAMPLE_N_ITEMS,
    END_TRAIN_DAY, HORIZON, RANDOM_SEEDS, USE_HAWKES
)
from src.data_loader import load_and_preprocess, get_train_test_split
from src.feature_engineering import engineer_features, get_feature_columns
from src.models.lightgbm_model import LightGBMForecaster
from src.models.mlp_model import MLPForecaster
from src.metrics.wrmsse import rmse, mae, SimplifiedWRMSSE
from src.chaos.fault_injection import (
    enumerate_experiments, inject_fault, get_experiment_label
)
from src.chaos.hawkes_process import fit_hawkes_from_mask
from src.pipeline.results_store import ResultsStore


class ExperimentRunner:
    """
    Orchestrates chaos engineering experiments on M5 forecasting models
    with full Hawkes process support.

    Attributes
    ----------
    models : dict
        Trained model instances keyed by name.
    data : dict
        Preprocessed data from data_loader.
    feature_cols : list
        Feature columns used by models.
    results : ResultsStore
        Experiment results tracker (Hawkes-extended).
    use_hawkes : bool
        Whether to use Hawkes process for chaos perturbations.
    """

    def __init__(self, sample_n: int = None, use_hawkes: bool = None):
        self.sample_n = sample_n or SAMPLE_N_ITEMS
        self.use_hawkes = use_hawkes if use_hawkes is not None else USE_HAWKES
        self.models = {}
        self.data = None
        self.df = None
        self.feature_cols = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.results = ResultsStore()
        self.wrmsse_evaluator = None

    def load_data(self):
        """Load and preprocess M5 data."""
        print("=" * 70)
        print("PHASE 1: DATA LOADING AND FEATURE ENGINEERING")
        print("=" * 70)

        set_global_seed(DEFAULT_SEED)

        # Load raw data
        self.data = load_and_preprocess(sample_n=self.sample_n)
        self.df = self.data['df']

        # Engineer features
        self.df, self.feature_cols = engineer_features(self.df, cache=True)

        # Train/test split
        df_train, df_test = get_train_test_split(self.df)

        # Prepare feature matrices
        self.X_train = df_train[self.feature_cols].copy()
        self.y_train = df_train['sales'].values.astype(np.float64)
        self.X_test = df_test[self.feature_cols].copy()
        self.y_test = df_test['sales'].values.astype(np.float64)

        # Keep test metadata for WRMSSE
        self.test_ids = df_test[['id', 'item_id', 'dept_id', 'cat_id',
                                  'store_id', 'state_id', 'day_num']].copy()

        # Copy store_id_enc to test features for store removal experiments
        if 'store_id_enc' in df_test.columns:
            self.X_test['store_id_enc'] = df_test['store_id_enc'].values
        if 'store_id' in df_test.columns:
            self.X_test['store_id'] = df_test['store_id'].values

        # Setup simplified WRMSSE evaluator
        self._setup_wrmsse()

        print(f"\nData loaded: {len(self.X_train):,} train, "
              f"{len(self.X_test):,} test samples")
        print(f"Features: {len(self.feature_cols)}")
        print(f"Hawkes process: {'ENABLED' if self.use_hawkes else 'DISABLED'}")

        del df_train, df_test
        gc.collect()

    def _setup_wrmsse(self):
        """Initialize the simplified WRMSSE evaluator."""
        unique_ids = self.df[self.df['day_num'] <= END_TRAIN_DAY]['id'].unique()
        train_data = self.df[self.df['day_num'] <= END_TRAIN_DAY]

        print("[ExperimentRunner] Building WRMSSE evaluator...")

        scale_start = max(1, END_TRAIN_DAY - 200)
        scale_data = train_data[train_data['day_num'] >= scale_start]

        try:
            scale_pivot = scale_data.pivot_table(
                index='id', columns='day_num', values='sales', fill_value=0
            ).values.astype(np.float64)

            last28 = train_data[train_data['day_num'] > END_TRAIN_DAY - 28]
            dollar_proxy = last28.groupby('id').agg(
                dollar=('sales', 'sum')
            ).reindex(unique_ids, fill_value=0)['dollar'].values

            self.wrmsse_evaluator = SimplifiedWRMSSE(
                train_sales=scale_pivot,
                dollar_sales=dollar_proxy,
            )

            self.wrmsse_series_ids = unique_ids

        except Exception as e:
            print(f"[ExperimentRunner] WARNING: Could not build WRMSSE evaluator: {e}")
            print("[ExperimentRunner] Will use RMSE/MAE only.")
            self.wrmsse_evaluator = None

    def train_models(self, model_names: list = None):
        """Train baseline models."""
        print("\n" + "=" * 70)
        print("PHASE 2: MODEL TRAINING")
        print("=" * 70)

        model_names = model_names or ['lgbm']

        val_start = END_TRAIN_DAY - HORIZON

        train_mask = self.df['day_num'] <= val_start
        val_mask = (self.df['day_num'] > val_start) & (self.df['day_num'] <= END_TRAIN_DAY)

        X_tr = self.df.loc[train_mask, self.feature_cols]
        y_tr = self.df.loc[train_mask, 'sales'].values.astype(np.float64)
        X_val = self.df.loc[val_mask, self.feature_cols]
        y_val = self.df.loc[val_mask, 'sales'].values.astype(np.float64)

        for name in model_names:
            print(f"\n--- Training {name.upper()} ---")
            set_global_seed(DEFAULT_SEED)

            if name == 'lgbm':
                model = LightGBMForecaster()
                model.train(
                    X_tr, y_tr, X_val, y_val,
                    feature_cols=self.feature_cols,
                )
                model.save('lgbm_baseline')

                fi = model.feature_importance()
                print(f"\nTop 10 features (by gain):")
                print(fi.head(10).to_string(index=False))

            elif name == 'mlp':
                model = MLPForecaster()
                model.train(
                    X_tr, y_tr, X_val, y_val,
                    feature_cols=self.feature_cols,
                )
                model.save('mlp_baseline')

            else:
                raise ValueError(f"Unknown model: {name}")

            self.models[name] = model

            # Baseline evaluation on clean test data
            preds = model.predict(self.X_test[self.feature_cols])
            test_rmse = rmse(self.y_test, preds)
            test_mae_val = mae(self.y_test, preds)
            print(f"\n{name.upper()} Baseline — Test RMSE: {test_rmse:.4f}, "
                  f"MAE: {test_mae_val:.4f}")

        del X_tr, y_tr, X_val, y_val
        gc.collect()

    def run_experiments(
        self,
        model_names: list = None,
        failure_types: list = None,
        seeds: list = None,
    ):
        """
        Run all chaos engineering experiments with Hawkes integration.
        """
        print("\n" + "=" * 70)
        print("PHASE 3: CHAOS ENGINEERING EXPERIMENTS"
              + (" (HAWKES)" if self.use_hawkes else " (BERNOULLI)"))
        print("=" * 70)

        model_names = model_names or list(self.models.keys())

        experiments = enumerate_experiments(
            models=model_names,
            failure_types=failure_types,
            seeds=seeds,
            use_hawkes=self.use_hawkes,
        )

        # Numeric-only feature cols
        numeric_feats = [c for c in self.feature_cols
                         if self.X_test[c].dtype in [np.float64, np.float32,
                                                       np.int64, np.int32,
                                                       np.int16, np.int8]]

        for exp in tqdm(experiments, desc="Experiments"):
            t0 = time.time()
            model_name = exp['model']
            model = self.models[model_name]

            # Apply chaos perturbation (returns HawkesProcess instance)
            X_test_perturbed, y_test_perturbed, hawkes_proc = inject_fault(
                experiment=exp,
                X_test=self.X_test.copy(),
                y_test=self.y_test.copy(),
                feature_cols=numeric_feats,
                df_full=self.df,
            )

            # Predict on perturbed data
            try:
                preds = model.predict(X_test_perturbed[self.feature_cols])
            except Exception as e:
                available_feats = [c for c in self.feature_cols
                                   if c in X_test_perturbed.columns]
                if len(available_feats) < len(self.feature_cols):
                    for c in self.feature_cols:
                        if c not in X_test_perturbed.columns:
                            X_test_perturbed[c] = 0
                preds = model.predict(X_test_perturbed[self.feature_cols])

            # Compute metrics
            if exp['failure_type'] in ('demand_spike', 'temporal_outage'):
                y_actual = y_test_perturbed[:len(preds)]
            else:
                y_actual = self.y_test[:len(preds)]

            metrics = {
                'rmse': rmse(y_actual, preds),
                'mae': mae(y_actual, preds),
                'wrmsse': np.nan,
            }

            # Compute simplified WRMSSE
            if self.wrmsse_evaluator is not None and exp['failure_type'] != 'store_removal':
                try:
                    n_test = len(y_actual)
                    n_series = len(self.wrmsse_series_ids)
                    h = HORIZON

                    if n_test == n_series * h:
                        y_true_wide = y_actual.reshape(n_series, h)
                        y_pred_wide = preds.reshape(n_series, h)
                        metrics['wrmsse'] = self.wrmsse_evaluator.evaluate(
                            y_true_wide, y_pred_wide
                        )
                except Exception:
                    pass

            # ─── Hawkes Statistics & MLE Fitting ─────────────────────

            hawkes_stats = None
            intensity_trace = None

            if hawkes_proc is not None:
                # Get summary stats from the Hawkes process
                hawkes_stats = hawkes_proc.get_summary_stats()
                intensity_trace = hawkes_proc.get_intensity_trace()

                # MLE-fit Hawkes parameters from the generated failure data
                if len(hawkes_proc.event_times) >= 2:
                    try:
                        n_steps = len(y_actual)
                        fitted_params = fit_hawkes_from_mask(
                            failure_mask=np.ones(n_steps),  # placeholder
                            dt=1.0,
                        )
                        # Use event times directly for better fit
                        from src.chaos.hawkes_process import fit_hawkes_mle
                        event_times = np.array(hawkes_proc.event_times)
                        T = float(max(n_steps, event_times.max() + 1))
                        fitted = fit_hawkes_mle(event_times, T)

                        hawkes_stats['mle_mu'] = fitted.mu
                        hawkes_stats['mle_alpha'] = fitted.alpha
                        hawkes_stats['mle_beta'] = fitted.beta
                        hawkes_stats['mle_branching_ratio'] = fitted.branching_ratio
                    except Exception as e:
                        hawkes_stats['mle_mu'] = np.nan
                        hawkes_stats['mle_alpha'] = np.nan
                        hawkes_stats['mle_beta'] = np.nan
                        hawkes_stats['mle_branching_ratio'] = np.nan

            elapsed = time.time() - t0

            self.results.add_result(
                experiment=exp,
                metrics=metrics,
                runtime_sec=elapsed,
                n_test_samples=len(preds),
                hawkes_stats=hawkes_stats,
                intensity_trace=intensity_trace,
            )

        # Compute robustness metrics
        self.results.compute_robustness()

        # Save results
        self.results.save()

        print(f"\n{'=' * 70}")
        print("EXPERIMENT RESULTS SUMMARY")
        print(f"{'=' * 70}")
        summary = self.results.get_summary()
        print(summary.to_string(index=False))

        return self.results

    def run_full_pipeline(
        self,
        model_names: list = None,
        failure_types: list = None,
        sample_n: int = None,
    ):
        """
        Run the complete pipeline: load → train → experiment → save.
        """
        if sample_n is not None:
            self.sample_n = sample_n

        self.load_data()
        self.train_models(model_names=model_names or ['lgbm', 'mlp'])
        results = self.run_experiments(
            model_names=model_names or list(self.models.keys()),
            failure_types=failure_types,
        )

        return results
