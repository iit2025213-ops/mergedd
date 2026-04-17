"""
M5 Chaos Forecasting — Fault Injection Testing (Systematic) + Hawkes Process
==============================================================================

Defines parameterized, reproducible experiments for chaos engineering
with full Hawkes process integration.

Each experiment is formally defined as:

    E_k = (model, failure_type, intensity, seed, α, β)

Key design decision (literature-grounded):
    μ is DERIVED from the 'intensity' parameter via:
        μ = -ln(1 - p_base)
    This ensures the Hawkes baseline rate matches the intended Bernoulli
    probability. Only α and β are Hawkes-specific sweep parameters.

    Reference: Standard Poisson process theory; Bacry et al. (2015).

This module:
1. Enumerates all experiment combinations including Hawkes parameter grid
2. Uses fixed random seeds for reproducibility
3. Instantiates a HawkesProcess per experiment with μ derived from intensity
4. Returns perturbed data ready for model evaluation

When USE_HAWKES is False (config), the Hawkes parameters are omitted
and the pipeline reduces to the original Bernoulli-based experiments:
    E_k = (model, failure_type, intensity, seed)
"""

import numpy as np
import pandas as pd
from itertools import product
from typing import List, Dict, Tuple, Optional

from src.chaos.chaos_config import (
    FAILURE_TYPES, FAILURE_INTENSITIES, RANDOM_SEEDS, DEMAND_SPIKE_FRAC,
    USE_HAWKES, HAWKES_ALPHA_VALUES, HAWKES_BETA_VALUES,
    mu_from_intensity,
)
from src.chaos.chaos_monkey import apply_chaos_monkey
from src.chaos.chaos_kong import (
    apply_store_removal, apply_temporal_outage, apply_demand_spike
)
from src.chaos.hawkes_process import HawkesProcess, HawkesParams


def enumerate_experiments(
    models: List[str] = None,
    failure_types: List[str] = None,
    seeds: List[int] = None,
    use_hawkes: bool = None,
) -> List[Dict]:
    """
    Enumerate all experiment configurations.

    E_k = (model, failure_type, intensity, seed, α, β)

    μ is derived from 'intensity' via μ = -ln(1 - intensity), NOT swept.

    When use_hawkes is True, the (α, β) grid is crossed with the
    existing failure × intensity × seed grid. μ is computed per-experiment
    from the intensity value.

    When use_hawkes is False, α=0 and β=1 (Bernoulli equivalent).

    Parameters
    ----------
    models : list of str
        Model names (e.g., ['lgbm', 'mlp']).
    failure_types : list of str
        Chaos types to test. Default: all.
    seeds : list of int
        Random seeds. Default: from config.
    use_hawkes : bool
        Whether to include Hawkes parameter sweep.
        Default: USE_HAWKES from config.

    Returns
    -------
    list of dicts, each with keys:
        'model', 'failure_type', 'intensity', 'seed',
        'mu', 'alpha', 'beta', 'experiment_id'
    """
    models = models or ['lgbm']
    failure_types = failure_types or FAILURE_TYPES
    seeds = seeds or RANDOM_SEEDS
    if use_hawkes is None:
        use_hawkes = USE_HAWKES

    # Hawkes parameter grid (α, β only — μ derived from intensity)
    if use_hawkes:
        hawkes_grid = list(product(
            HAWKES_ALPHA_VALUES, HAWKES_BETA_VALUES
        ))
    else:
        hawkes_grid = [(0.0, 1.0)]  # α=0 → Bernoulli equivalent

    experiments = []
    exp_id = 0

    # Baseline (no chaos) — one per model × seed
    for model in models:
        for seed in seeds:
            experiments.append({
                'experiment_id': exp_id,
                'model': model,
                'failure_type': 'baseline',
                'intensity': 0.0,
                'seed': seed,
                'mu': 0.0,
                'alpha': 0.0,
                'beta': 1.0,
            })
            exp_id += 1

    # Chaos experiments with Hawkes grid
    for model in models:
        for ft in failure_types:
            intensities = FAILURE_INTENSITIES.get(ft, [0.0])
            for intensity in intensities:
                for seed in seeds:
                    for (alpha, beta) in hawkes_grid:
                        # μ DERIVED from intensity (not independent)
                        # For failure types where intensity is not a
                        # probability (e.g., store_removal counts,
                        # temporal_outage days, demand_spike multipliers),
                        # we use a scaled mapping to [0, 1].
                        mu = _derive_mu(ft, intensity)

                        experiments.append({
                            'experiment_id': exp_id,
                            'model': model,
                            'failure_type': ft,
                            'intensity': intensity,
                            'seed': seed,
                            'mu': mu,
                            'alpha': alpha,
                            'beta': beta,
                        })
                        exp_id += 1

    n_hawkes = len(hawkes_grid)
    n_ft_intensities = sum(
        len(FAILURE_INTENSITIES.get(ft, [0])) for ft in failure_types
    )

    print(f"[FaultInjection] Enumerated {len(experiments)} experiments "
          f"({len(models)} models × {len(failure_types)} failures × "
          f"{n_ft_intensities} intensities × {len(seeds)} seeds × "
          f"{n_hawkes} Hawkes configs + baselines)")

    return experiments


def _derive_mu(failure_type: str, intensity: float) -> float:
    """
    Derive Hawkes μ from the chaos intensity parameter.

    For probability-based failures (missing_data, feature_dropout,
    batch_corruption), intensity IS the failure probability p, so:
        μ = -ln(1 - p)

    For other failure types, we map the intensity to a probability:
        - noise_injection: σ → p ~ min(σ/5, 0.5)  (heuristic)
        - store_removal: n_stores → p ~ n/10
        - temporal_outage: days → p ~ days/28
        - demand_spike: α_mult → p ~ min((α-1)/10, 0.5)

    Reference: Poisson process inverse link function.
    """
    if failure_type in ('missing_data', 'feature_dropout', 'batch_corruption'):
        # Intensity IS the failure probability — direct mapping
        return mu_from_intensity(intensity)

    elif failure_type == 'noise_injection':
        # σ_scale: map to probability heuristic
        p = min(intensity / 5.0, 0.5)
        return mu_from_intensity(max(p, 0.01))

    elif failure_type == 'store_removal':
        # n_stores: fraction of 10 total stores
        p = min(intensity / 10.0, 0.9)
        return mu_from_intensity(max(p, 0.01))

    elif failure_type == 'temporal_outage':
        # outage_days: fraction of 28-day horizon
        p = min(intensity / 28.0, 0.9)
        return mu_from_intensity(max(p, 0.01))

    elif failure_type == 'demand_spike':
        # multiplier: excess above 1 mapped to probability
        p = min((intensity - 1.0) / 10.0, 0.5)
        return mu_from_intensity(max(p, 0.01))

    else:
        # Fallback: use intensity directly if it's in (0, 1)
        if 0 < intensity < 1:
            return mu_from_intensity(intensity)
        return mu_from_intensity(0.1)


def _create_hawkes_for_experiment(
    experiment: Dict,
) -> Optional[HawkesProcess]:
    """
    Create a HawkesProcess instance from experiment specification.

    μ is derived from intensity (already computed in enumerate_experiments).
    Returns None if alpha == 0 (Bernoulli fallback) or baseline.

    Parameters
    ----------
    experiment : dict
        Must contain 'mu', 'alpha', 'beta', 'seed'.

    Returns
    -------
    HawkesProcess or None
    """
    alpha = experiment.get('alpha', 0.0)
    mu = experiment.get('mu', 0.0)
    beta = experiment.get('beta', 1.0)

    if alpha == 0.0 or experiment['failure_type'] == 'baseline':
        return None

    params = HawkesParams(mu=mu, alpha=alpha, beta=beta)
    params.validate_subcritical()  # warn if α/β ≥ 1
    return HawkesProcess(params=params, seed=experiment['seed'])


def inject_fault(
    experiment: Dict,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    feature_cols: list,
    df_full: pd.DataFrame = None,
) -> Tuple[pd.DataFrame, np.ndarray, Optional[HawkesProcess]]:
    """
    Apply the specified fault injection to test data.

    Now returns a 3-tuple including the HawkesProcess instance
    (for intensity trace logging).

    Parameters
    ----------
    experiment : dict
        Experiment specification with 'failure_type', 'intensity',
        'seed', 'mu', 'alpha', 'beta'.
    X_test : pd.DataFrame
        Test feature matrix.
    y_test : np.ndarray
        Test target values.
    feature_cols : list
        Numeric feature columns.
    df_full : pd.DataFrame, optional
        Full dataset (needed for Chaos Kong store removal).

    Returns
    -------
    (X_perturbed, y_perturbed, hawkes_process) : tuple
        hawkes_process is None for baseline or when Hawkes is disabled.
    """
    ft = experiment['failure_type']
    intensity = experiment['intensity']
    seed = experiment['seed']

    if ft == 'baseline':
        return X_test, y_test, None

    # Create Hawkes process for this experiment (μ already derived)
    hawkes = _create_hawkes_for_experiment(experiment)

    # ─── Chaos Monkey perturbations ─────────────────────────────────

    if ft == 'missing_data':
        lag_roll_cols = [c for c in feature_cols if any(
            c.startswith(p) for p in ('lag_', 'rmean_', 'rstd_')
        )]
        from src.chaos.chaos_monkey import apply_feature_dropout
        X_out = apply_feature_dropout(
            X_test, lag_roll_cols, p=intensity, seed=seed, hawkes=hawkes
        )
        return X_out, y_test, hawkes

    if ft in ('noise_injection', 'feature_dropout', 'batch_corruption'):
        X_out, y_out = apply_chaos_monkey(
            chaos_type=ft,
            X=X_test,
            y=y_test,
            feature_cols=feature_cols,
            intensity=intensity,
            seed=seed,
            hawkes=hawkes,
        )
        return X_out, y_out, hawkes

    # ─── Chaos Kong perturbations ───────────────────────────────────

    if ft == 'demand_spike':
        y_spiked, _ = apply_demand_spike(
            y=y_test,
            alpha=intensity,
            spike_fraction=DEMAND_SPIKE_FRAC,
            seed=seed,
            hawkes=hawkes,
        )
        return X_test, y_spiked, hawkes

    if ft == 'temporal_outage':
        n = len(y_test)
        rng = np.random.RandomState(seed)
        n_zero = min(int(intensity / 28 * n), n)
        zero_idx = rng.choice(n, size=n_zero, replace=False)
        y_out = y_test.copy()
        y_out[zero_idx] = 0

        # Record these outage events in Hawkes history (using sorted
        # time-indices for proper temporal ordering)
        if hawkes is not None:
            sorted_idx = np.sort(zero_idx)
            for idx in sorted_idx:
                hawkes.record_event(float(idx))

        return X_test, y_out, hawkes

    if ft == 'store_removal':
        if 'store_id' in X_test.columns:
            rng = np.random.RandomState(seed)
            all_stores = X_test['store_id'].unique()
            n_remove = min(int(intensity), len(all_stores) - 1)
            removed = rng.choice(all_stores, size=n_remove, replace=False)

            # Hawkes cascading: additional stores may be removed
            # FIX: don't break on first survivor — evaluate ALL remaining
            if hawkes is not None:
                remaining = [s for s in all_stores if s not in removed]
                extra_removed = []
                for step, store in enumerate(remaining):
                    hawkes.record_event(float(step))  # base removals excite
                    p = hawkes.failure_probability(float(n_remove + step))
                    if rng.uniform() < p:
                        extra_removed.append(store)
                        hawkes.record_event(float(n_remove + step))
                    # NO break — continue evaluating all remaining stores
                removed = list(removed) + extra_removed

            keep_mask = ~X_test['store_id'].isin(removed)
            return X_test[keep_mask].copy(), y_test[keep_mask.values], hawkes

        elif 'store_id_enc' in X_test.columns:
            rng = np.random.RandomState(seed)
            all_stores = X_test['store_id_enc'].unique()
            n_remove = min(int(intensity), len(all_stores) - 1)
            removed = rng.choice(all_stores, size=n_remove, replace=False)

            if hawkes is not None:
                remaining = [s for s in all_stores if s not in removed]
                extra_removed = []
                for step, store in enumerate(remaining):
                    hawkes.record_event(float(step))
                    p = hawkes.failure_probability(float(n_remove + step))
                    if rng.uniform() < p:
                        extra_removed.append(store)
                        hawkes.record_event(float(n_remove + step))
                    # NO break — continue evaluating all remaining stores
                removed = list(removed) + extra_removed

            keep_mask = ~X_test['store_id_enc'].isin(removed)
            return X_test[keep_mask].copy(), y_test[keep_mask.values], hawkes
        else:
            print(f"[FaultInjection] WARNING: store_id not in test data, "
                  f"skipping store_removal")
            return X_test, y_test, hawkes

    raise ValueError(f"Unknown failure_type: {ft}")


def get_experiment_label(experiment: Dict) -> str:
    """
    Get a human-readable label for an experiment.

    Includes Hawkes parameters when they are non-zero.
    """
    ft = experiment['failure_type']
    intensity = experiment['intensity']
    alpha = experiment.get('alpha', 0.0)
    beta = experiment.get('beta', 1.0)
    mu = experiment.get('mu', 0.0)

    labels = {
        'baseline': 'Baseline (no chaos)',
        'missing_data': f'Missing Data (p={intensity})',
        'noise_injection': f'Noise (σ={intensity})',
        'feature_dropout': f'Feature Dropout (p={intensity})',
        'batch_corruption': f'Batch Corruption (f={intensity})',
        'store_removal': f'Store Removal (n={int(intensity)})',
        'temporal_outage': f'Temporal Outage (Δ={int(intensity)}d)',
        'demand_spike': f'Demand Spike (α={intensity})',
    }

    label = labels.get(ft, f'{ft}({intensity})')

    # Append Hawkes params if active
    if alpha > 0:
        n_ratio = alpha / beta if beta > 0 else float('inf')
        label += f' [H: μ={mu:.3f}, α={alpha}, β={beta}, n={n_ratio:.2f}]'

    return label
