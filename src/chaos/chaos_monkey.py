"""
M5 Chaos Forecasting — Chaos Monkey (Random Failures) + Hawkes Process
========================================================================

Models random failures with optional self-exciting cascading behaviour.

Mathematical Framework
----------------------

**Original (Bernoulli):**  All perturbations use i.i.d. draws with constant p.

**Hawkes-extended:**  Failure probability varies over time:

    λ(t) = μ + Σ_{t_k < t} α · exp(-β · (t - t_k))
    p(t) = 1 - exp(-λ(t) · Δt)

KEY DESIGN (literature-grounded):
    μ is derived from the existing intensity parameter:
        μ = -ln(1 - p_base)
    This ensures the Hawkes baseline rate matches the intended intensity.
    α and β then modulate this baseline with temporal correlation.

    Reference: Poisson process theory; Bacry et al. (2015).

When α = 0 (no excitation), this reduces to constant-rate Bernoulli,
preserving full backward compatibility.

Perturbation Types
------------------

A. Missing Data (Hawkes-driven mask):
   μ = -ln(1 - p), then p(t) from Hawkes varies around p_base.
   M_{i,t} ~ Bernoulli(1 - p(t))

B. Gaussian Noise Injection (intensity-scaled):
   σ_eff(t) = σ_base · (λ(t) / μ)  — noise scales with cascade intensity.

C. Random Feature Dropout (Hawkes-driven):
   μ = -ln(1 - p_dropout), then p(t) from Hawkes varies per timestep.

D. Random Batch Corruption (Hawkes-driven):
   μ = -ln(1 - fraction), then corruption probability from Hawkes.

Event Recording Convention (consistent across all types):
    Events are recorded ONLY when an actual failure/perturbation occurs,
    using the temporal index (row index) as the event time.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional

from src.chaos.hawkes_process import HawkesProcess


def apply_missing_data(
    y: np.ndarray,
    p: float,
    seed: int = 42,
    hawkes: Optional[HawkesProcess] = None,
) -> np.ndarray:
    """
    Apply missing data mask — Hawkes-driven or Bernoulli.

    When hawkes is provided:
        μ is already set from p via μ = -ln(1-p), so at t=0
        the failure rate matches p. Self-excitation then causes
        temporal clustering around the intended base rate.

    When hawkes is None:
        Falls back to i.i.d. Bernoulli(1-p) — original behaviour.

    Parameters
    ----------
    y : np.ndarray
        Target values (sales). Can be 1D or 2D.
    p : float
        Base probability of each value being missing.
        When hawkes is provided: μ is already derived from this.
        When hawkes is None: used directly as Bernoulli parameter.
    seed : int
        Random seed for reproducibility.
    hawkes : HawkesProcess, optional
        Hawkes engine for self-exciting failures.

    Returns
    -------
    np.ndarray : corrupted target values (same shape as y)
    """
    if hawkes is not None:
        # Hawkes-driven missing data (μ already derived from p)
        if y.ndim == 2:
            mask = hawkes.simulate_2d(y.shape[0], y.shape[1])
        else:
            mask = hawkes.simulate(len(y))
        return y * mask.astype(y.dtype)
    else:
        # Original Bernoulli
        rng = np.random.RandomState(seed)
        mask = rng.binomial(1, 1 - p, size=y.shape).astype(y.dtype)
        return y * mask


def apply_noise_injection(
    X: pd.DataFrame,
    feature_cols: list,
    sigma_scale: float,
    seed: int = 42,
    hawkes: Optional[HawkesProcess] = None,
) -> pd.DataFrame:
    """
    Inject Gaussian noise — intensity-scaled when Hawkes is active.

    When hawkes is provided:
        σ_eff(t) = σ_base · (λ(t) / μ)
        Noise magnitude increases during cascade bursts.

    When hawkes is None:
        x' = x + ε,  ε ~ N(0, (σ_scale · σ_feature)²) — original.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    feature_cols : list
        Columns to perturb (numeric features only).
    sigma_scale : float
        Noise multiplier relative to feature std.
    seed : int
        Random seed.
    hawkes : HawkesProcess, optional
        Hawkes engine for intensity-scaled noise.

    Returns
    -------
    pd.DataFrame : copy of X with noise added.
    """
    rng = np.random.RandomState(seed)
    X_noisy = X.copy()

    if hawkes is not None:
        # Intensity-scaled noise: σ_eff(t) = σ_base · (λ(t) / μ)
        # μ is already derived from sigma_scale in the experiment setup.
        # scale_factor = λ(t)/μ ≥ 1, amplifying noise during cascades.
        n = len(X_noisy)
        scale_factors = np.ones(n)
        hawkes.intensity_trace = []

        for t in range(n):
            t_float = float(t)
            lam = hawkes.compute_intensity(t_float)
            hawkes.intensity_trace.append(lam)
            sf = lam / hawkes.params.mu if hawkes.params.mu > 0 else 1.0
            scale_factors[t] = sf

            # Record event only when noise is above a significant threshold
            # (scale_factor > 1.5 means cascade has amplified noise 50%+)
            if sf > 1.5 and rng.uniform() < 0.3:
                hawkes.record_event(t_float)

        for col in feature_cols:
            if col in X_noisy.columns:
                col_std = X_noisy[col].std()
                if col_std > 0:
                    base_noise = rng.normal(0, sigma_scale * col_std, size=n)
                    noise = base_noise * scale_factors
                    X_noisy[col] = X_noisy[col] + noise
    else:
        # Original Bernoulli-era noise injection
        for col in feature_cols:
            if col in X_noisy.columns:
                col_std = X_noisy[col].std()
                if col_std > 0:
                    noise = rng.normal(0, sigma_scale * col_std, size=len(X_noisy))
                    X_noisy[col] = X_noisy[col] + noise

    return X_noisy


def apply_feature_dropout(
    X: pd.DataFrame,
    feature_cols: list,
    p: float,
    seed: int = 42,
    hawkes: Optional[HawkesProcess] = None,
) -> pd.DataFrame:
    """
    Randomly zero out features — Hawkes-driven or Bernoulli.

    When hawkes is provided:
        Dropout probability p(t) varies per row based on Hawkes intensity.

    When hawkes is None:
        i.i.d. Bernoulli(1-p) dropout — original.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    feature_cols : list
        Columns eligible for dropout.
    p : float
        Base dropout probability (used when hawkes is None).
    seed : int
        Random seed.
    hawkes : HawkesProcess, optional
        Hawkes engine.

    Returns
    -------
    pd.DataFrame : copy of X with dropout applied.
    """
    rng = np.random.RandomState(seed)
    X_dropped = X.copy()

    if hawkes is not None:
        # Hawkes-driven dropout: p(t) from Hawkes intensity
        # μ is already derived from p via μ = -ln(1-p), so
        # p(0) ≈ p at time 0, then varies with cascade.
        n = len(X_dropped)
        hawkes.intensity_trace = []

        for t in range(n):
            t_float = float(t)
            p_t = hawkes.failure_probability(t_float)
            hawkes.intensity_trace.append(hawkes.compute_intensity(t_float))

            # For each feature, decide independently whether to drop
            any_dropped = False
            for col in feature_cols:
                if col in X_dropped.columns:
                    if rng.uniform() < p_t:
                        X_dropped.iloc[t, X_dropped.columns.get_loc(col)] = 0
                        any_dropped = True

            # Record event when ANY feature was actually dropped
            if any_dropped:
                hawkes.record_event(t_float)
    else:
        # Original Bernoulli
        for col in feature_cols:
            if col in X_dropped.columns:
                mask = rng.binomial(1, 1 - p, size=len(X_dropped))
                X_dropped[col] = X_dropped[col] * mask

    return X_dropped


def apply_batch_corruption(
    X: pd.DataFrame,
    y: np.ndarray,
    fraction: float,
    seed: int = 42,
    hawkes: Optional[HawkesProcess] = None,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Corrupt features for a random fraction of test samples — with
    optional Hawkes-driven corruption intensity.

    When hawkes is provided:
        Corruption fraction scales with λ(t)/μ at each batch position.
        Corruption events self-excite.

    When hawkes is None:
        Fixed fraction of rows corrupted — original behaviour.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : np.ndarray
        Target values (returned unchanged).
    fraction : float
        Base fraction of rows to corrupt.
    seed : int
        Random seed.
    hawkes : HawkesProcess, optional
        Hawkes engine.

    Returns
    -------
    (X_corrupted, y) : tuple
    """
    rng = np.random.RandomState(seed)
    X_corrupted = X.copy()
    n = len(X)
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    if hawkes is not None:
        # Hawkes-driven corruption: p(t) from Hawkes intensity.
        # μ is already derived from 'fraction' via μ = -ln(1-fraction).
        hawkes.intensity_trace = []
        for t in range(n):
            t_float = float(t)
            lam = hawkes.compute_intensity(t_float)
            hawkes.intensity_trace.append(lam)

            # Failure probability from Hawkes intensity
            p_corrupt = 1.0 - np.exp(-lam * hawkes.dt)
            p_corrupt = np.clip(p_corrupt, 0.0, 1.0)

            if rng.uniform() < p_corrupt:
                # Corrupt this row — record as single event
                for col in numeric_cols:
                    col_std = X[col].std()
                    if col_std > 0:
                        noise = rng.normal(0, 2 * col_std)
                        X_corrupted.iloc[t, X_corrupted.columns.get_loc(col)] += noise
                hawkes.record_event(t_float)  # one event per row, not per column
    else:
        # Original: fixed-fraction corruption
        n_corrupt = int(n * fraction)
        corrupt_idx = rng.choice(n, size=n_corrupt, replace=False)

        for col in numeric_cols:
            col_std = X[col].std()
            if col_std > 0:
                noise = rng.normal(0, 2 * col_std, size=n_corrupt)
                X_corrupted.iloc[corrupt_idx, X_corrupted.columns.get_loc(col)] += noise

    return X_corrupted, y


# ═══════════════════════════════════════════════════════════════════════════════
# Unified Interface
# ═══════════════════════════════════════════════════════════════════════════════

CHAOS_MONKEY_REGISTRY = {
    'missing_data': {
        'func': apply_missing_data,
        'target': 'y',        # perturbs target
        'description': 'Bernoulli/Hawkes mask on target values',
    },
    'noise_injection': {
        'func': apply_noise_injection,
        'target': 'X',        # perturbs features
        'description': 'Gaussian noise injection (intensity-scaled with Hawkes)',
    },
    'feature_dropout': {
        'func': apply_feature_dropout,
        'target': 'X',        # perturbs features
        'description': 'Random zeroing of feature values (Hawkes-driven)',
    },
    'batch_corruption': {
        'func': apply_batch_corruption,
        'target': 'Xy',       # perturbs both
        'description': 'Corrupt random fraction of training targets (Hawkes-driven)',
    },
}


def apply_chaos_monkey(
    chaos_type: str,
    X: pd.DataFrame,
    y: np.ndarray,
    feature_cols: list,
    intensity: float,
    seed: int = 42,
    apply_to: str = 'test',
    hawkes: Optional[HawkesProcess] = None,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Unified interface for all Chaos Monkey perturbations.

    Parameters
    ----------
    chaos_type : str
        One of: 'missing_data', 'noise_injection', 'feature_dropout',
                'batch_corruption'
    X : pd.DataFrame
        Feature matrix.
    y : np.ndarray
        Target values.
    feature_cols : list
        Numeric feature columns.
    intensity : float
        Perturbation intensity (meaning depends on chaos_type).
    seed : int
        Random seed.
    apply_to : str
        'train' or 'test' — determines which set is perturbed.
    hawkes : HawkesProcess, optional
        Hawkes engine for self-exciting failures. When None, uses
        original Bernoulli-based perturbations.

    Returns
    -------
    (X_perturbed, y_perturbed) : tuple
    """
    if chaos_type == 'missing_data':
        y_out = apply_missing_data(y, p=intensity, seed=seed, hawkes=hawkes)
        return X, y_out

    elif chaos_type == 'noise_injection':
        X_out = apply_noise_injection(
            X, feature_cols, sigma_scale=intensity, seed=seed, hawkes=hawkes
        )
        return X_out, y

    elif chaos_type == 'feature_dropout':
        X_out = apply_feature_dropout(
            X, feature_cols, p=intensity, seed=seed, hawkes=hawkes
        )
        return X_out, y

    elif chaos_type == 'batch_corruption':
        X_out, y_out = apply_batch_corruption(
            X, y, fraction=intensity, seed=seed, hawkes=hawkes
        )
        return X_out, y_out

    else:
        raise ValueError(f"Unknown chaos_type: {chaos_type}. "
                         f"Must be one of {list(CHAOS_MONKEY_REGISTRY.keys())}")
