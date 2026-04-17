"""
M5 Chaos Forecasting — Hawkes Process Engine
===============================================

Self-exciting point process for realistic cascading failure simulation.

Mathematical Framework
----------------------

A Hawkes Process is a self-exciting point process whose conditional
intensity function depends on the history of past events:

    λ(t) = μ + Σ_{t_k < t} α · exp(-β · (t - t_k))

Where:
    μ  ≥ 0 : baseline (background) failure rate
    α  ≥ 0 : excitation strength (jump size when an event occurs)
    β  > 0 : decay rate (how quickly excitation fades)

The process is stationary if and only if the branching ratio α/β < 1,
meaning each event triggers on average fewer than one child event.

Failure Probability (Poisson link)
-----------------------------------
At each discrete timestep with interval Δt:

    p(t) = 1 - exp(-λ(t) · Δt)

This is derived from the probability of at least one event in a small
interval under a Poisson process with rate λ(t).

When α = 0:
    λ(t) = μ  (constant), and
    p(t) = 1 - exp(-μ · Δt) ≈ μ · Δt  for small μ·Δt

This recovers the memoryless Bernoulli draw, making Hawkes a strict
generalisation of the existing chaos framework.

MLE Parameter Fitting
---------------------
Given observed event times {t_1, ..., t_n} in [0, T], the log-likelihood is:

    ℓ(μ, α, β) = Σ_{i=1}^{n} log λ(t_i) - ∫_0^T λ(s) ds

The integral has a closed form:
    ∫_0^T λ(s) ds = μT + (α/β) Σ_{i=1}^{n} [1 - exp(-β(T - t_i))]

We maximise ℓ using scipy.optimize.minimize (L-BFGS-B with bounds).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import json


# ═══════════════════════════════════════════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class HawkesParams:
    """
    Parameters for a univariate Hawkes process.

    Attributes
    ----------
    mu : float
        Baseline (background) intensity. Must be ≥ 0.
    alpha : float
        Excitation strength (jump size per event). Must be ≥ 0.
    beta : float
        Decay rate of excitation. Must be > 0.
    """
    mu: float = 0.1
    alpha: float = 0.6
    beta: float = 1.0

    def __post_init__(self):
        if self.mu < 0:
            raise ValueError(f"mu must be ≥ 0, got {self.mu}")
        if self.alpha < 0:
            raise ValueError(f"alpha must be ≥ 0, got {self.alpha}")
        if self.beta <= 0:
            raise ValueError(f"beta must be > 0, got {self.beta}")

    @property
    def branching_ratio(self) -> float:
        """α/β — must be < 1 for stationarity."""
        return self.alpha / self.beta

    @property
    def is_stationary(self) -> bool:
        return self.branching_ratio < 1.0

    def to_dict(self) -> dict:
        return {'mu': self.mu, 'alpha': self.alpha, 'beta': self.beta}

    @classmethod
    def from_dict(cls, d: dict) -> 'HawkesParams':
        return cls(mu=d['mu'], alpha=d['alpha'], beta=d['beta'])

    @staticmethod
    def mu_from_intensity(p_base: float) -> float:
        """Derive Hawkes baseline rate μ from intended failure probability.

        Uses the Poisson inverse link:  μ = -ln(1 - p_base)

        At Δt=1 and with no prior events:
            p(0) = 1 - exp(-μ · 1) = p_base

        This ensures the Hawkes process's baseline failure rate matches
        the intended Bernoulli probability exactly, making intensity
        parameters meaningful both with and without Hawkes.

        Reference: Standard Poisson process theory.

        Parameters
        ----------
        p_base : float
            Intended baseline failure probability (0, 1).

        Returns
        -------
        float : corresponding μ value
        """
        import math
        p_base = max(1e-8, min(p_base, 1.0 - 1e-8))
        return -math.log(1.0 - p_base)

    def validate_subcritical(self) -> None:
        """Warn if branching ratio α/β ≥ 1 (non-stationary).

        Reference: Bacry, Mastromatteo & Muzy (2015) — stationarity
        condition for univariate Hawkes processes.
        """
        if self.branching_ratio >= 1.0:
            import warnings
            warnings.warn(
                f"Hawkes process is non-stationary: α/β = "
                f"{self.alpha}/{self.beta} = {self.branching_ratio:.2f} ≥ 1. "
                f"Intensity may explode. Consider reducing α or increasing β. "
                f"Ref: Bacry et al. (2015), stationarity condition α/β < 1.",
                RuntimeWarning,
                stacklevel=2,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Core Hawkes Process
# ═══════════════════════════════════════════════════════════════════════════════

class HawkesProcess:
    """
    Univariate Hawkes process simulator with shared event history.

    The process maintains a running list of event times. Each call to
    `failure_probability(t)` computes λ(t) from the full history,
    and `record_event(t)` appends to the history so that excitation
    carries over across chaos stages in the pipeline.

    Parameters
    ----------
    params : HawkesParams
        Process parameters (μ, α, β).
    seed : int
        Random seed for simulation reproducibility.
    dt : float
        Discrete timestep interval (default 1.0 for daily data).
    """

    def __init__(
        self,
        params: HawkesParams,
        seed: int = 42,
        dt: float = 1.0,
    ):
        self.params = params
        self.seed = seed
        self.dt = dt
        self.rng = np.random.RandomState(seed)

        # Shared event history (persists across chaos stages)
        self.event_times: List[float] = []

        # Full intensity trace (filled during simulate())
        self.intensity_trace: List[float] = []

    # ─── Core Intensity Computation ───────────────────────────────────────

    def compute_intensity(self, t: float) -> float:
        """
        Compute the conditional intensity λ(t | H_t).

        λ(t) = μ + Σ_{t_k < t} α · exp(-β · (t - t_k))

        Parameters
        ----------
        t : float
            Current time.

        Returns
        -------
        float : intensity value λ(t)
        """
        mu = self.params.mu
        alpha = self.params.alpha
        beta = self.params.beta

        if alpha == 0 or len(self.event_times) == 0:
            return mu

        # Vectorised computation over event history
        events = np.array(self.event_times)
        past_mask = events < t
        if not past_mask.any():
            return mu

        past_events = events[past_mask]
        excitation = alpha * np.sum(np.exp(-beta * (t - past_events)))

        return mu + excitation

    def failure_probability(self, t: float) -> float:
        """
        Compute failure probability at time t via the Poisson link.

        p(t) = 1 - exp(-λ(t) · Δt)

        Clamped to [0, 1] for numerical safety.
        """
        lam = self.compute_intensity(t)
        p = 1.0 - np.exp(-lam * self.dt)
        return float(np.clip(p, 0.0, 1.0))

    def record_event(self, t: float):
        """Record a failure event at time t (updates shared history)."""
        self.event_times.append(float(t))

    # ─── Simulation ───────────────────────────────────────────────────────

    def simulate(self, n_steps: int) -> np.ndarray:
        """
        Simulate a Hawkes-driven failure mask over n_steps timesteps.

        At each step t:
            1. Compute λ(t) from current event history
            2. Derive p(t) = 1 - exp(-λ(t)·Δt)
            3. Draw U ~ Uniform(0,1); failure if U < p(t)
            4. If failure: record event, set mask[t] = 0 (data lost)

        Returns
        -------
        np.ndarray of shape (n_steps,) : binary mask
            1 = data survives, 0 = data lost (failure occurred)
        """
        mask = np.ones(n_steps, dtype=np.float64)
        self.intensity_trace = []

        for t in range(n_steps):
            t_float = float(t)
            lam = self.compute_intensity(t_float)
            self.intensity_trace.append(lam)

            p = 1.0 - np.exp(-lam * self.dt)
            p = np.clip(p, 0.0, 1.0)

            if self.rng.uniform() < p:
                mask[t] = 0.0
                self.record_event(t_float)

        return mask

    def simulate_2d(self, n_rows: int, n_cols: int) -> np.ndarray:
        """
        Simulate a 2D failure mask (e.g., items × timesteps).

        Each column (timestep) shares the same Hawkes intensity,
        but individual row failures are drawn independently.

        Returns
        -------
        np.ndarray of shape (n_rows, n_cols) : binary mask
        """
        mask = np.ones((n_rows, n_cols), dtype=np.float64)
        self.intensity_trace = []

        for t in range(n_cols):
            t_float = float(t)
            lam = self.compute_intensity(t_float)
            self.intensity_trace.append(lam)

            p = 1.0 - np.exp(-lam * self.dt)
            p = np.clip(p, 0.0, 1.0)

            # Draw failures independently across rows
            failures = self.rng.uniform(size=n_rows) < p
            mask[failures, t] = 0.0

            # If ANY failure occurred at this timestep, record as event
            if failures.any():
                self.record_event(t_float)

        return mask

    # ─── Intensity Scaling ────────────────────────────────────────────────

    def intensity_scale_factor(self, t: float) -> float:
        """
        Ratio of current intensity to baseline: λ(t) / μ.

        Used to scale noise magnitude proportionally to cascade intensity.
        Returns 1.0 if μ = 0 (degenerate case).
        """
        if self.params.mu <= 0:
            return 1.0
        return self.compute_intensity(t) / self.params.mu

    # ─── Accessors ────────────────────────────────────────────────────────

    def get_intensity_trace(self) -> np.ndarray:
        """Return the full λ(t) trace from the last simulation."""
        return np.array(self.intensity_trace)

    def get_summary_stats(self) -> dict:
        """Summary statistics of the intensity trace and event history."""
        trace = self.get_intensity_trace()
        stats = {
            'n_events': len(self.event_times),
            'lambda_mean': float(np.mean(trace)) if len(trace) > 0 else self.params.mu,
            'lambda_max': float(np.max(trace)) if len(trace) > 0 else self.params.mu,
            'lambda_min': float(np.min(trace)) if len(trace) > 0 else self.params.mu,
            'lambda_final': float(trace[-1]) if len(trace) > 0 else self.params.mu,
        }
        return stats

    def reset(self, keep_params: bool = True):
        """Reset event history and intensity trace."""
        self.event_times = []
        self.intensity_trace = []
        if not keep_params:
            self.rng = np.random.RandomState(self.seed)

    def save_trace(self, path: str):
        """Save intensity trace and event list to npz file."""
        np.savez(
            path,
            intensity_trace=np.array(self.intensity_trace),
            event_times=np.array(self.event_times),
            params=np.array([self.params.mu, self.params.alpha, self.params.beta]),
        )

    @staticmethod
    def load_trace(path: str) -> dict:
        """Load saved trace data."""
        data = np.load(path)
        return {
            'intensity_trace': data['intensity_trace'],
            'event_times': data['event_times'],
            'params': data['params'],
        }


# ═══════════════════════════════════════════════════════════════════════════════
# MLE Parameter Fitting
# ═══════════════════════════════════════════════════════════════════════════════

def hawkes_log_likelihood(
    params_vec: np.ndarray,
    event_times: np.ndarray,
    T: float,
) -> float:
    """
    Negative log-likelihood for a univariate Hawkes process.

    ℓ(μ, α, β) = Σ_{i=1}^{n} log λ(t_i) - ∫_0^T λ(s) ds

    Integral closed form:
        ∫_0^T λ(s) ds = μT + (α/β) Σ_i [1 - exp(-β(T - t_i))]

    Parameters
    ----------
    params_vec : array [mu, alpha, beta]
    event_times : array of event times in [0, T]
    T : float, observation window length

    Returns
    -------
    float : NEGATIVE log-likelihood (for minimisation)
    """
    mu, alpha, beta = params_vec
    n = len(event_times)

    if n == 0:
        # No events: ℓ = -μT
        return mu * T

    # Sort events just in case
    times = np.sort(event_times)

    # Term 1: Σ log λ(t_i)
    log_lam_sum = 0.0
    for i in range(n):
        t_i = times[i]
        lam_i = mu
        if i > 0:
            past = times[:i]
            lam_i += alpha * np.sum(np.exp(-beta * (t_i - past)))

        if lam_i <= 1e-15:
            lam_i = 1e-15  # prevent log(0)
        log_lam_sum += np.log(lam_i)

    # Term 2: ∫_0^T λ(s) ds = μT + (α/β) Σ_i [1 - exp(-β(T - t_i))]
    integral = mu * T
    if alpha > 0 and beta > 0:
        integral += (alpha / beta) * np.sum(1.0 - np.exp(-beta * (T - times)))

    # Negative log-likelihood
    nll = -log_lam_sum + integral
    return nll


def fit_hawkes_mle(
    event_times: np.ndarray,
    T: float,
    initial_params: Tuple[float, float, float] = (0.1, 0.5, 1.0),
    bounds: Tuple = ((1e-6, None), (1e-6, None), (1e-3, None)),
) -> HawkesParams:
    """
    Fit Hawkes process parameters via Maximum Likelihood Estimation.

    Maximises ℓ(μ, α, β) using scipy L-BFGS-B with bounds.

    Parameters
    ----------
    event_times : np.ndarray
        Observed event times in [0, T].
    T : float
        Total observation window length.
    initial_params : tuple
        Starting values (mu_0, alpha_0, beta_0).
    bounds : tuple of (min, max) per parameter

    Returns
    -------
    HawkesParams : fitted parameters
    """
    from scipy.optimize import minimize

    event_times = np.sort(np.asarray(event_times, dtype=np.float64))

    if len(event_times) < 2:
        # Too few events to fit — return defaults based on event rate
        rate = len(event_times) / T if T > 0 else 0.1
        return HawkesParams(mu=max(rate, 1e-4), alpha=0.0, beta=1.0)

    result = minimize(
        hawkes_log_likelihood,
        x0=np.array(initial_params),
        args=(event_times, T),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 500, 'ftol': 1e-10},
    )

    mu_fit, alpha_fit, beta_fit = result.x

    fitted = HawkesParams(
        mu=float(mu_fit),
        alpha=float(alpha_fit),
        beta=float(beta_fit),
    )

    return fitted


def fit_hawkes_from_mask(
    failure_mask: np.ndarray,
    dt: float = 1.0,
) -> HawkesParams:
    """
    Fit Hawkes parameters from a binary failure mask.

    Extracts event times from mask positions where mask == 0 (failure),
    then runs MLE fitting.

    Parameters
    ----------
    failure_mask : np.ndarray
        Binary mask (1 = survive, 0 = failure). Can be 1D or 2D.
    dt : float
        Timestep interval.

    Returns
    -------
    HawkesParams : fitted parameters
    """
    if failure_mask.ndim == 2:
        # For 2D mask: collapse across rows, event at time t if ANY row failed
        col_failures = (failure_mask == 0).any(axis=0)
        event_indices = np.where(col_failures)[0]
    else:
        event_indices = np.where(failure_mask == 0)[0]

    event_times = event_indices.astype(np.float64) * dt
    T = float(len(failure_mask) if failure_mask.ndim == 1
              else failure_mask.shape[1]) * dt

    return fit_hawkes_mle(event_times, T)
