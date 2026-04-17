"""
M5 Chaos Forecasting — Chaos Kong (Large-Scale Failures) + Hawkes Process
===========================================================================

Models catastrophic, large-scale failures with optional cascading behaviour.

Mathematical Framework
----------------------

Chaos Kong with Hawkes models CASCADING COVARIATE SHIFT:

    P_train(X, Y) ≠ P_test(X, Y)

When Hawkes is active, one large-scale failure EXCITES subsequent failures:
- A store shutdown increases the probability of more store shutdowns
- A temporal outage triggers cascading outages
- A demand spike propagates panic-buying contagion

μ is DERIVED from the intensity parameter (not independent):
    μ = -ln(1 - p_base)
This ensures Hawkes baseline matches the intended chaos intensity.

References:
    - Bacry, Mastromatteo & Muzy (2015): Hawkes processes in finance
    - Watts (2002): Simple model of global cascades on random networks
    - Bao & Bhatt (2014): EM-based Hawkes for infrastructure cascades

Perturbation Types (Hawkes-extended)
-------------------------------------

A. Store/Category Removal (cascading):
   First store removal triggers excitation → evaluates ALL remaining
   stores for cascading removal. Models supply-chain contagion.

B. Temporal Outage (clustered):
   Outage events cluster in time — one outage excites subsequent ones.
   Models cascading system failures where recovery is unstable.

C. Demand Spike (contagion):
   Spike events use a per-item cascade with proper temporal ordering.
   Models panic-buying contagion and viral product surges.

Event Recording Convention:
   Events are recorded as the sequential step index (not item ID),
   ensuring proper temporal ordering for the Hawkes kernel.

When hawkes=None, all functions behave identically to the original
implementation (backward compatible).
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional

from src.chaos.hawkes_process import HawkesProcess


def apply_store_removal(
    df: pd.DataFrame,
    n_stores: int,
    seed: int = 42,
    hawkes: Optional[HawkesProcess] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Remove stores — with optional Hawkes-driven cascading removal.

    When hawkes is provided:
        Each store removal is an "event". After the first removal,
        Hawkes excitation increases the probability of removing
        additional stores beyond the original n_stores target.

    When hawkes is None:
        Remove exactly n_stores randomly — original behaviour.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format data with 'store_id' column.
    n_stores : int
        Base number of stores to remove.
    seed : int
        Random seed.
    hawkes : HawkesProcess, optional
        Hawkes engine for cascading store failures.

    Returns
    -------
    (df_filtered, removed_stores) : tuple
    """
    rng = np.random.RandomState(seed)
    all_stores = df['store_id'].unique()

    if hawkes is not None:
        # Hawkes-driven cascading removal
        removed_stores = []
        available_stores = list(all_stores)

        for step in range(len(available_stores) - 1):  # keep at least 1
            p_remove = hawkes.failure_probability(float(step))

            if step < n_stores:
                # Guaranteed removal for first n_stores
                do_remove = True
            else:
                # Additional removals driven by Hawkes excitation
                do_remove = rng.uniform() < p_remove

            if do_remove:
                chosen = rng.choice(available_stores)
                removed_stores.append(chosen)
                available_stores.remove(chosen)
                hawkes.record_event(float(step))
            # No break — evaluate ALL remaining stores.
            # Previous code broke on first survivor, which is incorrect:
            # a Hawkes process doesn't stop cascading just because one
            # entity survives.
            # Reference: Watts (2002) — cascade continues through network.

        if not removed_stores:
            # Ensure at least one removal
            chosen = rng.choice(available_stores)
            removed_stores.append(chosen)
            hawkes.record_event(0.0)
    else:
        # Original: remove exactly n_stores
        n_to_remove = min(n_stores, len(all_stores) - 1)
        removed_stores = rng.choice(
            all_stores, size=n_to_remove, replace=False
        ).tolist()

    df_filtered = df[~df['store_id'].isin(removed_stores)].copy()

    print(f"[ChaosKong] Removed {len(removed_stores)} stores: {removed_stores}"
          + (" (Hawkes cascade)" if hawkes else ""))
    print(f"[ChaosKong] Remaining data: {len(df_filtered):,} rows "
          f"(from {len(df):,})")

    return df_filtered, removed_stores


def apply_category_removal(
    df: pd.DataFrame,
    categories: List[str] = None,
    n_depts: int = None,
    seed: int = 42,
    hawkes: Optional[HawkesProcess] = None,
) -> Tuple[pd.DataFrame, list]:
    """
    Remove categories or departments — with optional cascading.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format data.
    categories : list of str, optional
        Specific category names to remove.
    n_depts : int, optional
        Number of departments to randomly remove.
    seed : int
    hawkes : HawkesProcess, optional

    Returns
    -------
    (df_filtered, removed) : tuple
    """
    rng = np.random.RandomState(seed)

    if categories is not None:
        removed = list(categories)

        # Hawkes: additional categories may cascade
        if hawkes is not None:
            all_cats = df['cat_id'].unique()
            remaining = [c for c in all_cats if c not in removed]
            for step, _ in enumerate(remaining):
                hawkes.record_event(float(step))  # initial removals excite
                p = hawkes.failure_probability(float(len(removed) + step))
                if rng.uniform() < p and remaining:
                    extra = rng.choice(remaining)
                    removed.append(extra)
                    remaining.remove(extra)
                    hawkes.record_event(float(len(removed) + step))

        df_filtered = df[~df['cat_id'].isin(removed)].copy()

    elif n_depts is not None:
        all_depts = df['dept_id'].unique()

        if hawkes is not None:
            removed = []
            available = list(all_depts)
            for step in range(len(available) - 1):
                p = hawkes.failure_probability(float(step))
                if step < n_depts or rng.uniform() < p:
                    chosen = rng.choice(available)
                    removed.append(chosen)
                    available.remove(chosen)
                    hawkes.record_event(float(step))
                else:
                    break
        else:
            n_to_remove = min(n_depts, len(all_depts) - 1)
            removed = rng.choice(
                all_depts, size=n_to_remove, replace=False
            ).tolist()

        df_filtered = df[~df['dept_id'].isin(removed)].copy()
    else:
        raise ValueError("Specify either 'categories' or 'n_depts'.")

    print(f"[ChaosKong] Removed: {removed}"
          + (" (Hawkes cascade)" if hawkes else ""))
    return df_filtered, removed


def apply_temporal_outage(
    y_matrix: np.ndarray,
    outage_days: int,
    start_offset: int = None,
    seed: int = 42,
    hawkes: Optional[HawkesProcess] = None,
) -> np.ndarray:
    """
    Zero out sales — with Hawkes-driven clustered outages.

    When hawkes is provided:
        Outage events cluster in time. The first outage window excites
        additional outage windows, modelling cascading system failures.

    When hawkes is None:
        Single contiguous outage window — original behaviour.

    Parameters
    ----------
    y_matrix : np.ndarray of shape (n_series, n_days)
        Wide-format sales data.
    outage_days : int (Δ)
        Duration of each outage event.
    start_offset : int, optional
    seed : int
    hawkes : HawkesProcess, optional

    Returns
    -------
    np.ndarray : modified sales matrix with outage windows zeroed.
    """
    rng = np.random.RandomState(seed)
    n_series, n_days = y_matrix.shape
    y_out = y_matrix.copy()

    if hawkes is not None:
        # Hawkes-driven clustered outages
        t = 0
        outage_count = 0
        hawkes.intensity_trace = []

        while t < n_days:
            lam = hawkes.compute_intensity(float(t))
            hawkes.intensity_trace.append(lam)
            p = 1.0 - np.exp(-lam * hawkes.dt)

            if outage_count == 0 or rng.uniform() < p:
                # Apply outage at this position
                t_end = min(t + outage_days, n_days)
                y_out[:, t:t_end] = 0
                hawkes.record_event(float(t))
                outage_count += 1

                print(f"[ChaosKong] Hawkes temporal outage #{outage_count}: "
                      f"days [{t}, {t_end}] ({t_end - t} days, "
                      f"{n_series} series, λ={lam:.3f})")

                t = t_end + 1  # Skip past this outage
            else:
                t += 1

            # Safety: limit total outages to prevent wiping all data
            if outage_count >= 5:
                break
    else:
        # Original: single contiguous outage
        if start_offset is not None:
            t0 = max(0, n_days - start_offset)
        else:
            earliest_start = max(0, n_days - n_days // 3)
            latest_start = max(0, n_days - outage_days)
            if earliest_start >= latest_start:
                t0 = latest_start
            else:
                t0 = rng.randint(earliest_start, latest_start + 1)

        t_end = min(t0 + outage_days, n_days)
        y_out[:, t0:t_end] = 0

        print(f"[ChaosKong] Temporal outage: days [{t0}, {t_end}] "
              f"({t_end - t0} days zeroed, {n_series} series)")

    return y_out


def apply_demand_spike(
    y: np.ndarray,
    alpha: float,
    spike_fraction: float = 0.1,
    seed: int = 42,
    series_ids: np.ndarray = None,
    hawkes: Optional[HawkesProcess] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply demand spike — with Hawkes-driven contagion spread.

    When hawkes is provided:
        Initial spike excites additional spikes. Models panic-buying
        contagion where one product's spike triggers nearby products.

    When hawkes is None:
        Fixed fraction of items spiked — original behaviour.

    Parameters
    ----------
    y : np.ndarray
        Target values. Can be 1D or 2D.
    alpha : float
        Demand multiplier (α >> 1 for spikes).
    spike_fraction : float
        Base fraction of series to spike.
    seed : int
    series_ids : np.ndarray, optional
    hawkes : HawkesProcess, optional

    Returns
    -------
    (y_spiked, spiked_mask) : tuple
    """
    rng = np.random.RandomState(seed)
    y_spiked = y.copy().astype(np.float64)

    if y.ndim == 2:
        n_series = y.shape[0]
    else:
        n_series = len(y)

    if hawkes is not None:
        # Hawkes-driven contagion: iterate over items using a STEP-BASED
        # cascade (sequential step index as time, NOT item ID).
        # This ensures proper temporal ordering for the Hawkes kernel.
        #
        # Reference: Watts (2002) — cascade model on random networks.
        spiked_mask = np.zeros(n_series, dtype=bool)
        n_base_spike = max(1, int(n_series * spike_fraction))

        # First wave: guaranteed spikes
        first_wave = rng.choice(n_series, size=n_base_spike, replace=False)
        spiked_mask[first_wave] = True

        # Record first-wave events using sequential step indices
        for step, idx in enumerate(first_wave):
            hawkes.record_event(float(step))

        # Cascading spikes: iterate remaining items in random order,
        # using step counter as Hawkes time (not item index)
        remaining_items = [i for i in range(n_series) if not spiked_mask[i]]
        rng.shuffle(remaining_items)  # random order for fairness

        step_offset = len(first_wave)
        for step, i in enumerate(remaining_items):
            t = float(step_offset + step)
            p = hawkes.failure_probability(t)
            if rng.uniform() < p:
                spiked_mask[i] = True
                hawkes.record_event(t)

        if y.ndim == 2:
            y_spiked[spiked_mask] = y_spiked[spiked_mask] * alpha
        else:
            y_spiked[spiked_mask] = y_spiked[spiked_mask] * alpha

    else:
        # Original: fixed fraction
        if y.ndim == 2:
            n_spike = max(1, int(n_series * spike_fraction))
            spike_idx = rng.choice(n_series, size=n_spike, replace=False)
            spiked_mask = np.zeros(n_series, dtype=bool)
            spiked_mask[spike_idx] = True
            y_spiked[spike_idx] = y_spiked[spike_idx] * alpha
        else:
            n_spike = max(1, int(n_series * spike_fraction))
            spike_idx = rng.choice(n_series, size=n_spike, replace=False)
            spiked_mask = np.zeros(n_series, dtype=bool)
            spiked_mask[spike_idx] = True
            y_spiked[spike_idx] = y_spiked[spike_idx] * alpha

    print(f"[ChaosKong] Demand spike: α={alpha}, "
          f"{spiked_mask.sum()}/{len(spiked_mask)} items spiked"
          + (" (Hawkes contagion)" if hawkes else ""))

    return y_spiked, spiked_mask


# ═══════════════════════════════════════════════════════════════════════════════
# Unified Interface
# ═══════════════════════════════════════════════════════════════════════════════

CHAOS_KONG_REGISTRY = {
    'store_removal': {
        'func': apply_store_removal,
        'description': 'Remove entire stores (cascading with Hawkes)',
    },
    'temporal_outage': {
        'func': apply_temporal_outage,
        'description': 'Zero out contiguous time windows (clustered with Hawkes)',
    },
    'demand_spike': {
        'func': apply_demand_spike,
        'description': 'Multiply demand by large factor (contagion with Hawkes)',
    },
}
