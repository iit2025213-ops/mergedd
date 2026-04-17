"""
M5 Supreme — Chaos Config Reader
==================================

Thin adapter that loads configs/chaos_config.yaml and exposes the same
constants that Repo 1's config.py provided. This is the ONLY new code
needed to make the ported chaos modules resolve their imports.

All constants (FAILURE_TYPES, HAWKES_ALPHA_VALUES, mu_from_intensity, etc.)
are re-exported at module level for drop-in compatibility.
"""

import math
import yaml
from pathlib import Path

# ─── Load YAML ────────────────────────────────────────────────────────────────

_CONFIG_PATH = Path(__file__).parent.parent.parent / "configs" / "chaos_config.yaml"

def _load_config():
    with open(_CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)

_CFG = _load_config()

# ─── Project Paths ────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
RESULTS_DIR = PROJECT_ROOT / _CFG['output']['results_dir']
INTENSITY_TRACES_DIR = PROJECT_ROOT / _CFG['output']['intensity_traces_dir']
PLOTS_DIR = PROJECT_ROOT / _CFG['output']['plots_dir']
MODELS_DIR = PROJECT_ROOT / _CFG['output']['models_dir']

# Create directories
for d in [RESULTS_DIR, INTENSITY_TRACES_DIR, PLOTS_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── Hawkes Process Parameters ────────────────────────────────────────────────

USE_HAWKES = _CFG.get('use_hawkes', True)
HAWKES_ALPHA_VALUES = _CFG['hawkes']['alpha_values']
HAWKES_BETA_VALUES = _CFG['hawkes']['beta_values']
HAWKES_DEFAULT = {
    'alpha': _CFG['hawkes']['default_alpha'],
    'beta': _CFG['hawkes']['default_beta'],
}

def mu_from_intensity(p_base: float) -> float:
    """Derive Hawkes baseline rate μ from intended failure probability.

    Uses the Poisson inverse link: μ = -ln(1 - p_base)
    At Δt=1, this gives p(0) = 1 - exp(-μ) = p_base exactly.

    Reference: Standard Poisson process theory.
    """
    p_base = max(1e-8, min(p_base, 1.0 - 1e-8))
    return -math.log(1.0 - p_base)

# ─── Chaos Engineering Parameters ─────────────────────────────────────────────

# Chaos Monkey
MISSING_DATA_PROBS = _CFG['chaos_monkey']['missing_data_probs']
NOISE_SIGMAS = _CFG['chaos_monkey']['noise_sigmas']
FEATURE_DROPOUT_PROBS = _CFG['chaos_monkey']['feature_dropout_probs']
BATCH_CORRUPTION_FRACS = _CFG['chaos_monkey']['batch_corruption_fracs']

# Chaos Kong
STORE_REMOVAL_COUNTS = _CFG['chaos_kong']['store_removal_counts']
TEMPORAL_OUTAGE_DAYS = _CFG['chaos_kong']['temporal_outage_days']
DEMAND_SPIKE_ALPHAS = _CFG['chaos_kong']['demand_spike_alphas']
DEMAND_SPIKE_FRAC = _CFG['chaos_kong']['demand_spike_frac']

# ─── Experiment Settings ──────────────────────────────────────────────────────

RANDOM_SEEDS = _CFG['experiment']['random_seeds']
DEFAULT_SEED = _CFG['experiment']['default_seed']
FAILURE_TYPES = _CFG['experiment']['failure_types']

FAILURE_INTENSITIES = {
    "missing_data": MISSING_DATA_PROBS,
    "noise_injection": NOISE_SIGMAS,
    "feature_dropout": FEATURE_DROPOUT_PROBS,
    "batch_corruption": BATCH_CORRUPTION_FRACS,
    "store_removal": STORE_REMOVAL_COUNTS,
    "temporal_outage": TEMPORAL_OUTAGE_DAYS,
    "demand_spike": DEMAND_SPIKE_ALPHAS,
}

# ─── Data Settings ────────────────────────────────────────────────────────────

HORIZON = _CFG['data']['horizon']
END_TRAIN_DAY = _CFG['data']['end_train_day']
TOTAL_DAYS = _CFG['data']['total_days']
SAMPLE_N_ITEMS = _CFG['data'].get('sample_n_items', None)

# ─── Reproducibility ─────────────────────────────────────────────────────────

def set_global_seed(seed: int = DEFAULT_SEED):
    """Set random seeds for reproducibility across all libraries."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

# ─── Raw Data Paths (from Repo 1, only used by experiment_runner) ─────────────

DATA_RAW_DIR = Path(r"C:\Users\cynic\Downloads\m5-forecasting-accuracy")
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

SALES_TRAIN_VAL_PATH = DATA_RAW_DIR / "sales_train_validation.csv"
SALES_TRAIN_EVAL_PATH = DATA_RAW_DIR / "sales_train_evaluation.csv"
CALENDAR_PATH = DATA_RAW_DIR / "calendar.csv"
SELL_PRICES_PATH = DATA_RAW_DIR / "sell_prices.csv"
