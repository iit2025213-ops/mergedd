"""
M5 Chaos Forecasting — Analysis and Visualization (Hawkes-Extended)
======================================================================

Generates publication-quality plots and deep analysis of chaos engineering
experiment results, including Hawkes process intensity analysis.

Plots Generated
---------------
 1. Performance vs. Failure Intensity — Line plots with error bands
 2. Heatmap — Failure type × intensity → RMSE
 3. Robustness Radar Chart — R(p) for each failure type
 4. Error Variance Analysis — Box plots of per-experiment errors
 5. Sensitivity Ranking — Bar chart of failure type degradation
 6. Model Comparison — Grouped bars (if multiple models)
 7. Robustness Curves — Multi-panel R(p) vs intensity
 8. [NEW] Intensity vs Robustness — λ(t) mean/max vs R scatter
 9. [NEW] Hawkes Intensity Trace — Time-series of λ(t) with event markers
10. [NEW] Hawkes Parameter Heatmap — R over the (α, β) grid
11. [NEW] Adversarial Regime — Identifying worst-case (α, β)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PLOTS_DIR, INTENSITY_TRACES_DIR

# ─── Plot Configuration ──────────────────────────────────────────────────────

plt.rcParams.update({
    'figure.figsize': (12, 8),
    'font.size': 12,
    'font.family': 'sans-serif',
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# Color scheme
COLORS = {
    'baseline': '#2ecc71',
    'missing_data': '#e74c3c',
    'noise_injection': '#3498db',
    'feature_dropout': '#9b59b6',
    'batch_corruption': '#f39c12',
    'store_removal': '#1abc9c',
    'temporal_outage': '#e67e22',
    'demand_spike': '#c0392b',
}

FAILURE_LABELS = {
    'baseline': 'Baseline',
    'missing_data': 'Missing Data',
    'noise_injection': 'Noise Injection',
    'feature_dropout': 'Feature Dropout',
    'batch_corruption': 'Batch Corruption',
    'store_removal': 'Store Removal',
    'temporal_outage': 'Temporal Outage',
    'demand_spike': 'Demand Spike',
}


def load_results(csv_path: str = None) -> pd.DataFrame:
    """Load experiment results from CSV."""
    if csv_path is None:
        csv_path = Path(sys.path[0]).parent / "experiments" / "results" / "experiment_results.csv"
        if not csv_path.exists():
            from config import RESULTS_DIR
            csv_path = RESULTS_DIR / "experiment_results.csv"

    df = pd.read_csv(csv_path)
    print(f"[Analysis] Loaded {len(df)} results from {csv_path}")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 1: Performance vs. Failure Intensity
# ═══════════════════════════════════════════════════════════════════════════════

def plot_performance_vs_intensity(
    results: pd.DataFrame,
    metric: str = 'rmse',
    model: str = 'lgbm',
    save: bool = True,
):
    """
    Line plot showing metric degradation vs. failure intensity.
    Each failure type is a separate line, with error bands (±1 std) across seeds.
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    df = results[results['model'] == model].copy()

    for ft in df['failure_type'].unique():
        if ft == 'baseline':
            continue

        ft_data = df[df['failure_type'] == ft]
        grouped = ft_data.groupby('intensity')[metric].agg(['mean', 'std']).reset_index()

        color = COLORS.get(ft, '#95a5a6')
        label = FAILURE_LABELS.get(ft, ft)

        ax.plot(grouped['intensity'], grouped['mean'],
                'o-', color=color, linewidth=2, markersize=8, label=label)
        ax.fill_between(
            grouped['intensity'],
            grouped['mean'] - grouped['std'],
            grouped['mean'] + grouped['std'],
            alpha=0.2, color=color
        )

    baseline = df[df['failure_type'] == 'baseline'][metric].mean()
    ax.axhline(y=baseline, color=COLORS['baseline'], linestyle='--',
               linewidth=2, label=f'Baseline ({baseline:.4f})')

    ax.set_xlabel('Failure Intensity', fontsize=14)
    ax.set_ylabel(metric.upper(), fontsize=14)
    ax.set_title(f'{model.upper()} — {metric.upper()} vs. Failure Intensity',
                 fontsize=16, fontweight='bold')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    if save:
        path = PLOTS_DIR / f'performance_vs_intensity_{metric}_{model}.png'
        fig.savefig(path)
        print(f"[Plot] Saved: {path}")

    plt.close(fig)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 2: Heatmap — Failure Type × Intensity
# ═══════════════════════════════════════════════════════════════════════════════

def plot_heatmap(
    results: pd.DataFrame,
    metric: str = 'rmse',
    model: str = 'lgbm',
    save: bool = True,
):
    """Heatmap showing metric values across failure types and intensities."""
    df = results[(results['model'] == model) &
                 (results['failure_type'] != 'baseline')].copy()

    if df.empty:
        print("[Plot] No chaos results to plot heatmap.")
        return

    pivot = df.pivot_table(
        index='failure_type', columns='intensity',
        values=metric, aggfunc='mean'
    )

    pivot.index = [FAILURE_LABELS.get(ft, ft) for ft in pivot.index]

    fig, ax = plt.subplots(figsize=(12, 8))

    sns.heatmap(
        pivot, annot=True, fmt='.4f', cmap='YlOrRd',
        linewidths=0.5, ax=ax, cbar_kws={'label': metric.upper()}
    )

    ax.set_title(f'{model.upper()} — {metric.upper()} Heatmap\n'
                 f'(Failure Type × Intensity)',
                 fontsize=16, fontweight='bold')
    ax.set_xlabel('Intensity', fontsize=13)
    ax.set_ylabel('Failure Type', fontsize=13)

    if save:
        path = PLOTS_DIR / f'heatmap_{metric}_{model}.png'
        fig.savefig(path)
        print(f"[Plot] Saved: {path}")

    plt.close(fig)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 3: Robustness Radar Chart
# ═══════════════════════════════════════════════════════════════════════════════

def plot_robustness_radar(
    results: pd.DataFrame,
    model: str = 'lgbm',
    intensity_idx: int = 1,
    save: bool = True,
):
    """Radar chart showing robustness R(p) across failure types."""
    df = results[(results['model'] == model) &
                 (results['failure_type'] != 'baseline')].copy()

    if df.empty:
        return

    categories = []
    values = []

    for ft in df['failure_type'].unique():
        ft_data = df[df['failure_type'] == ft]
        intensities = sorted(ft_data['intensity'].unique())

        idx = min(intensity_idx, len(intensities) - 1)
        target_intensity = intensities[idx]

        robustness = ft_data[ft_data['intensity'] == target_intensity][
            'robustness_rmse'
        ].mean()

        categories.append(FAILURE_LABELS.get(ft, ft))
        values.append(robustness if not np.isnan(robustness) else 0)

    if not categories:
        return

    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    values_plot = values + [values[0]]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    ax.plot(angles, values_plot, 'o-', linewidth=2, color='#e74c3c', markersize=8)
    ax.fill(angles, values_plot, alpha=0.25, color='#e74c3c')

    ref = [1.0] * (N + 1)
    ax.plot(angles, ref, '--', linewidth=1.5, color='#2ecc71', alpha=0.7,
            label='R=1.0 (no degradation)')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, max(1.2, max(values) * 1.1))
    ax.set_title(f'{model.upper()} — Robustness Radar (R = baseline/chaos)\n'
                 f'R < 1.0 = degradation',
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='lower right')

    if save:
        path = PLOTS_DIR / f'robustness_radar_{model}.png'
        fig.savefig(path)
        print(f"[Plot] Saved: {path}")

    plt.close(fig)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 4: Error Variance Analysis (Box Plots)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_error_variance(
    results: pd.DataFrame,
    metric: str = 'rmse',
    model: str = 'lgbm',
    save: bool = True,
):
    """Box plots showing error distribution across seeds for each failure type."""
    df = results[results['model'] == model].copy()
    df['label'] = df.apply(
        lambda row: f"{FAILURE_LABELS.get(row['failure_type'], row['failure_type'])}\n"
                     f"(I={row['intensity']})",
        axis=1
    )

    fig, ax = plt.subplots(figsize=(16, 8))

    order = df.groupby('label')[metric].mean().sort_values(ascending=True).index.tolist()

    palette = []
    for label in order:
        ft = df[df['label'] == label]['failure_type'].iloc[0]
        palette.append(COLORS.get(ft, '#95a5a6'))

    sns.boxplot(data=df, x='label', y=metric, order=order, palette=palette, ax=ax)

    ax.set_xlabel('Failure Configuration', fontsize=12)
    ax.set_ylabel(metric.upper(), fontsize=12)
    ax.set_title(f'{model.upper()} — Error Variance Across Seeds\n'
                 f'(Each box = multiple seeds × Hawkes configs)',
                 fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)

    if save:
        path = PLOTS_DIR / f'error_variance_{metric}_{model}.png'
        fig.savefig(path)
        print(f"[Plot] Saved: {path}")

    plt.close(fig)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 5: Sensitivity Ranking
# ═══════════════════════════════════════════════════════════════════════════════

def plot_sensitivity_ranking(
    results: pd.DataFrame,
    metric: str = 'rmse',
    model: str = 'lgbm',
    save: bool = True,
):
    """Bar chart ranking failure types by worst-case degradation."""
    df = results[results['model'] == model].copy()

    baseline = df[df['failure_type'] == 'baseline'][metric].mean()
    chaos_df = df[df['failure_type'] != 'baseline']

    if chaos_df.empty or baseline == 0:
        return

    worst_case = chaos_df.groupby('failure_type')[metric].max()
    degradation = (worst_case / baseline).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(12, 7))

    colors = [COLORS.get(ft, '#95a5a6') for ft in degradation.index]
    labels = [FAILURE_LABELS.get(ft, ft) for ft in degradation.index]

    bars = ax.barh(range(len(degradation)), degradation.values, color=colors, edgecolor='white')
    ax.set_yticks(range(len(degradation)))
    ax.set_yticklabels(labels, fontsize=12)

    ax.axvline(x=1.0, color='#2ecc71', linestyle='--', linewidth=2,
               label='No degradation')

    for bar, val in zip(bars, degradation.values):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                f'{val:.2f}×', va='center', fontsize=11, fontweight='bold')

    ax.set_xlabel(f'Worst-Case {metric.upper()} / Baseline {metric.upper()}',
                  fontsize=13)
    ax.set_title(f'{model.upper()} — Sensitivity Ranking\n'
                 f'(Worst-case degradation per failure type)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.invert_yaxis()

    if save:
        path = PLOTS_DIR / f'sensitivity_ranking_{metric}_{model}.png'
        fig.savefig(path)
        print(f"[Plot] Saved: {path}")

    plt.close(fig)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 6: Model Comparison
# ═══════════════════════════════════════════════════════════════════════════════

def plot_model_comparison(
    results: pd.DataFrame,
    metric: str = 'rmse',
    save: bool = True,
):
    """Grouped bar chart comparing models across failure types."""
    models = results['model'].unique()
    if len(models) < 2:
        print("[Plot] Only one model found, skipping comparison plot.")
        return

    df = results[results['failure_type'] != 'baseline'].copy()

    fig, ax = plt.subplots(figsize=(14, 8))

    grouped = df.groupby(['failure_type', 'model'])[metric].mean().unstack()
    grouped.index = [FAILURE_LABELS.get(ft, ft) for ft in grouped.index]
    grouped.plot(kind='bar', ax=ax, width=0.7, edgecolor='white')

    ax.set_xlabel('Failure Type', fontsize=13)
    ax.set_ylabel(f'Mean {metric.upper()}', fontsize=13)
    ax.set_title(f'Model Comparison — {metric.upper()} by Failure Type',
                 fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.legend(title='Model')

    if save:
        path = PLOTS_DIR / f'model_comparison_{metric}.png'
        fig.savefig(path)
        print(f"[Plot] Saved: {path}")

    plt.close(fig)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 7: Robustness Curves (Multi-panel)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_robustness_curves(
    results: pd.DataFrame,
    model: str = 'lgbm',
    save: bool = True,
):
    """Multi-panel plot showing R(p) vs intensity for each failure type."""
    df = results[(results['model'] == model) &
                 (results['failure_type'] != 'baseline')].copy()

    failure_types = df['failure_type'].unique()
    n_types = len(failure_types)

    if n_types == 0:
        return

    n_cols = min(4, n_types)
    n_rows = int(np.ceil(n_types / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_types == 1:
        axes = np.array([axes])
    axes = axes.ravel()

    for i, ft in enumerate(failure_types):
        ax = axes[i]
        ft_data = df[df['failure_type'] == ft]

        grouped = ft_data.groupby('intensity')['robustness_rmse'].agg(
            ['mean', 'std']
        ).reset_index()

        color = COLORS.get(ft, '#95a5a6')

        ax.plot(grouped['intensity'], grouped['mean'],
                'o-', color=color, linewidth=2, markersize=8)
        ax.fill_between(
            grouped['intensity'],
            grouped['mean'] - grouped['std'],
            grouped['mean'] + grouped['std'],
            alpha=0.2, color=color
        )

        ax.axhline(y=1.0, color='#2ecc71', linestyle='--', alpha=0.7)
        ax.set_title(FAILURE_LABELS.get(ft, ft), fontsize=12, fontweight='bold')
        ax.set_xlabel('Intensity')
        ax.set_ylabel('R(p)')
        ax.set_ylim(0, max(1.3, grouped['mean'].max() * 1.1))
        ax.grid(True, alpha=0.3)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f'{model.upper()} — Robustness R(p) = Baseline/Chaos RMSE',
                 fontsize=16, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if save:
        path = PLOTS_DIR / f'robustness_curves_{model}.png'
        fig.savefig(path)
        print(f"[Plot] Saved: {path}")

    plt.close(fig)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 8: [NEW] Intensity vs Robustness Scatter
# ═══════════════════════════════════════════════════════════════════════════════

def plot_intensity_vs_robustness(
    results: pd.DataFrame,
    model: str = 'lgbm',
    save: bool = True,
):
    """
    Scatter plot: λ(t) mean/max vs R(p).

    Shows how cascade intensity correlates with model degradation.
    Each point is one experiment, coloured by failure type.
    """
    df = results[(results['model'] == model) &
                 (results['failure_type'] != 'baseline')].copy()

    if df.empty or 'lambda_mean' not in df.columns:
        print("[Plot] No Hawkes data for intensity vs robustness.")
        return

    # Filter to experiments with actual Hawkes activity
    df = df[df['lambda_mean'] > 0].copy()

    if df.empty:
        print("[Plot] No Hawkes activity found.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    for ax, lam_col, lam_label in zip(
        axes,
        ['lambda_mean', 'lambda_max'],
        ['Mean λ(t)', 'Max λ(t)']
    ):
        for ft in df['failure_type'].unique():
            ft_data = df[df['failure_type'] == ft]
            color = COLORS.get(ft, '#95a5a6')
            label = FAILURE_LABELS.get(ft, ft)

            ax.scatter(
                ft_data[lam_col], ft_data['robustness_rmse'],
                c=color, alpha=0.6, s=60, edgecolors='white',
                linewidth=0.5, label=label
            )

        ax.axhline(y=1.0, color='#2ecc71', linestyle='--',
                   alpha=0.7, label='R=1 (no degradation)')
        ax.set_xlabel(lam_label, fontsize=13)
        ax.set_ylabel('Robustness R(p)', fontsize=13)
        ax.set_title(f'{lam_label} vs Robustness', fontsize=14, fontweight='bold')
        ax.legend(fontsize=9, loc='lower left')
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'{model.upper()} — Hawkes Intensity vs Model Robustness',
                 fontsize=16, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if save:
        path = PLOTS_DIR / f'intensity_vs_robustness_{model}.png'
        fig.savefig(path)
        print(f"[Plot] Saved: {path}")

    plt.close(fig)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 9: [NEW] Hawkes Intensity Trace
# ═══════════════════════════════════════════════════════════════════════════════

def plot_hawkes_intensity_traces(
    results: pd.DataFrame,
    model: str = 'lgbm',
    n_traces: int = 6,
    save: bool = True,
):
    """
    Time-series plot of λ(t) for selected experiments, with event markers.

    Loads intensity traces from saved .npz files.
    """
    df = results[(results['model'] == model) &
                 (results['failure_type'] != 'baseline')].copy()

    if df.empty or 'lambda_trace_path' not in df.columns:
        return

    # Select experiments with the highest event counts
    df = df[df['lambda_trace_path'].notna() & (df['lambda_trace_path'] != '')].copy()

    if df.empty:
        return

    df = df.nlargest(n_traces, 'n_hawkes_events')

    n_plots = min(n_traces, len(df))
    if n_plots == 0:
        return

    n_cols = min(3, n_plots)
    n_rows = int(np.ceil(n_plots / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    if n_plots == 1:
        axes = np.array([axes])
    axes = np.atleast_1d(axes).ravel()

    for i, (_, row) in enumerate(df.head(n_plots).iterrows()):
        ax = axes[i]

        try:
            trace_path = row['lambda_trace_path']
            data = np.load(trace_path)
            trace = data['intensity_trace']
            events = data['event_times']
            params = data['params']

            t = np.arange(len(trace))
            color = COLORS.get(row['failure_type'], '#95a5a6')

            ax.plot(t, trace, color=color, linewidth=1.5, alpha=0.8)
            ax.axhline(y=params[0], color='gray', linestyle=':', alpha=0.5,
                       label=f'μ={params[0]:.2f}')

            # Mark events
            event_y = [trace[int(e)] if int(e) < len(trace) else params[0]
                       for e in events if int(e) < len(trace)]
            event_x = [int(e) for e in events if int(e) < len(trace)]
            ax.scatter(event_x, event_y, c='red', s=20, zorder=5,
                       alpha=0.7, marker='v', label='Events')

            ft_label = FAILURE_LABELS.get(row['failure_type'], row['failure_type'])
            ax.set_title(f'{ft_label}\n'
                         f'α={params[1]:.1f}, β={params[2]:.1f}, '
                         f'{len(events)} events',
                         fontsize=10, fontweight='bold')
            ax.set_xlabel('Timestep')
            ax.set_ylabel('λ(t)')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        except Exception as e:
            ax.text(0.5, 0.5, f'Load error:\n{e}',
                    transform=ax.transAxes, ha='center', va='center')

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f'{model.upper()} — Hawkes Intensity λ(t) Traces',
                 fontsize=16, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    if save:
        path = PLOTS_DIR / f'hawkes_intensity_traces_{model}.png'
        fig.savefig(path)
        print(f"[Plot] Saved: {path}")

    plt.close(fig)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 10: [NEW] Hawkes Parameter Heatmap — R over (α, β) grid
# ═══════════════════════════════════════════════════════════════════════════════

def plot_hawkes_parameter_heatmap(
    results: pd.DataFrame,
    model: str = 'lgbm',
    metric: str = 'rmse',
    save: bool = True,
):
    """
    Heatmap of robustness R over the (α, β) grid, averaged across
    failure types and intensities.

    Reveals the failure regime where models degrade fastest.
    """
    df = results[(results['model'] == model) &
                 (results['failure_type'] != 'baseline')].copy()

    if df.empty or 'alpha' not in df.columns:
        return

    # Filter to actual Hawkes experiments (α > 0)
    df = df[df['alpha'] > 0].copy()

    if df.empty:
        print("[Plot] No Hawkes experiments for parameter heatmap.")
        return

    rob_col = f'robustness_{metric}'

    # Pivot: α vs β → mean robustness
    pivot = df.pivot_table(
        index='alpha', columns='beta',
        values=rob_col, aggfunc='mean'
    )

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        pivot, annot=True, fmt='.3f', cmap='RdYlGn',
        linewidths=0.5, ax=ax,
        cbar_kws={'label': f'R({metric}) — higher = more robust'}
    )

    ax.set_xlabel('β (Decay Rate)', fontsize=13)
    ax.set_ylabel('α (Excitation Strength)', fontsize=13)
    ax.set_title(f'{model.upper()} — Robustness R(α, β)\n'
                 f'Averaged across failure types & intensities\n'
                 f'(Green = robust, Red = brittle)',
                 fontsize=14, fontweight='bold')

    # Mark the minimum R cell (most adversarial)
    min_r = pivot.min().min()
    min_pos = np.where(pivot.values == min_r)
    if len(min_pos[0]) > 0:
        ax.add_patch(plt.Rectangle(
            (min_pos[1][0], min_pos[0][0]), 1, 1,
            fill=False, edgecolor='black', linewidth=3
        ))
        ax.text(min_pos[1][0] + 0.5, min_pos[0][0] - 0.15,
                '★ WORST', ha='center', fontsize=10, fontweight='bold')

    if save:
        path = PLOTS_DIR / f'hawkes_parameter_heatmap_{metric}_{model}.png'
        fig.savefig(path)
        print(f"[Plot] Saved: {path}")

    plt.close(fig)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 11: [NEW] Adversarial Regime Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def plot_adversarial_regime(
    results: pd.DataFrame,
    model: str = 'lgbm',
    save: bool = True,
):
    """
    Multi-panel analysis of the adversarial Hawkes regime:
    1. Per-failure-type R vs α (at fixed β)
    2. Per-failure-type R vs β (at fixed α)
    3. Branching ratio α/β vs R
    """
    df = results[(results['model'] == model) &
                 (results['failure_type'] != 'baseline') &
                 (results['alpha'] > 0)].copy()

    if df.empty:
        return

    df['branching_ratio'] = df['alpha'] / df['beta']

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    # Panel 1: R vs α (averaged across β)
    ax = axes[0]
    for ft in df['failure_type'].unique():
        ft_data = df[df['failure_type'] == ft]
        grouped = ft_data.groupby('alpha')['robustness_rmse'].mean().reset_index()
        color = COLORS.get(ft, '#95a5a6')
        ax.plot(grouped['alpha'], grouped['robustness_rmse'],
                'o-', color=color, linewidth=2, markersize=8,
                label=FAILURE_LABELS.get(ft, ft))
    ax.axhline(y=1.0, color='#2ecc71', linestyle='--', alpha=0.7)
    ax.set_xlabel('α (Excitation)', fontsize=13)
    ax.set_ylabel('R (Robustness)', fontsize=13)
    ax.set_title('R vs Excitation α', fontsize=14, fontweight='bold')
    ax.legend(fontsize=8, loc='lower left')
    ax.grid(True, alpha=0.3)

    # Panel 2: R vs β (averaged across α)
    ax = axes[1]
    for ft in df['failure_type'].unique():
        ft_data = df[df['failure_type'] == ft]
        grouped = ft_data.groupby('beta')['robustness_rmse'].mean().reset_index()
        color = COLORS.get(ft, '#95a5a6')
        ax.plot(grouped['beta'], grouped['robustness_rmse'],
                's-', color=color, linewidth=2, markersize=8,
                label=FAILURE_LABELS.get(ft, ft))
    ax.axhline(y=1.0, color='#2ecc71', linestyle='--', alpha=0.7)
    ax.set_xlabel('β (Decay Rate)', fontsize=13)
    ax.set_ylabel('R (Robustness)', fontsize=13)
    ax.set_title('R vs Decay β', fontsize=14, fontweight='bold')
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3)

    # Panel 3: R vs Branching Ratio α/β
    ax = axes[2]
    for ft in df['failure_type'].unique():
        ft_data = df[df['failure_type'] == ft]
        grouped = ft_data.groupby('branching_ratio')['robustness_rmse'].mean().reset_index()
        color = COLORS.get(ft, '#95a5a6')
        ax.plot(grouped['branching_ratio'], grouped['robustness_rmse'],
                'D-', color=color, linewidth=2, markersize=8,
                label=FAILURE_LABELS.get(ft, ft))
    ax.axhline(y=1.0, color='#2ecc71', linestyle='--', alpha=0.7)
    ax.axvline(x=1.0, color='red', linestyle=':', alpha=0.5,
               label='α/β = 1 (critical)')
    ax.set_xlabel('α/β (Branching Ratio)', fontsize=13)
    ax.set_ylabel('R (Robustness)', fontsize=13)
    ax.set_title('R vs Branching Ratio\n(Critical threshold α/β = 1)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=8, loc='lower left')
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'{model.upper()} — Adversarial Hawkes Regime Analysis',
                 fontsize=16, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    if save:
        path = PLOTS_DIR / f'adversarial_regime_{model}.png'
        fig.savefig(path)
        print(f"[Plot] Saved: {path}")

    plt.close(fig)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Master Analysis Function
# ═══════════════════════════════════════════════════════════════════════════════

def run_analysis(results: pd.DataFrame = None, csv_path: str = None):
    """
    Run all analysis and generate all plots (original + Hawkes).
    """
    if results is None:
        results = load_results(csv_path)

    print("\n" + "=" * 70)
    print("PHASE 4: ANALYSIS AND VISUALIZATION (HAWKES-EXTENDED)")
    print("=" * 70)

    models = results['model'].unique()
    has_hawkes = 'alpha' in results.columns and (results.get('alpha', 0) > 0).any()

    for model in models:
        print(f"\n--- Generating plots for {model.upper()} ---")

        # Original plots
        for metric in ['rmse', 'mae']:
            plot_performance_vs_intensity(results, metric=metric, model=model)
        for metric in ['rmse', 'mae']:
            plot_heatmap(results, metric=metric, model=model)
        plot_robustness_radar(results, model=model)
        plot_error_variance(results, model=model)
        plot_sensitivity_ranking(results, model=model)
        plot_robustness_curves(results, model=model)

        # Hawkes-specific plots
        if has_hawkes:
            print(f"\n--- Generating Hawkes-specific plots for {model.upper()} ---")
            plot_intensity_vs_robustness(results, model=model)
            plot_hawkes_intensity_traces(results, model=model)
            for metric in ['rmse', 'mae']:
                plot_hawkes_parameter_heatmap(results, metric=metric, model=model)
            plot_adversarial_regime(results, model=model)

    if len(models) > 1:
        plot_model_comparison(results)

    # ────────────────────────────────────────────────────────────────────────
    # Print Summary Table
    # ────────────────────────────────────────────────────────────────────────

    print("\n" + "=" * 70)
    print("FINAL RESULTS TABLE")
    print("=" * 70)

    group_cols = ['model', 'failure_type', 'intensity']
    agg_dict = {
        'rmse': ['mean', 'std'],
        'mae': ['mean', 'std'],
        'wrmsse': ['mean', 'std'],
        'robustness_rmse': 'mean',
    }

    if has_hawkes:
        group_cols.extend(['alpha', 'beta'])
        agg_dict['lambda_mean'] = 'mean'
        agg_dict['n_hawkes_events'] = 'mean'

    summary = results.groupby(group_cols).agg(agg_dict).reset_index()
    summary.columns = ['_'.join(c).strip('_') for c in summary.columns]

    pd.set_option('display.max_columns', 25)
    pd.set_option('display.width', 250)
    pd.set_option('display.float_format', '{:.4f}'.format)
    print(summary.to_string(index=False))

    # ────────────────────────────────────────────────────────────────────────
    # Key Insights
    # ────────────────────────────────────────────────────────────────────────

    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)

    for model in models:
        model_df = results[(results['model'] == model) &
                           (results['failure_type'] != 'baseline')]

        if model_df.empty:
            continue

        baseline_rmse = results[
            (results['model'] == model) &
            (results['failure_type'] == 'baseline')
        ]['rmse'].mean()

        worst = model_df.groupby('failure_type')['rmse'].max()
        most_damaging = worst.idxmax()
        least_damaging = worst.idxmin()

        r_min = model_df['robustness_rmse'].min()
        r_max = model_df['robustness_rmse'].max()

        print(f"\n{model.upper()} Model:")
        print(f"  Baseline RMSE: {baseline_rmse:.4f}")
        print(f"  Most damaging failure: {FAILURE_LABELS.get(most_damaging, most_damaging)} "
              f"(worst RMSE: {worst[most_damaging]:.4f}, "
              f"{worst[most_damaging]/baseline_rmse:.1f}× baseline)")
        print(f"  Least damaging failure: {FAILURE_LABELS.get(least_damaging, least_damaging)} "
              f"(worst RMSE: {worst[least_damaging]:.4f}, "
              f"{worst[least_damaging]/baseline_rmse:.1f}× baseline)")
        print(f"  Robustness range: R ∈ [{r_min:.3f}, {r_max:.3f}]")

        # Sensitivity analysis
        print(f"\n  Sensitivity Analysis:")
        for ft in model_df['failure_type'].unique():
            ft_data = model_df[model_df['failure_type'] == ft]
            intensities = sorted(ft_data['intensity'].unique())
            if len(intensities) >= 2:
                low = ft_data[ft_data['intensity'] == intensities[0]]['rmse'].mean()
                high = ft_data[ft_data['intensity'] == intensities[-1]]['rmse'].mean()
                gradient = (high - low) / (intensities[-1] - intensities[0]) if intensities[-1] != intensities[0] else 0
                print(f"    {FAILURE_LABELS.get(ft, ft):20s}: "
                      f"RMSE {low:.4f} → {high:.4f} "
                      f"(gradient: {gradient:.4f} per unit intensity)")

        # ─── Hawkes-Specific Insights ─────────────────────────────────
        if has_hawkes:
            hawkes_df = model_df[model_df['alpha'] > 0]

            if not hawkes_df.empty:
                print(f"\n  Hawkes Process Insights:")

                # Most adversarial configuration
                worst_hawkes = hawkes_df.loc[
                    hawkes_df['robustness_rmse'].idxmin()
                ]
                print(f"    Most adversarial config: "
                      f"α={worst_hawkes['alpha']:.1f}, "
                      f"β={worst_hawkes['beta']:.1f}, "
                      f"μ={worst_hawkes['mu']:.2f} "
                      f"→ R={worst_hawkes['robustness_rmse']:.3f}")

                # Correlation between intensity and robustness
                if 'lambda_mean' in hawkes_df.columns:
                    lam_vals = hawkes_df['lambda_mean'].dropna()
                    rob_vals = hawkes_df.loc[lam_vals.index, 'robustness_rmse']
                    if len(lam_vals) > 3:
                        corr = np.corrcoef(lam_vals.values, rob_vals.values)[0, 1]
                        print(f"    Correlation(λ_mean, R): {corr:.3f}")

                # Branching ratio analysis
                hawkes_df_copy = hawkes_df.copy()
                hawkes_df_copy['br'] = hawkes_df_copy['alpha'] / hawkes_df_copy['beta']
                subcrit = hawkes_df_copy[hawkes_df_copy['br'] < 1]['robustness_rmse'].mean()
                supercrit = hawkes_df_copy[hawkes_df_copy['br'] >= 1]['robustness_rmse'].mean()
                print(f"    Sub-critical (α/β < 1) mean R: {subcrit:.3f}")
                print(f"    Super-critical (α/β ≥ 1) mean R: {supercrit:.3f}")

                # Predicted high-risk windows
                if 'lambda_max' in hawkes_df.columns:
                    max_lam = hawkes_df['lambda_max'].max()
                    expected_inter_event = 1.0 / max_lam if max_lam > 0 else float('inf')
                    print(f"    Peak λ_max: {max_lam:.3f} "
                          f"(E[inter-event] = {expected_inter_event:.1f} steps)")

    print(f"\nAll plots saved to: {PLOTS_DIR}")

    return summary


if __name__ == '__main__':
    run_analysis()
