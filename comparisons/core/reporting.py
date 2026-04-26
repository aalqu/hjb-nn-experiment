import os
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

from .io import ensure_dir, save_summary_csv

os.environ.setdefault('MPLCONFIGDIR', str(Path('/tmp') / 'matplotlib-codex'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


_NUMERIC_FIELDS = {
    'train_time_sec', 'solve_time_sec', 'eval_time_sec', 'target_hit_rate',
    'mean_terminal_wealth', 'median_terminal_wealth', 'terminal_wealth_p05',
    'expected_shortfall', 'mean_gross_leverage', 'max_gross_leverage',
    'mean_net_exposure', 'mean_concentration', 'max_single_name_weight',
    'turnover', 'wealth_volatility', 'max_drawdown', 'initial_wealth', 'target_wealth'
}


def aggregate_summary_rows(rows: List[Dict], group_fields=('method_name', 'method_family', 'n_assets')):
    groups = {}
    for row in rows:
        key = tuple(row[field] for field in group_fields)
        groups.setdefault(key, []).append(row)

    aggregated = []
    for key, items in groups.items():
        base = {field: value for field, value in zip(group_fields, key)}
        base['n_runs'] = len(items)
        for field in _NUMERIC_FIELDS:
            vals = [float(item[field]) for item in items if field in item]
            if vals:
                base[field] = float(np.mean(vals))
        aggregated.append(base)

    aggregated.sort(key=lambda row: (row['n_assets'], row['method_family'], row['method_name']))
    return aggregated


def filter_rows(rows: Iterable[Dict], **criteria):
    out = []
    for row in rows:
        if all(row.get(k) == v for k, v in criteria.items()):
            out.append(row)
    return out


_FD_METHODS = {'fd_1d_proxy', 'fd_merton_multi', 'fd_goalreach_proxy'}
_NN_METHODS = {'deep_bsde', 'pinn', 'actor_critic', 'lstm', 'transformer',
               'nn_mlp_small', 'nn_mlp_deep'}


def save_comparison_tables(summary_rows: List[Dict], summary_dir: Path):
    summary_dir = ensure_dir(summary_dir)
    aggregated = aggregate_summary_rows(summary_rows)
    neural_rows = [row for row in aggregated if row['method_family'] == 'nn']
    fd_vs_nn_rows = [row for row in aggregated if row['method_name'] in
                     _FD_METHODS | _NN_METHODS]

    save_summary_csv(summary_dir / 'aggregated_results.csv', aggregated)
    save_summary_csv(summary_dir / 'neural_family_results.csv', neural_rows)
    save_summary_csv(summary_dir / 'fd_vs_neural_results.csv', fd_vs_nn_rows)
    return {
        'aggregated': aggregated,
        'neural_family': neural_rows,
        'fd_vs_neural': fd_vs_nn_rows,
    }


_COLOR_MAP = {
    'fd_1d_proxy': '#1f2937',
    'fd_merton_multi': '#374151',
    'deep_bsde': '#b91c1c',
    'pinn': '#0f766e',
    'actor_critic': '#1d4ed8',
    'lstm': '#a16207',
    'transformer': '#7c3aed',
    'nn_mlp_small': '#6d28d9',
    'nn_mlp_deep': '#0369a1',
    # keep old name for backward-compat with any cached rows
    'fd_goalreach_proxy': '#1f2937',
}


def _line_plot(ax, rows, methods, metric, ylabel, title):
    for method in methods:
        pts = [row for row in rows if row['method_name'] == method]
        if not pts:
            continue
        pts = sorted(pts, key=lambda row: row['n_assets'])
        x = [row['n_assets'] for row in pts]
        y = [row[metric] for row in pts]
        ax.plot(x, y, marker='o', linewidth=2, label=method, color=_COLOR_MAP.get(method))
    ax.set_xlabel('Number of assets')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=8)


def save_plots(summary_rows: List[Dict], plots_dir: Path):
    plots_dir = ensure_dir(plots_dir)
    aggregated = aggregate_summary_rows(summary_rows)
    target_methods = [
        'fd_1d_proxy', 'fd_merton_multi',
        'deep_bsde', 'pinn', 'actor_critic', 'lstm', 'transformer',
        'nn_mlp_small', 'nn_mlp_deep',
        # backward-compat
        'fd_goalreach_proxy',
    ]
    compare_rows = [row for row in aggregated if row['method_name'] in target_methods]

    fig, ax = plt.subplots(figsize=(8, 5))
    _line_plot(ax, compare_rows, target_methods, 'target_hit_rate', 'Target hit rate', 'FD vs Neural: Target Hit Rate')
    fig.tight_layout()
    fig.savefig(plots_dir / 'fd_vs_neural_target_hit_rate.png', dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    _line_plot(ax, compare_rows, target_methods, 'mean_terminal_wealth', 'Mean terminal wealth', 'FD vs Neural: Mean Terminal Wealth')
    fig.tight_layout()
    fig.savefig(plots_dir / 'fd_vs_neural_mean_terminal_wealth.png', dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    _line_plot(ax, compare_rows, target_methods, 'train_time_sec', 'Train / solve time (s)', 'FD vs Neural: Computational Cost')
    fig.tight_layout()
    fig.savefig(plots_dir / 'fd_vs_neural_runtime.png', dpi=160)
    plt.close(fig)

    all_method_rows = [row for row in aggregated if row['method_family'] in ('nn', 'fd')]
    fig, ax = plt.subplots(figsize=(8, 5))
    for method in ['fd_1d_proxy', 'fd_merton_multi', 'deep_bsde', 'pinn',
                   'actor_critic', 'lstm', 'transformer', 'nn_mlp_small', 'nn_mlp_deep']:
        pts = [row for row in all_method_rows if row['method_name'] == method]
        if not pts:
            continue
        ax.scatter(
            [row['mean_gross_leverage'] for row in pts],
            [row['target_hit_rate'] for row in pts],
            s=[40 + 12 * row['n_assets'] for row in pts],
            alpha=0.8,
            label=method,
            color=_COLOR_MAP.get(method),
        )
    ax.set_xlabel('Mean gross leverage')
    ax.set_ylabel('Target hit rate')
    ax.set_title('FD & Neural: Risk vs Performance')
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / 'neural_risk_vs_performance.png', dpi=160)
    plt.close(fig)

    return [
        plots_dir / 'fd_vs_neural_target_hit_rate.png',
        plots_dir / 'fd_vs_neural_mean_terminal_wealth.png',
        plots_dir / 'fd_vs_neural_runtime.png',
        plots_dir / 'neural_risk_vs_performance.png',
    ]
