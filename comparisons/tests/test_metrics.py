import numpy as np

from comparisons.core.metrics import (
    compute_target_metrics,
    compute_weight_metrics,
    compute_drawdown_series,
)


class TestTargetMetrics:
    def test_goalreach_summary(self):
        terminal = np.array([0.9, 1.0, 1.2, 1.4])
        metrics = compute_target_metrics(terminal, target=1.1)
        assert metrics['target_hit_rate'] == 0.5
        assert abs(metrics['mean_terminal_wealth'] - terminal.mean()) < 1e-12
        assert metrics['expected_shortfall'] > 0
        assert metrics['terminal_wealth_p05'] <= metrics['median_terminal_wealth']


class TestWeightMetrics:
    def test_weight_path_summary(self):
        weights = np.array([
            [0.5, 0.5],
            [0.75, 0.25],
            [1.0, 0.0],
        ])
        metrics = compute_weight_metrics(weights)
        assert abs(metrics['mean_gross_leverage'] - 1.0) < 1e-12
        assert abs(metrics['mean_net_exposure'] - 1.0) < 1e-12
        assert metrics['max_single_name_weight'] == 1.0
        assert metrics['turnover'] > 0
        assert metrics['mean_concentration'] > 0.5


class TestDrawdownSeries:
    def test_drawdown_starts_at_zero(self):
        wealth = np.array([1.0, 1.1, 0.9, 1.2])
        dd = compute_drawdown_series(wealth)
        assert dd[0] == 0.0
        assert dd.min() < 0
        assert dd[-1] == 0.0
