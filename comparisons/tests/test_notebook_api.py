import numpy as np
from pathlib import Path

from comparisons.core.config import BenchmarkConfig
from comparisons.core.evaluation import run_real_data_portfolio_comparison
from comparisons.core.notebook_api import (
    list_available_runs,
    load_fd_policy_bundle,
    load_nn_model_bundle,
    load_summary_table,
)


class TestNotebookApi:
    def test_load_summary_and_list_runs(self, tmp_path: Path):
        config = BenchmarkConfig(
            n_assets_list=[5],
            start_date='2020-01-01',
            end_date='2020-03-31',
            initial_wealth_levels=[1.0],
            target_multiplier=1.05,
            results_dir=tmp_path,
            include_fd_benchmark=True,
            include_nn=True,
            nn_architectures=['nn_mlp_small'],
            nn_iters=1,
            nn_paths=16,
            nn_steps=4,
            random_seeds=[1],
        )
        run_real_data_portfolio_comparison(config)
        summary = load_summary_table('main_results', results_dir=tmp_path)
        runs = list_available_runs(results_dir=tmp_path)
        assert len(summary) > 0
        assert any(run['method_name'] == 'fd_1d_proxy' for run in runs)
        assert any(run['method_name'] == 'nn_mlp_small' for run in runs)

    def test_load_fd_policy_bundle(self, tmp_path: Path):
        config = BenchmarkConfig(
            n_assets_list=[5],
            start_date='2020-01-01',
            end_date='2020-03-31',
            initial_wealth_levels=[1.0],
            target_multiplier=1.05,
            results_dir=tmp_path,
            include_fd_benchmark=True,
            include_nn=False,
            random_seeds=[1],
        )
        run_real_data_portfolio_comparison(config)
        bundle = load_fd_policy_bundle(n_assets=5, seed=1, initial_wealth=1.0, results_dir=tmp_path)
        assert bundle['result']['method_name'] == 'fd_1d_proxy'
        assert callable(bundle['policy'])
        pi = bundle['policy'](1.0, 0.5)
        assert np.isfinite(pi)

    def test_load_nn_model_bundle(self, tmp_path: Path):
        config = BenchmarkConfig(
            n_assets_list=[5],
            start_date='2020-01-01',
            end_date='2020-03-31',
            initial_wealth_levels=[1.0],
            target_multiplier=1.05,
            results_dir=tmp_path,
            include_fd_benchmark=False,
            include_nn=True,
            nn_architectures=['nn_mlp_small'],
            nn_iters=1,
            nn_paths=16,
            nn_steps=4,
            random_seeds=[1],
        )
        run_real_data_portfolio_comparison(config)
        bundle = load_nn_model_bundle('nn_mlp_small', n_assets=5, seed=1, initial_wealth=1.0, results_dir=tmp_path)
        assert bundle['result']['method_name'] == 'nn_mlp_small'
        weights = bundle['weights_fn'](1.0, 1.05)
        assert weights.shape == (5,)
        assert np.all(np.isfinite(weights))
