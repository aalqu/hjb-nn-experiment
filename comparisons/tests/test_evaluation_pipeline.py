import numpy as np
from pathlib import Path

from comparisons.core.evaluation import (
    REQUIRED_RESULT_KEYS,
    evaluate_static_portfolio,
    run_real_data_portfolio_comparison,
    validate_result_schema,
)
from comparisons.core.io import load_run_result, save_run_result
from comparisons.core.config import BenchmarkConfig
from real_data_loader import load_portfolio


class TestStaticPortfolioEvaluation:
    def test_schema_and_shapes(self):
        mkt = load_portfolio(5, start='2020-01-01', end='2020-12-31')
        weights = np.ones(mkt.n) / mkt.n
        result = evaluate_static_portfolio(
            method_name='equal_weight',
            method_family='baseline',
            market_data=mkt,
            weights=weights,
            initial_wealth=1.0,
            target_multiplier=1.1,
        )
        validate_result_schema(result)
        for key in REQUIRED_RESULT_KEYS:
            assert key in result
        assert result['n_assets'] == 5
        assert result['wealth_path'].shape[0] == mkt.log_ret.shape[0] + 1
        assert result['weight_path'].shape == (mkt.log_ret.shape[0], mkt.n)

    def test_save_roundtrip(self, tmp_path: Path):
        mkt = load_portfolio(5, start='2020-01-01', end='2020-06-30')
        result = evaluate_static_portfolio(
            method_name='equal_weight',
            method_family='baseline',
            market_data=mkt,
            weights=np.ones(mkt.n) / mkt.n,
        )
        path = tmp_path / 'run_result.npz'
        save_run_result(path, result)
        loaded = load_run_result(path)
        assert loaded['method_name'] == result['method_name']
        np.testing.assert_allclose(loaded['wealth_path'], result['wealth_path'])
        np.testing.assert_allclose(loaded['weight_path'], result['weight_path'])


class TestRealDataComparison:
    def test_smoke_run(self, tmp_path: Path):
        config = BenchmarkConfig(
            n_assets_list=[5],
            start_date='2020-01-01',
            end_date='2020-12-31',
            initial_wealth_levels=[1.0],
            target_multiplier=1.05,
            results_dir=tmp_path,
            include_fd_benchmark=False,
            include_nn=False,
            random_seeds=[1],
        )
        outputs = run_real_data_portfolio_comparison(config)
        assert outputs['summary'].shape[0] >= 3
        assert set(outputs['summary']['method_name']) >= {
            'equal_weight',
            'max_sharpe',
            'market_cap',
        }
        summary_path = tmp_path / 'summary' / 'main_results.csv'
        assert summary_path.exists()
        assert (tmp_path / 'summary' / 'aggregated_results.csv').exists()
        assert (tmp_path / 'summary' / 'neural_family_results.csv').exists()
        assert (tmp_path / 'plots' / 'fd_vs_neural_target_hit_rate.png').exists()
