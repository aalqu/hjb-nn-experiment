import numpy as np
from pathlib import Path

from comparisons.core.config import BenchmarkConfig
from comparisons.core.evaluation import run_real_data_portfolio_comparison
from comparisons.core.nn_models import NumpyPolicyNet, train_numpy_policy_net, policy_weights
from real_data_loader import load_portfolio


class TestNumpyPolicyNet:
    def test_output_shape(self):
        net = NumpyPolicyNet(n_assets=3, hidden_layers=(8,))
        out = net.forward(np.array([[0.9], [1.1]]))
        assert out.shape == (2, 3)

    def test_output_bounds(self):
        net = NumpyPolicyNet(n_assets=4, hidden_layers=(6, 6), d=-2.0, u=1.5)
        out = net.forward(np.linspace(0.5, 1.5, 5)[:, None])
        assert np.all(out >= -2.0 - 1e-9)
        assert np.all(out <= 1.5 + 1e-9)

    def test_training_returns_finite_model(self):
        mkt = load_portfolio(5, start='2020-01-01', end='2020-06-30')
        net, meta = train_numpy_policy_net(
            mu_vec=mkt.mu_ann,
            omega_mat=mkt.omega,
            r=mkt.r,
            architecture_name='nn_mlp_small',
            w0=1.0,
            goal_mult=1.05,
            n_paths=48,
            n_steps=8,
            n_iters=3,
            population_size=6,
            elite_frac=0.5,
            seed=7,
        )
        pi = policy_weights(net, 1.0, 1.05)
        assert pi.shape == (mkt.n,)
        assert not np.any(np.isnan(pi))
        assert meta['architecture_name'] == 'nn_mlp_small'


class TestComparisonIncludesNeuralResults:
    def test_smoke_run_with_numpy_nn(self, tmp_path: Path):
        config = BenchmarkConfig(
            n_assets_list=[5],
            start_date='2020-01-01',
            end_date='2020-06-30',
            initial_wealth_levels=[1.0],
            target_multiplier=1.05,
            results_dir=tmp_path,
            include_fd_benchmark=False,
            include_nn=True,
            nn_architectures=['nn_mlp_small', 'nn_mlp_deep'],
            nn_iters=2,
            nn_population_size=5,
            nn_paths=32,
            nn_steps=6,
            random_seeds=[1],
        )
        outputs = run_real_data_portfolio_comparison(config)
        names = set(outputs['summary']['method_name'])
        assert 'nn_mlp_small' in names
        assert 'nn_mlp_deep' in names
        assert (tmp_path / 'raw' / 'nn_mlp_small_n5_seed1_w1.00.npz').exists()
