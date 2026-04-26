import numpy as np
import pytest
from pathlib import Path

try:
    import torch  # noqa: F401
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

from comparisons.core.config import BenchmarkConfig
from comparisons.core.evaluation import run_real_data_portfolio_comparison

if HAS_TORCH:
    from comparisons.core.torch_nn_models import (
        TORCH_ARCHITECTURES,
        TorchPolicyNet,
        policy_weights,
        train_torch_policy_net,
    )
    from real_data_loader import load_portfolio


pytestmark = pytest.mark.skipif(not HAS_TORCH, reason='PyTorch not available')


class TestTorchPolicyNet:
    def test_output_shape(self):
        net = TorchPolicyNet(n_assets=3, hidden_layers=(8, 4), d=-2.0, u=1.5)
        out = net(torch.ones(5, 1))
        assert tuple(out.shape) == (5, 3)

    def test_output_bounds(self):
        net = TorchPolicyNet(n_assets=4, hidden_layers=(6,), d=-3.0, u=2.0)
        with torch.no_grad():
            out = net(torch.linspace(0.5, 1.5, 7).unsqueeze(1))
        assert out.min().item() >= -3.0 - 1e-6
        assert out.max().item() <= 2.0 + 1e-6

    def test_training_smoke(self):
        mkt = load_portfolio(5, start='2020-01-01', end='2020-06-30')
        net, meta = train_torch_policy_net(
            mu_vec=mkt.mu_ann,
            omega_mat=mkt.omega,
            r=mkt.r,
            architecture_name='nn_mlp_small',
            w0=1.0,
            goal_mult=1.05,
            n_paths=32,
            n_iters=2,
            n_steps=6,
            seed=11,
        )
        pi = policy_weights(net, 1.0, 1.05)
        assert pi.shape == (mkt.n,)
        assert not np.any(np.isnan(pi))
        assert meta['architecture_name'] == 'nn_mlp_small'
        assert meta['backend'] == 'torch'

    @pytest.mark.parametrize(
        'architecture_name',
        ['deep_bsde', 'pinn', 'actor_critic', 'lstm', 'transformer'],
    )
    def test_requested_methods_train(self, architecture_name):
        mkt = load_portfolio(5, start='2020-01-01', end='2020-03-31')
        net, meta = train_torch_policy_net(
            mu_vec=mkt.mu_ann,
            omega_mat=mkt.omega,
            r=mkt.r,
            architecture_name=architecture_name,
            w0=1.0,
            goal_mult=1.05,
            n_paths=16,
            n_iters=1,
            n_steps=4,
            seed=3,
        )
        pi = policy_weights(net, 1.0, 1.05, history=[1.0, 1.01, 1.02], step_idx=3, total_steps=4)
        assert pi.shape == (mkt.n,)
        assert not np.any(np.isnan(pi))
        assert meta['architecture_name'] == architecture_name
        assert architecture_name in TORCH_ARCHITECTURES


class TestEvaluationPrefersTorch:
    def test_smoke_run_uses_torch_backend(self, tmp_path: Path):
        config = BenchmarkConfig(
            n_assets_list=[5],
            start_date='2020-01-01',
            end_date='2020-06-30',
            initial_wealth_levels=[1.0],
            target_multiplier=1.05,
            results_dir=tmp_path,
            include_fd_benchmark=False,
            include_nn=True,
            nn_architectures=['nn_mlp_small'],
            nn_iters=2,
            nn_paths=32,
            nn_steps=6,
            random_seeds=[1],
        )
        outputs = run_real_data_portfolio_comparison(config)
        rows = outputs['summary']
        row = rows[rows['method_name'] == 'nn_mlp_small'][0]
        assert row['train_time_sec'] > 0
        raw_path = tmp_path / 'raw' / 'nn_mlp_small_n5_seed1_w1.00.npz'
        assert raw_path.exists()

    def test_requested_methods_appear_in_summary(self, tmp_path: Path):
        config = BenchmarkConfig(
            n_assets_list=[5],
            start_date='2020-01-01',
            end_date='2020-03-31',
            initial_wealth_levels=[1.0],
            target_multiplier=1.05,
            results_dir=tmp_path,
            include_fd_benchmark=False,
            include_nn=True,
            nn_architectures=['deep_bsde', 'pinn', 'actor_critic', 'lstm', 'transformer'],
            nn_iters=1,
            nn_paths=16,
            nn_steps=4,
            random_seeds=[1],
        )
        outputs = run_real_data_portfolio_comparison(config)
        names = set(outputs['summary']['method_name'])
        assert {'deep_bsde', 'pinn', 'actor_critic', 'lstm', 'transformer'} <= names
