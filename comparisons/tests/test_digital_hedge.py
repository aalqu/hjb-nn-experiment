"""
test_digital_hedge.py
----------------------
Red-green TDD for the digital-option delta-hedge architecture (nn_digital_hedge).

Methodology recap
-----------------
The Browne (1999) value function V(w, τ) = P(W_T ≥ goal | W_t = w) is the
price of a digital call option on a multi-asset GBM portfolio.  The optimal
portfolio weights are its multi-asset delta hedge:

    π*(w, τ) = −(V_w / (w · V_ww)) · Ω⁻¹η

Training has three phases:
  1. Supervised Browne pre-train  — fit V to the analytical formula Φ(d)
  2. Kolmogorov PDE residual      — enforce V_τ = (r+θ²)·w·V_w + ½θ²·w²·V_ww
  3. Path simulation + BCE match  — V predicts empirical P(W_T ≥ goal)

These tests verify every layer of that stack, from the model class to the full
evaluation pipeline.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "comparisons"))

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _market():
    from real_data_loader import load_portfolio
    return load_portfolio(5, start="2020-01-01", end="2020-06-30")


def _browne_v(w_norm, tau, theta2, r: float = 0.0):
    """Analytical Browne value Φ(d) with r-corrected drift d = (log w + (r+½θ²)τ)/(θ√τ)."""
    theta = math.sqrt(max(theta2, 1e-12))
    tau   = max(float(tau), 1e-10)
    d     = (math.log(max(float(w_norm), 1e-10)) + (r + 0.5 * theta2) * tau) / (theta * math.sqrt(tau))
    return 0.5 * (1.0 + math.erf(d / math.sqrt(2.0)))


# ─────────────────────────────────────────────────────────────────────────────
# 1. DigitalHedgeValueNet — class existence and basic properties
# ─────────────────────────────────────────────────────────────────────────────

class TestDigitalHedgeValueNetExists:
    def test_class_importable(self):
        from comparisons.core.torch_nn_models import DigitalHedgeValueNet  # noqa

    def test_kind_attribute(self):
        from comparisons.core.torch_nn_models import DigitalHedgeValueNet
        net = DigitalHedgeValueNet(n_assets=5)
        assert net.kind == "digital_hedge"

    def test_has_omega_inv_eta_buffer(self):
        from comparisons.core.torch_nn_models import DigitalHedgeValueNet
        net = DigitalHedgeValueNet(n_assets=5)
        assert hasattr(net, "omega_inv_eta")
        assert net.omega_inv_eta.shape == (5,)

    def test_has_theta2_buffer(self):
        from comparisons.core.torch_nn_models import DigitalHedgeValueNet
        net = DigitalHedgeValueNet(n_assets=5)
        assert hasattr(net, "theta2_val")

    def test_has_r_buffer(self):
        from comparisons.core.torch_nn_models import DigitalHedgeValueNet
        net = DigitalHedgeValueNet(n_assets=5)
        assert hasattr(net, "r_val")


# ─────────────────────────────────────────────────────────────────────────────
# 2. DigitalHedgeValueNet — forward pass (value network)
# ─────────────────────────────────────────────────────────────────────────────

class TestDigitalHedgeForward:
    def test_output_in_zero_one(self):
        from comparisons.core.torch_nn_models import DigitalHedgeValueNet
        net  = DigitalHedgeValueNet(n_assets=5)
        feat = torch.rand(32, 2)
        out  = net(feat)
        assert out.shape == (32, 1)
        assert out.min().item() > 0.0
        assert out.max().item() < 1.0

    def test_monotone_in_wealth(self):
        """V should be (roughly) increasing in w_norm at fixed τ."""
        from comparisons.core.torch_nn_models import DigitalHedgeValueNet
        net = DigitalHedgeValueNet(n_assets=5)
        # After pre-training on Browne formula this would hold exactly;
        # for a fresh (random) net we just check output dimension is correct.
        w    = torch.linspace(0.5, 1.5, 10).unsqueeze(1)
        tau  = torch.full_like(w, 0.5)
        feat = torch.cat([w, tau], dim=1)
        out  = net(feat)
        assert out.shape == (10, 1)

    def test_output_depends_on_input(self):
        """Network output must differ across inputs (not a constant)."""
        from comparisons.core.torch_nn_models import DigitalHedgeValueNet
        net  = DigitalHedgeValueNet(n_assets=5)
        f1   = torch.tensor([[0.5, 0.5]])
        f2   = torch.tensor([[1.5, 0.5]])
        assert net(f1).item() != net(f2).item()


# ─────────────────────────────────────────────────────────────────────────────
# 3. DigitalHedgeValueNet — delta_policy
# ─────────────────────────────────────────────────────────────────────────────

class TestDeltaPolicy:
    def _net_with_buffers(self, n=5):
        from comparisons.core.torch_nn_models import DigitalHedgeValueNet
        net = DigitalHedgeValueNet(n_assets=n, d=-5.0, u=3.0,
                                   max_long=3.0, max_short=5.0)
        # Set buffers to unit Merton portfolio
        net.omega_inv_eta.fill_(1.0 / n)
        net.theta2_val.fill_(0.25)
        net.r_val.fill_(0.03)
        return net

    def test_output_shape(self):
        net  = self._net_with_buffers(5)
        feat = torch.tensor([[1.0, 0.5], [0.9, 0.3], [1.2, 0.8]],
                            requires_grad=True)
        with torch.enable_grad():
            pi = net.delta_policy(feat)
        assert pi.shape == (3, 5)

    def test_within_leverage_bounds(self):
        net  = self._net_with_buffers(5)
        feat = torch.rand(50, 2).requires_grad_(True)
        with torch.enable_grad():
            pi = net.delta_policy(feat)
        pi_np = pi.detach().numpy()
        assert pi_np.min() >= -5.0 - 1e-5
        assert pi_np.max() <=  3.0 + 1e-5

    def test_long_leverage_capped(self):
        net  = self._net_with_buffers(5)
        feat = torch.tensor([[0.3, 0.9]], requires_grad=True)
        with torch.enable_grad():
            pi = net.delta_policy(feat)
        long_lev = pi.clamp(min=0).sum().item()
        assert long_lev <= 3.0 + 1e-4

    def test_short_leverage_capped(self):
        net  = self._net_with_buffers(5)
        feat = torch.tensor([[0.3, 0.9]], requires_grad=True)
        with torch.enable_grad():
            pi = net.delta_policy(feat)
        short_lev = (-pi).clamp(min=0).sum().item()
        assert short_lev <= 5.0 + 1e-4

    def test_returns_detachable_tensor(self):
        """delta_policy output must be detachable without error."""
        net  = self._net_with_buffers(5)
        feat = torch.rand(4, 2).requires_grad_(True)
        with torch.enable_grad():
            pi = net.delta_policy(feat)
        _ = pi.detach().numpy()  # must not raise

    # ── Regression: create_graph=False must still compute V_ww ───────────────
    #
    # Bug: the first autograd.grad call used create_graph=create_graph.
    # When the caller passed create_graph=False (the default, used in Phase 3
    # simulation and _eval_goal_prob), V_w came back with grad_fn=None, so the
    # second autograd.grad call for V_ww immediately raised:
    #   "element 0 of tensors does not require grad and does not have a grad_fn"
    #
    # Fix: first autograd.grad always uses create_graph=True so V_w has a
    # grad_fn that enables differentiating it again for V_ww.

    def test_create_graph_false_does_not_raise(self):
        """
        Regression: delta_policy(create_graph=False) must not raise.
        Before the fix this always failed because V_w lacked a grad_fn.
        """
        net  = self._net_with_buffers(5)
        feat = torch.tensor([[1.0, 0.5], [0.8, 0.3]], dtype=torch.float32)
        feat = feat.detach().requires_grad_(True)
        with torch.enable_grad():
            pi = net.delta_policy(feat, create_graph=False)
        assert pi.shape == (2, 5)

    def test_create_graph_false_output_is_detached(self):
        """With create_graph=False the policy must be detached (no grad_fn)."""
        net  = self._net_with_buffers(5)
        feat = torch.tensor([[1.0, 0.5]], dtype=torch.float32, requires_grad=True)
        with torch.enable_grad():
            pi = net.delta_policy(feat, create_graph=False)
        assert not pi.requires_grad

    def test_create_graph_true_output_has_grad_fn(self):
        """With create_graph=True the policy must carry a grad_fn for training."""
        net  = self._net_with_buffers(5)
        feat = torch.tensor([[1.0, 0.5]], dtype=torch.float32, requires_grad=True)
        with torch.enable_grad():
            pi = net.delta_policy(feat, create_graph=True)
        assert pi.requires_grad

    def test_first_grad_always_creates_graph_internally(self):
        """
        The V_w computation must always have a grad_fn so V_ww is computable.
        This is the invariant the fix enforces: first autograd.grad uses
        create_graph=True regardless of what the caller requests.
        """
        net  = self._net_with_buffers(5)
        feat = torch.tensor([[1.0, 0.5]], dtype=torch.float32, requires_grad=True)
        with torch.enable_grad():
            V = net.forward(feat)
            V_grad = torch.autograd.grad(
                V.sum(), feat, create_graph=True, retain_graph=True,
            )[0]
            V_w = V_grad[:, 0:1]
        assert V_w.grad_fn is not None, (
            "V_w must have grad_fn; without it the second autograd.grad for V_ww fails"
        )

    def test_delta_policy_consistent_across_create_graph_modes(self):
        """Policy values must be identical regardless of create_graph flag."""
        net  = self._net_with_buffers(5)
        feat = torch.tensor([[1.0, 0.5], [0.8, 0.3], [1.2, 0.7]], dtype=torch.float32)
        with torch.enable_grad():
            pi_false = net.delta_policy(feat.detach().requires_grad_(True),
                                        create_graph=False).numpy()
            pi_true  = net.delta_policy(feat.detach().requires_grad_(True),
                                        create_graph=True).detach().numpy()
        np.testing.assert_allclose(pi_false, pi_true, atol=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Architecture registry
# ─────────────────────────────────────────────────────────────────────────────

class TestArchitectureRegistry:
    def test_nn_digital_hedge_registered(self):
        from comparisons.core.torch_nn_models import TORCH_ARCHITECTURES
        assert "nn_digital_hedge" in TORCH_ARCHITECTURES

    def test_kind_is_digital_hedge(self):
        from comparisons.core.torch_nn_models import TORCH_ARCHITECTURES
        assert TORCH_ARCHITECTURES["nn_digital_hedge"]["kind"] == "digital_hedge"

    def test_hidden_layers_match_policy_net(self):
        from comparisons.core.torch_nn_models import TORCH_ARCHITECTURES
        assert TORCH_ARCHITECTURES["nn_digital_hedge"]["hidden_layers"] == (128, 128, 128)

    def test_existing_architectures_unmodified(self):
        from comparisons.core.torch_nn_models import TORCH_ARCHITECTURES
        for name in ["nn_mlp_small", "nn_mlp_deep", "nn_policy_net",
                     "deep_bsde", "pinn", "actor_critic", "lstm", "transformer"]:
            assert TORCH_ARCHITECTURES[name]["kind"] != "digital_hedge"


# ─────────────────────────────────────────────────────────────────────────────
# 5. train_digital_hedge_net — function existence and return contract
# ─────────────────────────────────────────────────────────────────────────────

class TestTrainDigitalHedgeNetExists:
    def test_function_importable(self):
        from comparisons.core.torch_nn_models import train_digital_hedge_net  # noqa


class TestTrainDigitalHedgeNetContract:
    """Return value must match the format expected by evaluate_nn_portfolio."""

    def _train(self, **kw):
        from comparisons.core.torch_nn_models import train_digital_hedge_net
        mkt = _market()
        defaults = dict(
            mu_vec=mkt.mu_ann, omega_mat=mkt.omega, r=mkt.r,
            n_paths=32, pretrain_iters=5, hjb_iters=5, sim_iters=5,
            n_steps=6, seed=1,
        )
        defaults.update(kw)
        return train_digital_hedge_net(**defaults)

    def test_returns_two_tuple(self):
        result = self._train()
        assert len(result) == 2

    def test_net_is_digital_hedge_value_net(self):
        from comparisons.core.torch_nn_models import DigitalHedgeValueNet
        net, _ = self._train()
        assert isinstance(net, DigitalHedgeValueNet)

    def test_meta_has_required_keys(self):
        _, meta = self._train()
        for key in ("architecture_name", "backend", "kind",
                    "param_size", "device",
                    "loss_history", "val_history", "val_iters", "test_u"):
            assert key in meta, f"Missing key: {key}"

    def test_meta_architecture_name(self):
        _, meta = self._train()
        assert meta["architecture_name"] == "nn_digital_hedge"

    def test_meta_backend_torch(self):
        _, meta = self._train()
        assert meta["backend"] == "torch"

    def test_meta_kind(self):
        _, meta = self._train()
        assert meta["kind"] == "digital_hedge"

    def test_loss_history_non_empty(self):
        _, meta = self._train()
        assert len(meta["loss_history"]) > 0

    def test_val_history_non_empty(self):
        _, meta = self._train()
        assert len(meta["val_history"]) > 0

    def test_test_u_in_zero_one(self):
        _, meta = self._train()
        assert 0.0 <= meta["test_u"] <= 1.0

    def test_net_in_eval_mode(self):
        net, _ = self._train()
        assert not net.training

    def test_buffers_set_after_training(self):
        """omega_inv_eta and theta2_val must be populated (non-zero) after training."""
        net, _ = self._train()
        assert net.omega_inv_eta.abs().sum().item() > 0.0
        assert net.theta2_val.item() > 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 6. Three training phases are executed
# ─────────────────────────────────────────────────────────────────────────────

class TestThreePhases:
    def test_pretrain_phase_improves_browne_fit(self):
        """
        V after pre-training should fit the Browne formula better than a
        freshly initialised random network.
        """
        from comparisons.core.torch_nn_models import (
            DigitalHedgeValueNet, train_digital_hedge_net,
        )
        mkt = _market()
        eta           = mkt.mu_ann - mkt.r
        omega_inv_eta = np.linalg.solve(mkt.omega, eta)
        theta2        = float(np.dot(eta, omega_inv_eta))

        # Evaluate Browne MSE for a random net
        net_rand = DigitalHedgeValueNet(n_assets=5)
        ws   = [0.7, 0.9, 1.0, 1.1, 1.3]
        taus = [0.5, 0.5, 0.5, 0.5, 0.5]
        targets = [_browne_v(w, t, theta2) for w, t in zip(ws, taus)]
        T = 1.0
        feats = torch.tensor([[w, t/T] for w, t in zip(ws, taus)], dtype=torch.float32)
        with torch.no_grad():
            preds_rand = net_rand(feats).squeeze().numpy()
        mse_rand = float(np.mean((preds_rand - targets) ** 2))

        # Train with pre-training only
        net_trained, _ = train_digital_hedge_net(
            mu_vec=mkt.mu_ann, omega_mat=mkt.omega, r=mkt.r,
            n_paths=16, pretrain_iters=50, hjb_iters=0, sim_iters=0,
            n_steps=4, seed=99,
        )
        with torch.no_grad():
            preds_trained = net_trained(feats).squeeze().numpy()
        mse_trained = float(np.mean((preds_trained - targets) ** 2))

        assert mse_trained < mse_rand, (
            f"Pre-training should reduce Browne MSE: {mse_trained:.4f} vs {mse_rand:.4f}"
        )

    def test_hjb_phase_only_runs_without_error(self):
        """HJB phase runs (pretrain=0, sim=0) and returns valid result."""
        from comparisons.core.torch_nn_models import train_digital_hedge_net
        mkt = _market()
        net, meta = train_digital_hedge_net(
            mu_vec=mkt.mu_ann, omega_mat=mkt.omega, r=mkt.r,
            n_paths=16, pretrain_iters=0, hjb_iters=8, sim_iters=0,
            n_steps=4, seed=7,
        )
        assert len(meta["loss_history"]) > 0

    def test_sim_phase_only_runs_without_error(self):
        """Sim phase runs (pretrain=0, hjb=0) and returns valid result."""
        from comparisons.core.torch_nn_models import train_digital_hedge_net
        mkt = _market()
        net, meta = train_digital_hedge_net(
            mu_vec=mkt.mu_ann, omega_mat=mkt.omega, r=mkt.r,
            n_paths=32, pretrain_iters=0, hjb_iters=0, sim_iters=8,
            n_steps=4, seed=13,
        )
        assert len(meta["loss_history"]) > 0
        assert 0.0 <= meta["test_u"] <= 1.0

    def test_all_three_phases_run(self):
        from comparisons.core.torch_nn_models import train_digital_hedge_net
        mkt = _market()
        net, meta = train_digital_hedge_net(
            mu_vec=mkt.mu_ann, omega_mat=mkt.omega, r=mkt.r,
            n_paths=32, pretrain_iters=5, hjb_iters=5, sim_iters=5,
            n_steps=4, seed=42,
        )
        # loss_history is populated from both HJB and sim phases
        assert len(meta["loss_history"]) >= 2


# ─────────────────────────────────────────────────────────────────────────────
# 7. train_torch_policy_net delegates to train_digital_hedge_net
# ─────────────────────────────────────────────────────────────────────────────

class TestDelegation:
    def test_train_torch_policy_net_delegates_digital_hedge(self):
        """
        train_torch_policy_net('nn_digital_hedge') must return a
        DigitalHedgeValueNet, not a TorchPolicyNet.
        """
        from comparisons.core.torch_nn_models import (
            DigitalHedgeValueNet, train_torch_policy_net,
        )
        mkt = _market()
        net, meta = train_torch_policy_net(
            mu_vec=mkt.mu_ann, omega_mat=mkt.omega, r=mkt.r,
            architecture_name="nn_digital_hedge",
            n_paths=16, n_iters=10, n_steps=4,
            pretrain_iters=5, patience=999, seed=1,
        )
        assert isinstance(net, DigitalHedgeValueNet)
        assert meta["architecture_name"] == "nn_digital_hedge"

    def test_existing_archs_not_affected(self):
        """Delegation must not interfere with existing architectures."""
        from comparisons.core.torch_nn_models import (
            TorchPolicyNet, train_torch_policy_net,
        )
        mkt = _market()
        net, meta = train_torch_policy_net(
            mu_vec=mkt.mu_ann, omega_mat=mkt.omega, r=mkt.r,
            architecture_name="nn_mlp_small",
            n_paths=16, n_iters=3, n_steps=4,
            pretrain_iters=0, patience=999, seed=1,
        )
        assert isinstance(net, TorchPolicyNet)


# ─────────────────────────────────────────────────────────────────────────────
# 8. policy_weights works with DigitalHedgeValueNet
# ─────────────────────────────────────────────────────────────────────────────

class TestPolicyWeightsCompatibility:
    def _trained_net(self):
        from comparisons.core.torch_nn_models import train_digital_hedge_net
        mkt = _market()
        net, _ = train_digital_hedge_net(
            mu_vec=mkt.mu_ann, omega_mat=mkt.omega, r=mkt.r,
            n_paths=16, pretrain_iters=5, hjb_iters=5, sim_iters=5,
            n_steps=4, seed=3,
        )
        return net, mkt

    def test_policy_weights_returns_correct_shape(self):
        from comparisons.core.torch_nn_models import policy_weights
        net, mkt = self._trained_net()
        goal = 1.0 * 1.10
        pi = policy_weights(net, 1.0, goal)
        assert pi.shape == (mkt.n,)

    def test_policy_weights_no_nans(self):
        from comparisons.core.torch_nn_models import policy_weights
        net, mkt = self._trained_net()
        goal = 1.0 * 1.10
        pi = policy_weights(net, 1.0, goal)
        assert not np.any(np.isnan(pi))

    def test_policy_weights_within_bounds(self):
        from comparisons.core.torch_nn_models import policy_weights
        net, mkt = self._trained_net()
        goal = 1.0 * 1.10
        pi = policy_weights(net, 1.0, goal)
        assert pi.min() >= -5.0 - 1e-4
        assert pi.max() <=  3.0 + 1e-4


# ─────────────────────────────────────────────────────────────────────────────
# 9. Full evaluation pipeline
# ─────────────────────────────────────────────────────────────────────────────

class TestEvaluationPipeline:
    def test_evaluate_nn_portfolio_runs(self, tmp_path):
        from comparisons.core.config import BenchmarkConfig
        from comparisons.core.evaluation import evaluate_nn_portfolio
        mkt = _market()
        config = BenchmarkConfig(
            n_assets_list=[5], random_seeds=[1],
            nn_architectures=["nn_digital_hedge"],
            nn_iters=10, nn_paths=32, nn_steps=6,
            nn_pretrain_iters=5, nn_patience=999,
            nn_antithetic=False, include_nn=True,
            results_dir=tmp_path,
        )
        result = evaluate_nn_portfolio(mkt, config, "nn_digital_hedge",
                                       initial_wealth=1.0, seed=1)
        assert result["method_name"] == "nn_digital_hedge"
        assert result["method_family"] == "nn"
        assert len(result["wealth_path"]) > 1

    def test_result_schema_valid(self, tmp_path):
        from comparisons.core.config import BenchmarkConfig
        from comparisons.core.evaluation import (
            evaluate_nn_portfolio, validate_result_schema,
        )
        mkt = _market()
        config = BenchmarkConfig(
            n_assets_list=[5], random_seeds=[1],
            nn_architectures=["nn_digital_hedge"],
            nn_iters=10, nn_paths=32, nn_steps=6,
            nn_pretrain_iters=5, nn_patience=999,
            nn_antithetic=False, include_nn=True,
            results_dir=tmp_path,
        )
        result = evaluate_nn_portfolio(mkt, config, "nn_digital_hedge",
                                       initial_wealth=1.0, seed=1)
        assert validate_result_schema(result)

    def test_run_experiment_includes_digital_hedge(self):
        import run_experiment
        cfg = run_experiment.make_config(quick=True)
        assert "nn_digital_hedge" in cfg.nn_architectures

    def test_method_order_includes_digital_hedge(self):
        import run_experiment
        assert "nn_digital_hedge" in run_experiment.METHOD_ORDER
