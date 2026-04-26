"""
test_ste_goalreach.py
----------------------
Red-green TDD tests for the GoalReachSTE Straight-Through Estimator approach.

Philosophy
----------
The sigmoid smoothing used by every other NN method changes the incentive
structure: it rewards pushing wealth *above* the goal, whereas the true
HJB terminal condition is the binary step 1{W_T >= goal}.

GoalReachSTE resolves this by:
  Forward  : exact step function  -> loss = -P(W_T >= goal)   (matches fd_nd)
  Backward : sigmoid surrogate    -> gradient flows to the network

These tests verify:
  1. The STE autograd function has the correct forward / backward behaviour.
  2. The new architecture is registered in TORCH_ARCHITECTURES.
  3. _terminal_utility dispatches correctly for 'goalreach_ste'.
  4. train_torch_policy_net picks up the STE utility from the arch spec.
  5. The full evaluate_nn_portfolio pipeline works end-to-end.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

# Make project root importable regardless of where pytest is invoked from
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

def _small_market():
    """Minimal 5-asset market data for fast integration tests."""
    from real_data_loader import load_portfolio
    return load_portfolio(5, start="2020-01-01", end="2020-06-30")


# ─────────────────────────────────────────────────────────────────────────────
# 1. GoalReachSTE – class existence
# ─────────────────────────────────────────────────────────────────────────────

class TestGoalReachSTEExists:
    def test_class_is_importable(self):
        """GoalReachSTE must be importable from torch_nn_models."""
        from comparisons.core.torch_nn_models import GoalReachSTE  # noqa: F401


# ─────────────────────────────────────────────────────────────────────────────
# 2. GoalReachSTE – forward pass is the exact step function
# ─────────────────────────────────────────────────────────────────────────────

class TestGoalReachSTEForward:
    def test_positive_input_gives_one(self):
        from comparisons.core.torch_nn_models import GoalReachSTE
        x = torch.tensor([0.1, 1.0, 100.0])
        out = GoalReachSTE.apply(x, 0.05)
        assert out.tolist() == [1.0, 1.0, 1.0]

    def test_negative_input_gives_zero(self):
        from comparisons.core.torch_nn_models import GoalReachSTE
        x = torch.tensor([-0.1, -1.0, -100.0])
        out = GoalReachSTE.apply(x, 0.05)
        assert out.tolist() == [0.0, 0.0, 0.0]

    def test_zero_input_gives_one(self):
        """Convention: step is 1 at exactly 0 (x >= 0)."""
        from comparisons.core.torch_nn_models import GoalReachSTE
        x = torch.tensor([0.0])
        out = GoalReachSTE.apply(x, 0.05)
        assert out.item() == 1.0

    def test_output_is_binary(self):
        """Every output value must be exactly 0.0 or 1.0."""
        from comparisons.core.torch_nn_models import GoalReachSTE
        x = torch.linspace(-2.0, 2.0, 41)
        out = GoalReachSTE.apply(x, 0.05)
        unique = set(out.tolist())
        assert unique <= {0.0, 1.0}

    def test_forward_differs_from_sigmoid(self):
        """STE forward must not equal sigmoid (they coincide nowhere except 0.5)."""
        from comparisons.core.torch_nn_models import GoalReachSTE
        x = torch.tensor([0.5])
        ste_out = GoalReachSTE.apply(x, 0.05)
        sig_out = torch.sigmoid(x / 0.05)
        # STE gives 1.0; sigmoid gives ~1.0 but never exactly 1.0
        assert ste_out.item() == 1.0
        assert sig_out.item() < 1.0 - 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# 3. GoalReachSTE – backward pass provides informative gradients
# ─────────────────────────────────────────────────────────────────────────────

class TestGoalReachSTEBackward:
    def test_gradient_is_non_zero(self):
        """Gradient must not be zero (unlike a raw step function)."""
        from comparisons.core.torch_nn_models import GoalReachSTE
        x = torch.tensor([0.2, -0.2, 0.05], requires_grad=True)
        out = GoalReachSTE.apply(x, 0.05)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.abs().sum().item() > 0.0

    def test_gradient_matches_sigmoid_surrogate(self):
        """Backward must use the sigmoid derivative as surrogate."""
        from comparisons.core.torch_nn_models import GoalReachSTE
        temp = 0.05
        val = 0.1
        x = torch.tensor([val], requires_grad=True)
        out = GoalReachSTE.apply(x, temp)
        out.sum().backward()

        s = math.exp(val / temp) / (1.0 + math.exp(val / temp))   # sigmoid(val/temp)
        expected = s * (1.0 - s) / temp
        assert abs(x.grad.item() - expected) < 1e-5

    def test_gradient_is_symmetric_around_zero(self):
        """Sigmoid surrogate is symmetric: grad(x) == grad(-x)."""
        from comparisons.core.torch_nn_models import GoalReachSTE
        temp = 0.05
        for val in [0.1, 0.5, 1.0]:
            x_pos = torch.tensor([val],  requires_grad=True)
            x_neg = torch.tensor([-val], requires_grad=True)
            GoalReachSTE.apply(x_pos, temp).sum().backward()
            GoalReachSTE.apply(x_neg, temp).sum().backward()
            assert abs(x_pos.grad.item() - x_neg.grad.item()) < 1e-6

    def test_gradient_peaks_at_zero(self):
        """Sigmoid surrogate gradient is largest at x=0."""
        from comparisons.core.torch_nn_models import GoalReachSTE
        temp = 0.05
        vals = [0.0, 0.1, 0.5, 1.0]
        grads = []
        for val in vals:
            x = torch.tensor([val], requires_grad=True)
            GoalReachSTE.apply(x, temp).sum().backward()
            grads.append(x.grad.item())
        assert grads[0] == max(grads)

    def test_no_gradient_flows_to_temp(self):
        """Temperature is not differentiable (second return of backward is None)."""
        from comparisons.core.torch_nn_models import GoalReachSTE
        x = torch.tensor([0.1], requires_grad=True)
        # apply() does not expose temp as a leaf tensor — just verify no error
        out = GoalReachSTE.apply(x, 0.05)
        out.sum().backward()   # must not raise


# ─────────────────────────────────────────────────────────────────────────────
# 4. Architecture registry
# ─────────────────────────────────────────────────────────────────────────────

class TestArchitectureRegistry:
    def test_nn_ste_goalreach_in_torch_architectures(self):
        from comparisons.core.torch_nn_models import TORCH_ARCHITECTURES
        assert "nn_ste_goalreach" in TORCH_ARCHITECTURES

    def test_spec_kind_is_mlp(self):
        from comparisons.core.torch_nn_models import TORCH_ARCHITECTURES
        assert TORCH_ARCHITECTURES["nn_ste_goalreach"]["kind"] == "mlp"

    def test_spec_has_utility_goalreach_ste(self):
        """Arch spec must advertise its non-default utility."""
        from comparisons.core.torch_nn_models import TORCH_ARCHITECTURES
        assert TORCH_ARCHITECTURES["nn_ste_goalreach"].get("utility") == "goalreach_ste"

    def test_spec_hidden_layers_match_policy_net(self):
        """nn_ste_goalreach uses the same capacity as nn_policy_net."""
        from comparisons.core.torch_nn_models import TORCH_ARCHITECTURES
        assert TORCH_ARCHITECTURES["nn_ste_goalreach"]["hidden_layers"] == (128, 128, 128)

    def test_existing_architectures_unmodified(self):
        """No existing architecture should have a 'utility' key."""
        from comparisons.core.torch_nn_models import TORCH_ARCHITECTURES
        existing = [
            "nn_mlp_small", "nn_mlp_deep", "nn_policy_net",
            "deep_bsde", "pinn", "actor_critic", "lstm", "transformer",
        ]
        for name in existing:
            assert "utility" not in TORCH_ARCHITECTURES[name], (
                f"{name} should not have a 'utility' key — existing behaviour must be preserved"
            )


# ─────────────────────────────────────────────────────────────────────────────
# 5. _terminal_utility dispatch
# ─────────────────────────────────────────────────────────────────────────────

class TestTerminalUtilityDispatch:
    def test_goalreach_ste_returns_binary_values(self):
        from comparisons.core.torch_nn_models import _terminal_utility
        W    = torch.tensor([[0.5], [0.9], [1.0], [1.1], [2.0]])
        goal = 1.0
        out  = _terminal_utility(W, goal, "goalreach_ste", 0.5, 1.2, 1.0, sig_temp=0.05)
        unique = set(out.tolist())
        assert unique <= {0.0, 1.0}, f"Expected binary output, got: {unique}"

    def test_goalreach_ste_correct_boundary(self):
        from comparisons.core.torch_nn_models import _terminal_utility
        W    = torch.tensor([[0.99], [1.00], [1.01]])
        goal = 1.0
        out  = _terminal_utility(W, goal, "goalreach_ste", 0.5, 1.2, 1.0, sig_temp=0.05)
        assert out[0].item() == 0.0, "W=0.99 < goal -> 0"
        assert out[1].item() == 1.0, "W=1.00 = goal -> 1"
        assert out[2].item() == 1.0, "W=1.01 > goal -> 1"

    def test_goalreach_ste_has_gradient(self):
        """Despite step-function output, gradients must flow through the network."""
        from comparisons.core.torch_nn_models import _terminal_utility
        W = torch.tensor([[1.05]], requires_grad=True)
        out = _terminal_utility(W, 1.0, "goalreach_ste", 0.5, 1.2, 1.0, sig_temp=0.05)
        out.sum().backward()
        assert W.grad is not None
        assert W.grad.abs().item() > 0.0

    def test_goalreach_ste_output_differs_from_sigmoid(self):
        """STE output != sigmoid output for any non-extreme wealth."""
        from comparisons.core.torch_nn_models import _terminal_utility
        W    = torch.tensor([[1.1]])   # clearly above goal; sigmoid < 1
        goal = 1.0
        ste = _terminal_utility(W, goal, "goalreach_ste", 0.5, 1.2, 1.0, sig_temp=0.05)
        sig = _terminal_utility(W, goal, "goalreach",     0.5, 1.2, 1.0, sig_temp=0.05)
        assert ste.item() == 1.0
        assert sig.item() < 1.0 - 1e-6

    def test_existing_goalreach_utility_unchanged(self):
        """'goalreach' (sigmoid) must still work exactly as before."""
        from comparisons.core.torch_nn_models import _terminal_utility
        W    = torch.tensor([[1.0]])
        goal = 1.0
        out  = _terminal_utility(W, goal, "goalreach", 0.5, 1.2, 1.0, sig_temp=0.05)
        # sigmoid(0/0.05) = sigmoid(0) = 0.5
        assert abs(out.item() - 0.5) < 1e-5

    def test_unknown_utility_still_raises(self):
        from comparisons.core.torch_nn_models import _terminal_utility
        with pytest.raises(ValueError):
            _terminal_utility(torch.tensor([[1.0]]), 1.0, "bad_utility",
                              0.5, 1.2, 1.0, sig_temp=0.05)


# ─────────────────────────────────────────────────────────────────────────────
# 6. train_torch_policy_net picks up utility from arch spec
# ─────────────────────────────────────────────────────────────────────────────

class TestTrainingUsesSTEUtility:
    def test_ste_utility_called_during_training(self, monkeypatch):
        """
        Patch _terminal_utility to record which utility strings are used.
        When architecture_name='nn_ste_goalreach', every call must use
        'goalreach_ste', never 'goalreach'.
        """
        import comparisons.core.torch_nn_models as mod

        recorded = []
        original = mod._terminal_utility

        def spy(W, goal, utility, *args, **kwargs):
            recorded.append(utility)
            return original(W, goal, utility, *args, **kwargs)

        monkeypatch.setattr(mod, "_terminal_utility", spy)

        mkt = _small_market()
        mod.train_torch_policy_net(
            mu_vec=mkt.mu_ann,
            omega_mat=mkt.omega,
            r=mkt.r,
            architecture_name="nn_ste_goalreach",
            n_paths=32,
            n_iters=5,
            n_steps=8,
            pretrain_iters=0,
            patience=999,
            seed=42,
        )

        assert len(recorded) > 0, "No utility calls recorded — training may have been skipped"
        assert all(u == "goalreach_ste" for u in recorded), (
            f"Expected only 'goalreach_ste', got: {set(recorded)}"
        )

    def test_other_archs_still_use_sigmoid_utility(self, monkeypatch):
        """Existing architectures must continue to use 'goalreach' (sigmoid)."""
        import comparisons.core.torch_nn_models as mod

        recorded = []
        original = mod._terminal_utility

        def spy(W, goal, utility, *args, **kwargs):
            recorded.append(utility)
            return original(W, goal, utility, *args, **kwargs)

        monkeypatch.setattr(mod, "_terminal_utility", spy)

        mkt = _small_market()
        mod.train_torch_policy_net(
            mu_vec=mkt.mu_ann,
            omega_mat=mkt.omega,
            r=mkt.r,
            architecture_name="nn_policy_net",
            n_paths=32,
            n_iters=3,
            n_steps=8,
            pretrain_iters=0,
            patience=999,
            seed=1,
        )

        assert all(u == "goalreach" for u in recorded), (
            f"nn_policy_net should use 'goalreach', got: {set(recorded)}"
        )

    def test_train_completes_and_returns_histories(self):
        """nn_ste_goalreach training must produce train/val/test_u histories."""
        from comparisons.core.torch_nn_models import train_torch_policy_net
        mkt = _small_market()
        net, meta = train_torch_policy_net(
            mu_vec=mkt.mu_ann,
            omega_mat=mkt.omega,
            r=mkt.r,
            architecture_name="nn_ste_goalreach",
            n_paths=32,
            n_iters=10,
            n_steps=8,
            pretrain_iters=0,
            patience=999,
            seed=7,
        )
        assert meta["architecture_name"] == "nn_ste_goalreach"
        assert meta["backend"] == "torch"
        assert len(meta["loss_history"]) > 0
        assert len(meta["val_history"]) > 0
        assert not math.isnan(meta["test_u"])

    def test_train_loss_is_probability_in_zero_one(self):
        """
        With STE utility, train loss = -E[1{W_T>=goal}] so each recorded
        E[U] value must be in [0, 1].
        """
        from comparisons.core.torch_nn_models import train_torch_policy_net
        mkt = _small_market()
        _, meta = train_torch_policy_net(
            mu_vec=mkt.mu_ann,
            omega_mat=mkt.omega,
            r=mkt.r,
            architecture_name="nn_ste_goalreach",
            n_paths=64,
            n_iters=8,
            n_steps=8,
            pretrain_iters=0,
            patience=999,
            seed=3,
        )
        for val in meta["loss_history"]:
            assert 0.0 <= val <= 1.0 + 1e-6, f"Loss out of [0,1]: {val}"


# ─────────────────────────────────────────────────────────────────────────────
# 7. End-to-end integration with evaluate_nn_portfolio
# ─────────────────────────────────────────────────────────────────────────────

class TestEvaluationPipeline:
    def test_evaluate_nn_portfolio_runs(self, tmp_path):
        """nn_ste_goalreach must work through the full evaluation pipeline."""
        from comparisons.core.config import BenchmarkConfig
        from comparisons.core.evaluation import evaluate_nn_portfolio
        mkt = _small_market()
        config = BenchmarkConfig(
            n_assets_list=[5],
            random_seeds=[1],
            nn_architectures=["nn_ste_goalreach"],
            nn_iters=5,
            nn_paths=32,
            nn_steps=8,
            nn_pretrain_iters=0,
            nn_patience=999,
            nn_antithetic=False,
            include_nn=True,
            results_dir=tmp_path,
        )
        result = evaluate_nn_portfolio(mkt, config, "nn_ste_goalreach",
                                       initial_wealth=1.0, seed=1)
        assert result["method_name"] == "nn_ste_goalreach"
        assert result["method_family"] == "nn"
        assert len(result["wealth_path"]) > 1
        assert "nn_param_count" in result
        assert "train_history" in result
        assert "val_history" in result
        assert "test_u" in result

    def test_result_schema_valid(self, tmp_path):
        """Result dict must pass validate_result_schema."""
        from comparisons.core.config import BenchmarkConfig
        from comparisons.core.evaluation import evaluate_nn_portfolio, validate_result_schema
        mkt = _small_market()
        config = BenchmarkConfig(
            n_assets_list=[5],
            random_seeds=[1],
            nn_architectures=["nn_ste_goalreach"],
            nn_iters=5,
            nn_paths=32,
            nn_steps=8,
            nn_pretrain_iters=0,
            nn_patience=999,
            nn_antithetic=False,
            include_nn=True,
            results_dir=tmp_path,
        )
        result = evaluate_nn_portfolio(mkt, config, "nn_ste_goalreach",
                                       initial_wealth=1.0, seed=1)
        assert validate_result_schema(result)

    def test_run_experiment_includes_ste_goalreach(self, tmp_path):
        """run_experiment.make_config() must include nn_ste_goalreach in arch list."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(ROOT))
        import run_experiment
        cfg = run_experiment.make_config(quick=True)
        assert "nn_ste_goalreach" in cfg.nn_architectures

    def test_method_order_includes_ste_goalreach(self):
        """run_experiment.METHOD_ORDER must list nn_ste_goalreach for sorted plots."""
        import sys
        sys.path.insert(0, str(ROOT))
        import run_experiment
        assert "nn_ste_goalreach" in run_experiment.METHOD_ORDER
