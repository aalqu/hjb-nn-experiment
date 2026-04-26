"""
test_new_features.py
---------------------
Red-green TDD for three additions:
  1. _require_torch bug fix   — train_digital_hedge_net must not raise NameError
  2. nn_historical_replay     — architecture, rich features, training pipeline
  3. goal multiplier sweep    — run_experiment produces goal_mult column

Run before the fix to confirm RED, then after to confirm GREEN.
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

# ── shared tiny market ────────────────────────────────────────────────────────

def _market(n: int = 3):
    mu   = np.array([0.08, 0.06, 0.04][:n], dtype=float)
    omega = np.eye(n) * 0.04 + np.full((n, n), 0.005)
    np.fill_diagonal(omega, 0.04)
    r    = 0.03
    # 3 years of fake daily returns (T × n)
    rng  = np.random.default_rng(0)
    T    = 252 * 3
    rets = rng.normal(mu / 252, np.sqrt(np.diag(omega) / 252), size=(T, n)).astype(np.float32)
    return mu, omega, r, rets


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  _require_torch bug
# ═══════════════════════════════════════════════════════════════════════════════

class TestRequireTorchBug:
    """train_digital_hedge_net must not raise NameError: '_require_torch'."""

    def test_digital_hedge_does_not_raise_name_error(self):
        from comparisons.core.torch_nn_models import train_torch_policy_net
        mu, omega, r, _ = _market(3)
        # A NameError would surface immediately on entry to the function.
        try:
            net, meta = train_torch_policy_net(
                mu_vec=mu, omega_mat=omega, r=r,
                architecture_name="nn_digital_hedge",
                w0=1.0, goal_mult=1.10,
                n_paths=16, n_iters=2,
                lr=1e-3, n_steps=8,
                pretrain_iters=2,
                patience=5, T=1.0, seed=1,
            )
        except NameError as e:
            pytest.fail(f"NameError raised: {e}")

    def test_digital_hedge_long_only_does_not_raise_name_error(self):
        from comparisons.core.torch_nn_models import train_torch_policy_net
        mu, omega, r, _ = _market(3)
        try:
            net, meta = train_torch_policy_net(
                mu_vec=mu, omega_mat=omega, r=r,
                architecture_name="nn_digital_hedge_long_only",
                w0=1.0, goal_mult=1.10,
                n_paths=16, n_iters=2,
                lr=1e-3, n_steps=8,
                pretrain_iters=2,
                patience=5, T=1.0, seed=1,
            )
        except NameError as e:
            pytest.fail(f"NameError raised: {e}")

    def test_digital_hedge_returns_expected_keys(self):
        from comparisons.core.torch_nn_models import train_torch_policy_net
        mu, omega, r, _ = _market(3)
        net, meta = train_torch_policy_net(
            mu_vec=mu, omega_mat=omega, r=r,
            architecture_name="nn_digital_hedge",
            w0=1.0, goal_mult=1.10,
            n_paths=16, n_iters=2,
            lr=1e-3, n_steps=8,
            pretrain_iters=2,
            patience=5, T=1.0, seed=1,
        )
        for key in ("architecture_name", "backend", "kind", "param_size",
                    "loss_history", "val_history", "val_iters", "test_u"):
            assert key in meta, f"Missing key: {key}"


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  nn_historical_replay  — architecture + rich features + training
# ═══════════════════════════════════════════════════════════════════════════════

class TestHistoricalReplayArchitecture:
    """Architecture registration and model construction."""

    def test_in_torch_architectures(self):
        from comparisons.core.torch_nn_models import TORCH_ARCHITECTURES
        assert "nn_historical_replay" in TORCH_ARCHITECTURES

    def test_kind_is_historical_replay(self):
        from comparisons.core.torch_nn_models import TORCH_ARCHITECTURES
        assert TORCH_ARCHITECTURES["nn_historical_replay"]["kind"] == "historical_replay"

    def test_n_features_equals_rich_feature_dim(self):
        from comparisons.core.torch_nn_models import TORCH_ARCHITECTURES, RICH_FEATURE_DIM
        spec = TORCH_ARCHITECTURES["nn_historical_replay"]
        assert spec.get("n_features") == RICH_FEATURE_DIM

    def test_long_only_variant_in_torch_architectures(self):
        from comparisons.core.torch_nn_models import TORCH_ARCHITECTURES
        assert "nn_historical_replay_long_only" in TORCH_ARCHITECTURES

    def test_long_only_has_correct_constraints(self):
        from comparisons.core.torch_nn_models import TORCH_ARCHITECTURES
        c = TORCH_ARCHITECTURES["nn_historical_replay_long_only"]["constraints"]
        assert c["d"] == 0.0
        assert c["u"] == 1.0
        assert c["max_long"] == 1.0
        assert c["max_short"] == 0.0

    def test_build_model_produces_correct_n_features(self):
        from comparisons.core.torch_nn_models import _build_model, RICH_FEATURE_DIM
        model = _build_model("nn_historical_replay", n_assets=3, n_steps=8,
                             d=0.0, u=1.0)
        assert model.n_features == RICH_FEATURE_DIM

    def test_model_accepts_rich_feature_input(self):
        from comparisons.core.torch_nn_models import _build_model, RICH_FEATURE_DIM
        model = _build_model("nn_historical_replay", n_assets=3, n_steps=8,
                             d=0.0, u=1.0)
        model.eval()
        x = torch.zeros(4, RICH_FEATURE_DIM)
        out = model(x)
        assert out.shape == (4, 3), f"Expected (4, 3), got {out.shape}"

    def test_model_kind_attribute(self):
        from comparisons.core.torch_nn_models import _build_model
        model = _build_model("nn_historical_replay", n_assets=3, n_steps=8,
                             d=0.0, u=1.0)
        assert model.kind == "historical_replay"


class TestRichFeatures:
    """_current_features_rich returns correct shape and values."""

    def _call(self, B=4, step=5, total=20, history_len=0):
        from comparisons.core.torch_nn_models import _current_features_rich, RICH_FEATURE_DIM
        W = torch.full((B, 1), 1.1)
        goal = 1.0
        if history_len > 0:
            history = [torch.full((B, 1), 1.0 + 0.01 * i)
                       for i in range(history_len)]
        else:
            history = []
        feat = _current_features_rich(W, goal, step, total, history)
        return feat, RICH_FEATURE_DIM

    def test_output_shape_no_history(self):
        feat, dim = self._call(B=4, history_len=0)
        assert feat.shape == (4, dim)

    def test_output_shape_with_history(self):
        feat, dim = self._call(B=4, history_len=25)
        assert feat.shape == (4, dim)

    def test_w_norm_feature_correct(self):
        feat, _ = self._call(B=2)
        # w=1.1, goal=1.0 → w_norm should be ~1.1
        assert abs(feat[0, 0].item() - 1.1) < 1e-4

    def test_tau_norm_in_range(self):
        feat, _ = self._call(step=5, total=20)
        tau = feat[0, 1].item()
        assert 0.0 < tau <= 1.0

    def test_log_w_feature(self):
        feat, _ = self._call()
        # feature[2] should be log(w_norm) = log(1.1) ≈ 0.0953
        assert abs(feat[0, 2].item() - math.log(1.1)) < 1e-4

    def test_rolling_features_zero_with_no_history(self):
        feat, _ = self._call(history_len=0)
        # vol and ret should be 0 when history is empty
        assert feat[0, 3].item() == pytest.approx(0.0)
        assert feat[0, 4].item() == pytest.approx(0.0)

    def test_rolling_features_nonzero_with_history(self):
        feat, _ = self._call(history_len=25)
        # rolling vol should be nonzero when history exists
        assert feat[0, 3].item() != pytest.approx(0.0)

    def test_forward_policy_routes_to_rich_features(self):
        from comparisons.core.torch_nn_models import _forward_policy, _build_model
        model = _build_model("nn_historical_replay", n_assets=3, n_steps=8,
                             d=0.0, u=1.0)
        W = torch.full((2, 1), 1.05)
        goal = 1.0
        history = [W.clone()]
        pi, aux = _forward_policy(model, W, goal, 4, 8, history, [4])
        assert pi.shape == (2, 3)
        assert aux is None


class TestHistoricalReplayTraining:
    """train_historical_replay_net contract."""

    def _tiny_run(self, arch="nn_historical_replay", n=3):
        from comparisons.core.torch_nn_models import train_historical_replay_net
        mu, omega, r, rets = _market(n)
        return train_historical_replay_net(
            historical_returns=rets,
            mu_vec=mu, omega_mat=omega, r=r,
            architecture_name=arch,
            w0=1.0, goal_mult=1.10,
            n_paths=16, n_iters=4,
            n_steps=20, block_size=5,
            d=-5.0, u=3.0,
            max_long_leverage=3.0, max_short_leverage=5.0,
            patience=10, T=1.0, seed=1,
        )

    def test_returns_model_and_meta(self):
        net, meta = self._tiny_run()
        assert net is not None
        assert isinstance(meta, dict)

    def test_meta_keys(self):
        _, meta = self._tiny_run()
        for key in ("architecture_name", "backend", "kind",
                    "param_size", "loss_history", "val_history",
                    "val_iters", "test_u"):
            assert key in meta, f"Missing key: {key}"

    def test_kind_is_historical_replay(self):
        _, meta = self._tiny_run()
        assert meta["kind"] == "historical_replay"

    def test_loss_history_non_empty(self):
        _, meta = self._tiny_run()
        assert len(meta["loss_history"]) > 0

    def test_val_history_non_empty(self):
        _, meta = self._tiny_run()
        assert len(meta["val_history"]) > 0

    def test_loss_history_values_in_range(self):
        _, meta = self._tiny_run()
        for v in meta["loss_history"]:
            assert 0.0 <= v <= 1.0, f"P(goal) out of [0,1]: {v}"

    def test_model_outputs_correct_shape(self):
        from comparisons.core.torch_nn_models import RICH_FEATURE_DIM
        net, _ = self._tiny_run(n=3)
        x = torch.zeros(5, RICH_FEATURE_DIM)
        with torch.no_grad():
            out = net(x)
        assert out.shape == (5, 3)

    def test_long_only_constraints_respected(self):
        net, _ = self._tiny_run(arch="nn_historical_replay_long_only")
        from comparisons.core.torch_nn_models import RICH_FEATURE_DIM
        x = torch.rand(32, RICH_FEATURE_DIM)
        with torch.no_grad():
            pi = net(x)
        assert (pi >= -1e-5).all(), "Long-only: negative weights found"
        assert (pi.sum(dim=1) <= 1.0 + 1e-4).all(), "Long-only: total > 1"

    def test_requires_historical_returns(self):
        from comparisons.core.torch_nn_models import train_torch_policy_net
        mu, omega, r, _ = _market(3)
        with pytest.raises(ValueError, match="historical_returns"):
            train_torch_policy_net(
                mu_vec=mu, omega_mat=omega, r=r,
                architecture_name="nn_historical_replay",
                w0=1.0, goal_mult=1.10,
                n_paths=8, n_iters=2, n_steps=8,
                pretrain_iters=0, patience=5, T=1.0, seed=1,
                historical_returns=None,  # must raise
            )

    def test_delegation_through_train_torch_policy_net(self):
        from comparisons.core.torch_nn_models import train_torch_policy_net
        mu, omega, r, rets = _market(3)
        net, meta = train_torch_policy_net(
            mu_vec=mu, omega_mat=omega, r=r,
            architecture_name="nn_historical_replay",
            w0=1.0, goal_mult=1.10,
            n_paths=16, n_iters=4, n_steps=20,
            pretrain_iters=0, patience=10, T=1.0, seed=1,
            historical_returns=rets,
        )
        assert meta["kind"] == "historical_replay"


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  Goal multiplier sweep
# ═══════════════════════════════════════════════════════════════════════════════

class TestGoalMultiplierSweep:
    """run_experiment config and summary include goal_mult."""

    def test_config_has_goal_multipliers(self):
        sys.path.insert(0, str(ROOT))
        from run_experiment import make_config
        cfg = make_config(quick=True)
        assert hasattr(cfg, "goal_multipliers") or \
               hasattr(cfg, "target_multiplier"), \
               "Config missing goal_multipliers"

    def test_goal_multipliers_is_list(self):
        from run_experiment import make_config
        cfg = make_config(quick=True)
        mults = getattr(cfg, "goal_multipliers", [cfg.target_multiplier])
        assert isinstance(mults, list)
        assert len(mults) >= 1

    def test_default_multipliers_include_1_10(self):
        from run_experiment import make_config
        cfg = make_config(quick=False)
        mults = getattr(cfg, "goal_multipliers", [cfg.target_multiplier])
        assert 1.10 in mults

    def test_summary_has_goal_mult_column(self):
        """build_summary must produce a goal_mult column."""
        from run_experiment import build_summary
        import numpy as np
        fake_results = [{
            "method_name": "fd_nd", "method_family": "fd",
            "n_assets": 5, "seed": 1, "goal_mult": 1.15,
            "goal_hit": np.array([True]),
            "wealth_path": np.array([1.0, 1.1, 1.15]),
            "weight_path": np.array([[0.1, 0.1, 0.1]] * 2),
            "target_wealth": 1.15,
            "train_time_sec": 0.0, "solve_time_sec": 1.0,
            "drawdown_path": np.array([0.0, 0.0, 0.0]),
            "gross_leverage_path": np.array([0.3, 0.3]),
            "net_exposure_path": np.array([0.3, 0.3]),
        }]
        df = build_summary(fake_results)
        assert "goal_mult" in df.columns

    def test_summary_goal_mult_value_correct(self):
        from run_experiment import build_summary
        import numpy as np
        fake_results = [{
            "method_name": "fd_nd", "method_family": "fd",
            "n_assets": 5, "seed": 1, "goal_mult": 1.20,
            "goal_hit": np.array([True]),
            "wealth_path": np.array([1.0, 1.1, 1.2]),
            "weight_path": np.array([[0.1, 0.1, 0.1]] * 2),
            "target_wealth": 1.20,
            "train_time_sec": 0.0, "solve_time_sec": 1.0,
            "drawdown_path": np.array([0.0, 0.0, 0.0]),
            "gross_leverage_path": np.array([0.3, 0.3]),
            "net_exposure_path": np.array([0.3, 0.3]),
        }]
        df = build_summary(fake_results)
        assert df["goal_mult"].iloc[0] == pytest.approx(1.20)

    def test_method_order_includes_historical_replay(self):
        from run_experiment import METHOD_ORDER
        assert "nn_historical_replay" in METHOD_ORDER
        assert "nn_historical_replay_long_only" in METHOD_ORDER
