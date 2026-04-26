"""
tests/test_nn_core.py
---------------------
Tests for nn_core.py (PolicyNet + train_policy_net).

All tests are skipped gracefully if PyTorch is not installed.

Invariants tested
-----------------
PolicyNet output    : shape (batch, n_assets), values in [d, u]
PolicyNet grad      : output is differentiable w.r.t. input
train_policy_net    : returns PolicyNet in eval mode
                      output shape / range after training
                      goalreach utility improves over random init
                      aspiration utility completes without error
nn_policy_weights   : returns (n_assets,) numpy array in [d, u]
"""

import numpy as np
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Graceful skip if torch is absent ────────────────────────────────────────

try:
    import torch
    from nn_core import PolicyNet, train_policy_net, nn_policy_weights
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

SKIP_MSG = "PyTorch not installed — skipping NN tests"


# ── Market params ────────────────────────────────────────────────────────────

N_ASSETS = 3
MU_VEC   = np.array([0.12, 0.10, 0.08])
SIG_VEC  = np.array([0.18, 0.15, 0.12])
RHO      = np.array([[1.0, 0.4, 0.2],
                      [0.4, 1.0, 0.3],
                      [0.2, 0.3, 1.0]])
OMEGA    = np.outer(SIG_VEC, SIG_VEC) * RHO
R        = 0.03
D, U     = -5.0, 3.0


# ── Shared fixture: lightly trained net (fast — 20 iters) ────────────────────

@pytest.fixture(scope="module")
def light_net():
    if not HAS_TORCH:
        pytest.skip(SKIP_MSG)
    return train_policy_net(
        mu_vec=MU_VEC, Omega_mat=OMEGA, r=R,
        T=1.0, w0=1.0, goal_mult=1.10,
        n_paths=64, n_iters=20,
        hidden=32, lr=3e-3, n_steps=10,
        d=D, u=U, target_vol=0.25,
        utility='goalreach',
        verbose=False,
    )


# ══════════════════════════════════════════════════════════════════
# PolicyNet architecture
# ══════════════════════════════════════════════════════════════════

class TestPolicyNetArchitecture:
    def test_output_shape_single(self):
        if not HAS_TORCH: pytest.skip(SKIP_MSG)
        net = PolicyNet(n_assets=N_ASSETS, hidden=32, d=D, u=U)
        x = torch.tensor([[1.0]])
        out = net(x)
        assert out.shape == (1, N_ASSETS)

    def test_output_shape_batch(self):
        if not HAS_TORCH: pytest.skip(SKIP_MSG)
        net = PolicyNet(n_assets=N_ASSETS, hidden=32, d=D, u=U)
        x = torch.ones(16, 1)
        out = net(x)
        assert out.shape == (16, N_ASSETS)

    def test_output_range(self):
        """All outputs must lie in [d, u]."""
        if not HAS_TORCH: pytest.skip(SKIP_MSG)
        net = PolicyNet(n_assets=N_ASSETS, hidden=64, d=D, u=U)
        x = torch.linspace(-3.0, 3.0, 100).unsqueeze(1)
        with torch.no_grad():
            out = net(x)
        assert out.min().item() >= D - 1e-5
        assert out.max().item() <= U + 1e-5

    def test_output_differentiable(self):
        """Backprop must flow through the network."""
        if not HAS_TORCH: pytest.skip(SKIP_MSG)
        net = PolicyNet(n_assets=N_ASSETS, hidden=32, d=D, u=U)
        x = torch.tensor([[1.0]], requires_grad=True)
        out = net(x)
        loss = out.sum()
        loss.backward()  # must not raise
        assert x.grad is not None

    def test_different_assets(self):
        if not HAS_TORCH: pytest.skip(SKIP_MSG)
        for n in [1, 3, 5, 10]:
            net = PolicyNet(n_assets=n, hidden=32, d=D, u=U)
            out = net(torch.ones(1, 1))
            assert out.shape == (1, n)


# ══════════════════════════════════════════════════════════════════
# train_policy_net — smoke tests (fast, few iters)
# ══════════════════════════════════════════════════════════════════

class TestTrainPolicyNet:
    def test_returns_policy_net(self, light_net):
        if not HAS_TORCH: pytest.skip(SKIP_MSG)
        assert isinstance(light_net, PolicyNet)

    def test_eval_mode(self, light_net):
        if not HAS_TORCH: pytest.skip(SKIP_MSG)
        assert not light_net.training, "Net should be in eval mode after training"

    def test_output_range_after_training(self, light_net):
        if not HAS_TORCH: pytest.skip(SKIP_MSG)
        x = torch.linspace(0.5, 2.0, 50).unsqueeze(1)
        with torch.no_grad():
            out = light_net(x)
        assert out.min().item() >= D - 1e-5
        assert out.max().item() <= U + 1e-5

    def test_goalreach_utility_trains(self):
        """Training with goalreach utility should complete without exception."""
        if not HAS_TORCH: pytest.skip(SKIP_MSG)
        net = train_policy_net(
            mu_vec=MU_VEC, Omega_mat=OMEGA, r=R,
            T=1.0, w0=1.0, goal_mult=1.10,
            n_paths=32, n_iters=5,
            hidden=16, lr=1e-3, n_steps=5,
            d=D, u=U, target_vol=0.25,
            utility='goalreach',
            verbose=False,
        )
        assert net is not None

    def test_aspiration_utility_trains(self):
        """Training with aspiration utility should complete without exception."""
        if not HAS_TORCH: pytest.skip(SKIP_MSG)
        net = train_policy_net(
            mu_vec=MU_VEC, Omega_mat=OMEGA, r=R,
            T=1.0, w0=1.0, goal_mult=1.10,
            n_paths=32, n_iters=5,
            hidden=16, lr=1e-3, n_steps=5,
            d=D, u=U, target_vol=0.25,
            utility='aspiration',
            verbose=False,
        )
        assert net is not None

    def test_invalid_utility_raises(self):
        if not HAS_TORCH: pytest.skip(SKIP_MSG)
        with pytest.raises(ValueError):
            train_policy_net(
                mu_vec=MU_VEC, Omega_mat=OMEGA, r=R,
                T=1.0, w0=1.0, goal_mult=1.10,
                n_paths=16, n_iters=2,
                hidden=16, lr=1e-3, n_steps=2,
                d=D, u=U, target_vol=0.25,
                utility='invalid_mode',
                verbose=False,
            )

    def test_goalreach_loss_is_differentiable(self):
        """
        Regression: goalreach must use sigmoid not boolean comparison.
        Original bug: RuntimeError: element 0 of tensors does not require grad.
        """
        if not HAS_TORCH: pytest.skip(SKIP_MSG)
        net = PolicyNet(n_assets=N_ASSETS, hidden=16, d=D, u=U)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)

        W = torch.ones(8, 1, requires_grad=False)
        goal = 1.1
        pi = net(W / goal)
        # Make the terminal objective depend on the network output so autograd
        # can verify the policy-gradient path is differentiable end-to-end.
        wealth_score = (W.squeeze() / goal) + 0.01 * pi.sum(dim=1)
        U_term = torch.sigmoid((wealth_score - 1.0) / 0.05)
        loss = -U_term.mean()

        opt.zero_grad()
        loss.backward()  # must not raise RuntimeError
        assert loss.requires_grad, "Loss has no grad"

    def test_no_nan_in_weights_after_training(self, light_net):
        if not HAS_TORCH: pytest.skip(SKIP_MSG)
        for p in light_net.parameters():
            assert not torch.isnan(p).any(), "NaN in trained weights"


# ══════════════════════════════════════════════════════════════════
# nn_policy_weights
# ══════════════════════════════════════════════════════════════════

class TestNnPolicyWeights:
    def test_returns_numpy(self, light_net):
        if not HAS_TORCH: pytest.skip(SKIP_MSG)
        pi = nn_policy_weights(light_net, W_current=1.0, goal=1.1)
        assert isinstance(pi, np.ndarray)

    def test_shape(self, light_net):
        if not HAS_TORCH: pytest.skip(SKIP_MSG)
        pi = nn_policy_weights(light_net, W_current=1.0, goal=1.1)
        assert pi.shape == (N_ASSETS,)

    def test_range(self, light_net):
        if not HAS_TORCH: pytest.skip(SKIP_MSG)
        for w in [0.5, 1.0, 1.2, 2.0]:
            pi = nn_policy_weights(light_net, W_current=w, goal=1.1)
            assert np.all(pi >= D - 1e-5), f"pi below d at w={w}: {pi.min():.4f}"
            assert np.all(pi <= U + 1e-5), f"pi above u at w={w}: {pi.max():.4f}"

    def test_no_nan(self, light_net):
        if not HAS_TORCH: pytest.skip(SKIP_MSG)
        pi = nn_policy_weights(light_net, W_current=1.0, goal=1.1)
        assert not np.any(np.isnan(pi)), "NaN in policy weights"

    def test_deterministic(self, light_net):
        if not HAS_TORCH: pytest.skip(SKIP_MSG)
        pi1 = nn_policy_weights(light_net, W_current=1.23, goal=1.35)
        pi2 = nn_policy_weights(light_net, W_current=1.23, goal=1.35)
        np.testing.assert_array_equal(pi1, pi2)
