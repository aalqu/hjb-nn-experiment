"""
tests/test_fd_core.py
---------------------
Red-green tests for fd_core.py.

Mathematical invariants tested
-------------------------------
Thomas solver   : exact solution of known tridiagonal system
Terminal BC     : V(w, tau->0) == U(w)  (boundary condition)
Boundary values : V[0] == UB,  V[-1] == UA
Policy bounds   : all Pi in [d, u]
Monotonicity    : V increasing in w for goal-reaching utility
Asymptotic      : asymp_goalreach(goal, tau->0) -> 1
Browne V        : browne_V(w > goal) > 0.5 for reasonable params
pi_browne       : clipped to [d, u], smooth in w
"""

import numpy as np
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fd_core import (
    thomas, normcdf, policy_from_V,
    browne_V, pi_browne,
    asymp_goalreach, asymp_aspiration,
    goal_utility, aspiration_utility,
    fd_solve, make_fd_policy,
)

# ── Shared fixture: fast FD solve for goal-reaching ─────────────────────────
MU, R, SIG = 0.12, 0.03, 0.18
D, U = -5.0, 3.0

@pytest.fixture(scope="module")
def gr_solution():
    """Precompute goal-reaching FD solution once for the whole module."""
    w, V, Pi = fd_solve(
        mu=MU, r=R, sigma=SIG, T=1.0, A=2.0,
        Nw=100, Nt=80, d=D, u=U,
        utility_fn=goal_utility,
        asymptotic_fn=lambda w, tau: asymp_goalreach(w, tau, SIG, D, U),
        UB=0.0, UA=1.0,
    )
    return w, V, Pi

@pytest.fixture(scope="module")
def asp_solution():
    """Precompute aspiration FD solution once."""
    w, V, Pi = fd_solve(
        mu=MU, r=R, sigma=SIG, T=1.0, A=2.5,
        Nw=100, Nt=80, d=D, u=U,
        utility_fn=lambda w: aspiration_utility(w, p=0.5, c1=1.2, R=1.0),
        asymptotic_fn=lambda w, tau: asymp_aspiration(w, tau, SIG, D, U),
        UB=0.0, UA=float(aspiration_utility(np.array([2.5]))[0]),
    )
    return w, V, Pi


# ══════════════════════════════════════════════════════════════════
# Thomas solver
# ══════════════════════════════════════════════════════════════════

class TestThomas:
    def test_identity_system(self):
        """b=1, a=c=0 => x = rhs."""
        n   = 10
        a   = np.zeros(n)
        b   = np.ones(n)
        c   = np.zeros(n)
        rhs = np.arange(1, n+1, dtype=float)
        x   = thomas(a, b, c, rhs)
        np.testing.assert_allclose(x, rhs, atol=1e-12)

    def test_known_solution(self):
        """Tridiagonal 2I - I system: solution known analytically."""
        n   = 5
        b   = 2.0 * np.ones(n)
        a   = -1.0 * np.ones(n)
        c   = -1.0 * np.ones(n)
        rhs = np.array([1.0, 0.0, 0.0, 0.0, 1.0])
        x   = thomas(a, b, c, rhs)
        # Ax = rhs check via explicit reconstruction
        Ax     = b*x
        Ax[1:] += a[1:] * x[:-1]
        Ax[:-1]+= c[:-1] * x[1:]
        np.testing.assert_allclose(Ax, rhs, atol=1e-10)

    def test_random_tridiagonal(self):
        """Random diagonally dominant system — residual should be ~0."""
        rng = np.random.default_rng(0)
        n   = 50
        a   = -rng.uniform(0.1, 0.4, n)
        c   = -rng.uniform(0.1, 0.4, n)
        b   = np.abs(a) + np.abs(c) + rng.uniform(0.5, 1.0, n)   # diag dominant
        rhs = rng.standard_normal(n)
        x   = thomas(a, b, c, rhs)
        Ax     = b*x
        Ax[1:] += a[1:] * x[:-1]
        Ax[:-1]+= c[:-1] * x[1:]
        np.testing.assert_allclose(Ax, rhs, atol=1e-8)


# ══════════════════════════════════════════════════════════════════
# Boundary conditions
# ══════════════════════════════════════════════════════════════════

class TestBoundaryConditions:
    def test_lower_boundary_goalreach(self, gr_solution):
        """V[0] == UB == 0 for goal-reaching."""
        _, V, _ = gr_solution
        assert abs(V[0] - 0.0) < 1e-10, f"V[0]={V[0]}"

    def test_upper_boundary_goalreach(self, gr_solution):
        """V[-1] == UA == 1 for goal-reaching (w=2 >> goal=1)."""
        _, V, _ = gr_solution
        assert abs(V[-1] - 1.0) < 1e-10, f"V[-1]={V[-1]}"

    def test_lower_boundary_aspiration(self, asp_solution):
        """V[0] == 0 for aspiration utility."""
        _, V, _ = asp_solution
        assert abs(V[0] - 0.0) < 1e-10

    def test_upper_boundary_aspiration(self, asp_solution):
        """V[-1] matches aspiration utility evaluated at w=A=2.5."""
        _, V, _ = asp_solution
        UA_expected = float(aspiration_utility(np.array([2.5]))[0])
        assert abs(V[-1] - UA_expected) < 1e-6


# ══════════════════════════════════════════════════════════════════
# Terminal condition  (run a fast solve with Nt=2 and check)
# ══════════════════════════════════════════════════════════════════

class TestTerminalCondition:
    def test_V_near_terminal_goalreach(self):
        """At very small T, V(w) should be close to goal_utility(w)."""
        w, V, _ = fd_solve(
            mu=MU, r=R, sigma=SIG, T=0.02, A=2.0,
            Nw=200, Nt=5, d=D, u=U,
            utility_fn=goal_utility,
            asymptotic_fn=lambda w, tau: asymp_goalreach(w, tau, SIG, D, U),
            UB=0.0, UA=1.0,
        )
        # Away from the discontinuity at w=1, V should be close to U(w)
        mask_below = w < 0.85
        mask_above = w > 1.15
        assert np.all(V[mask_below] < 0.15), "V should be ~0 far below goal"
        assert np.all(V[mask_above] > 0.85), "V should be ~1 far above goal"


# ══════════════════════════════════════════════════════════════════
# Policy bounds
# ══════════════════════════════════════════════════════════════════

class TestPolicyBounds:
    def test_pi_in_constraints_goalreach(self, gr_solution):
        """All optimal policies must lie in [d, u]."""
        _, _, Pi = gr_solution
        assert np.all(Pi >= D - 1e-9), f"Pi min={Pi.min():.4f} < d={D}"
        assert np.all(Pi <= U + 1e-9), f"Pi max={Pi.max():.4f} > u={U}"

    def test_pi_in_constraints_aspiration(self, asp_solution):
        _, _, Pi = asp_solution
        assert np.all(Pi >= D - 1e-9)
        assert np.all(Pi <= U + 1e-9)

    def test_pi_browne_clipped(self):
        """pi_browne must always be clipped to [d, u]."""
        w_vals = np.linspace(0.1, 2.0, 50)
        for w in w_vals:
            pi = pi_browne(w, tau=0.5, mu=MU, r=R, sigma=SIG, goal=1.0, d=D, u=U)
            assert D - 1e-9 <= pi <= U + 1e-9, f"pi={pi} out of [{D},{U}] at w={w}"

    def test_make_fd_policy_clipped(self, gr_solution):
        w_grid, _, Pi_grid = gr_solution
        policy = make_fd_policy(w_grid, Pi_grid, d=D, u=U)
        for w_norm in [0.1, 0.5, 0.9, 1.0, 1.5, 2.0]:
            pi = policy(w_norm, 0.5)
            assert D - 1e-9 <= pi <= U + 1e-9


# ══════════════════════════════════════════════════════════════════
# Monotonicity  (goal-reaching: V should be increasing in w)
# ══════════════════════════════════════════════════════════════════

class TestMonotonicity:
    def test_V_monotone_goalreach(self, gr_solution):
        """V(w) non-decreasing in w for goal-reaching utility."""
        _, V, _ = gr_solution
        diffs = np.diff(V)
        # Allow tiny numerical noise (< 1e-6) but no genuine inversions
        assert np.all(diffs >= -1e-6), (
            f"V not monotone; worst inversion = {diffs.min():.2e}"
        )

    def test_V_range_goalreach(self, gr_solution):
        """V(w) stays in [0, 1] for goal-reaching."""
        _, V, _ = gr_solution
        assert V.min() >= -1e-9,   f"V below 0: {V.min()}"
        assert V.max() <= 1 + 1e-9, f"V above 1: {V.max()}"


# ══════════════════════════════════════════════════════════════════
# Asymptotic functions
# ══════════════════════════════════════════════════════════════════

class TestAsymptotics:
    def test_goalreach_at_goal_small_tau(self):
        """asymp_goalreach(w=goal, tau->0) -> 1."""
        val = asymp_goalreach(np.array([1.0]), tau=1e-6, sigma=SIG, d=D, u=U, goal=1.0)
        assert abs(float(val.item()) - 1.0) < 0.01, f"Got {float(val.item()):.4f}"

    def test_goalreach_far_below(self):
        """asymp_goalreach(w << goal, tau=0.001) should be near 0."""
        val = asymp_goalreach(np.array([0.3]), tau=0.001, sigma=SIG, d=D, u=U, goal=1.0)
        assert float(val.item()) < 0.1, f"Got {float(val.item()):.4f}"

    def test_goalreach_above_goal(self):
        """For w > goal, asymp_goalreach should return exactly 1 (min{0,log>0}=0 -> 2*Phi(0)=1)."""
        val = asymp_goalreach(np.array([1.5]), tau=0.01, sigma=SIG, d=D, u=U, goal=1.0)
        assert abs(float(val.item()) - 1.0) < 1e-9

    def test_aspiration_above_R(self):
        """For w > R, asymp_aspiration returns the upper utility value K_R."""
        p, c1, R = 0.5, 1.2, 1.0
        K_R = c1 * R**p / p
        val = asymp_aspiration(np.array([1.5]), tau=0.01, sigma=SIG, d=D, u=U,
                               p=p, c1=c1, R=R)
        assert abs(float(val.item()) - K_R) < 1e-9


# ══════════════════════════════════════════════════════════════════
# Browne value function
# ══════════════════════════════════════════════════════════════════

class TestBrowneV:
    def test_above_goal_gives_high_prob(self):
        """browne_V(w > goal) > 0.5 for positive drift."""
        eta = MU - R
        v = browne_V(np.array([1.2]), eta=eta, sigma=SIG, tau=0.5, goal=1.0)
        assert float(v.item()) > 0.5

    def test_below_goal_gives_low_prob(self):
        """browne_V(w << goal) < 0.5."""
        eta = MU - R
        v = browne_V(np.array([0.5]), eta=eta, sigma=SIG, tau=0.5, goal=1.0)
        assert float(v.item()) < 0.5

    def test_range(self):
        """browne_V always in [0, 1]."""
        eta  = MU - R
        w    = np.linspace(0.01, 2.0, 100)
        vals = browne_V(w, eta=eta, sigma=SIG, tau=0.5, goal=1.0)
        assert np.all(vals >= 0) and np.all(vals <= 1)


# ══════════════════════════════════════════════════════════════════
# Utility functions
# ══════════════════════════════════════════════════════════════════

class TestUtilities:
    def test_goal_utility_step(self):
        assert float(goal_utility(np.array([0.99])).item()) == 0.0
        assert float(goal_utility(np.array([1.00])).item()) == 1.0
        assert float(goal_utility(np.array([1.01])).item()) == 1.0

    def test_aspiration_continuity_below(self):
        w = np.array([0.5])
        assert abs(float(aspiration_utility(w).item()) - float(0.5**0.5 / 0.5)) < 1e-10

    def test_aspiration_kink_at_R(self):
        """U(R-) < U(R) for c1 > 1 — the aspiration jump."""
        R, p, c1 = 1.0, 0.5, 1.2
        below = float(aspiration_utility(np.array([R - 1e-8]), p=p, c1=c1, R=R).item())
        above = float(aspiration_utility(np.array([R + 1e-8]), p=p, c1=c1, R=R).item())
        assert above > below, "No jump at aspiration level"
