"""
tests/test_backtest_core.py
---------------------------
Tests for backtest_core.py.

Invariants tested
-----------------
simulate_one_year_1asset  : W_path starts at W_start, pi_path in [d,u],
                            shape (days+1,) / (days,), W never drops below floor
simulate_one_year_5asset  : same guarantees for multi-asset
run_backtest_1d           : wealth > 0 at all times, shape matches log_ret length
compute_metrics           : return/vol/sharpe/max_dd/goal_rate well-defined,
                            max_dd <= 0, goal_rate in [0,1]
run_multi_year            : year_end_W length = n_years, goal_hit is boolean,
                            W starts at W0
"""

import numpy as np
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backtest_core import (
    simulate_one_year_1asset,
    simulate_one_year_5asset,
    run_backtest_1d,
    compute_metrics,
    run_multi_year,
)
from fd_core import (
    fd_solve, make_fd_policy,
    goal_utility, aspiration_utility,
    asymp_goalreach, asymp_aspiration,
)

# ── Shared market params ─────────────────────────────────────────────────────

MU, R, SIG = 0.12, 0.03, 0.18
D, U = -5.0, 3.0
DAYS = 63   # one quarter — fast for tests


# ── Simple constant policy ────────────────────────────────────────────────────

def const_policy(w_norm, tau, pi=0.5):
    return pi


# ── Module-level fixtures (work with both pytest and the custom runner) ──────

@pytest.fixture(scope="module")
def fd_gr_policy():
    w, V, Pi = fd_solve(
        mu=MU, r=R, sigma=SIG, T=1.0, A=2.0,
        Nw=80, Nt=40, d=D, u=U,
        utility_fn=goal_utility,
        asymptotic_fn=lambda w, tau: asymp_goalreach(w, tau, SIG, D, U),
        UB=0.0, UA=1.0,
    )
    return make_fd_policy(w, Pi, d=D, u=U)


@pytest.fixture(scope="module")
def fd_as_policy():
    wa, Va, Pia = fd_solve(
        mu=MU, r=R, sigma=SIG, T=1.0, A=2.5,
        Nw=80, Nt=40, d=D, u=U,
        utility_fn=lambda w: aspiration_utility(w, p=0.5, c1=1.2, R=1.0),
        asymptotic_fn=lambda w, tau: asymp_aspiration(w, tau, SIG, D, U),
        UB=0.0,
        UA=float(aspiration_utility(np.array([2.5]))[0]),
    )
    return make_fd_policy(wa, Pia, d=D, u=U)


@pytest.fixture
def sample_returns():
    rng = np.random.default_rng(0)
    return rng.standard_normal(252) * SIG / np.sqrt(252)


@pytest.fixture
def flat_wealth():
    """Wealth that goes exactly up 10% in 1 year."""
    return np.linspace(1.0, 1.10, 253)


@pytest.fixture
def random_wealth():
    rng = np.random.default_rng(7)
    rets = rng.standard_normal(252) * 0.01
    return np.cumprod(np.concatenate([[1.0], 1 + rets]))


# ── 5-asset market params ────────────────────────────────────────────────────

N5 = 3
MU5 = np.array([0.12, 0.10, 0.08])
SIG5 = np.array([0.18, 0.15, 0.12])
RHO5 = np.array([
    [1.0, 0.4, 0.2],
    [0.4, 1.0, 0.3],
    [0.2, 0.3, 1.0],
])
OMEGA5 = np.outer(SIG5, SIG5) * RHO5


def const_nn_policy(W, goal):
    return np.ones(N5) * 0.3


# ══════════════════════════════════════════════════════════════════
# simulate_one_year_1asset
# ══════════════════════════════════════════════════════════════════

class TestSimulateOneYear1Asset:
    def test_shapes(self):
        W_path, pi_path = simulate_one_year_1asset(
            W_start=1.0, goal=1.1,
            policy_fn=const_policy,
            mu=MU, r=R, sigma=SIG,
            days=DAYS, rng=np.random.default_rng(0),
        )
        assert W_path.shape  == (DAYS + 1,)
        assert pi_path.shape == (DAYS,)

    def test_starts_at_W_start(self):
        W_start = 1.234
        W_path, _ = simulate_one_year_1asset(
            W_start=W_start, goal=1.5,
            policy_fn=const_policy,
            mu=MU, r=R, sigma=SIG,
            days=DAYS, rng=np.random.default_rng(1),
        )
        assert abs(W_path[0] - W_start) < 1e-12

    def test_wealth_positive(self):
        W_path, _ = simulate_one_year_1asset(
            W_start=1.0, goal=1.1,
            policy_fn=const_policy,
            mu=MU, r=R, sigma=SIG,
            days=DAYS, rng=np.random.default_rng(2),
        )
        assert np.all(W_path > 0), "Wealth went non-positive"

    def test_pi_in_range(self):
        def clipped_policy(w_norm, tau):
            return float(np.clip(w_norm * 2, D, U))

        _, pi_path = simulate_one_year_1asset(
            W_start=1.0, goal=1.1,
            policy_fn=clipped_policy,
            mu=MU, r=R, sigma=SIG,
            days=DAYS, rng=np.random.default_rng(3),
        )
        assert np.all(pi_path >= D - 1e-9)
        assert np.all(pi_path <= U + 1e-9)

    def test_deterministic_with_seed(self):
        kwargs = dict(W_start=1.0, goal=1.1, policy_fn=const_policy,
                      mu=MU, r=R, sigma=SIG, days=DAYS)
        W1, _ = simulate_one_year_1asset(**kwargs, rng=np.random.default_rng(99))
        W2, _ = simulate_one_year_1asset(**kwargs, rng=np.random.default_rng(99))
        np.testing.assert_array_equal(W1, W2)

    def test_fd_policy_in_range(self, fd_gr_policy):
        _, pi_path = simulate_one_year_1asset(
            W_start=1.0, goal=1.1,
            policy_fn=fd_gr_policy,
            mu=MU, r=R, sigma=SIG,
            days=DAYS, rng=np.random.default_rng(42),
        )
        assert np.all(pi_path >= D - 1e-9)
        assert np.all(pi_path <= U + 1e-9)


# ══════════════════════════════════════════════════════════════════
# simulate_one_year_5asset
# ══════════════════════════════════════════════════════════════════

class TestSimulateOneYear5Asset:
    def test_shapes(self):
        W_path, pi_mat = simulate_one_year_5asset(
            W_start=1.0, goal=1.1,
            nn_policy_fn=const_nn_policy,
            mu_vec=MU5, Omega_mat=OMEGA5, r=R,
            d=D, u=U, target_vol=0.25,
            days=DAYS, rng=np.random.default_rng(0),
        )
        assert W_path.shape  == (DAYS + 1,)
        assert pi_mat.shape  == (DAYS, N5)

    def test_starts_at_W_start(self):
        W_start = 2.5
        W_path, _ = simulate_one_year_5asset(
            W_start=W_start, goal=2.75,
            nn_policy_fn=const_nn_policy,
            mu_vec=MU5, Omega_mat=OMEGA5, r=R,
            d=D, u=U, target_vol=0.25,
            days=DAYS, rng=np.random.default_rng(1),
        )
        assert abs(W_path[0] - W_start) < 1e-12

    def test_wealth_positive(self):
        W_path, _ = simulate_one_year_5asset(
            W_start=1.0, goal=1.1,
            nn_policy_fn=const_nn_policy,
            mu_vec=MU5, Omega_mat=OMEGA5, r=R,
            d=D, u=U, target_vol=0.25,
            days=DAYS, rng=np.random.default_rng(2),
        )
        assert np.all(W_path > 0)

    def test_pi_clipped(self):
        """Weights after variance normalisation must be in [d, u]."""
        _, pi_mat = simulate_one_year_5asset(
            W_start=1.0, goal=1.1,
            nn_policy_fn=lambda W, g: np.ones(N5) * 10.0,  # extreme raw policy
            mu_vec=MU5, Omega_mat=OMEGA5, r=R,
            d=D, u=U, target_vol=0.25,
            days=DAYS, rng=np.random.default_rng(3),
        )
        assert np.all(pi_mat >= D - 1e-9)
        assert np.all(pi_mat <= U + 1e-9)


# ══════════════════════════════════════════════════════════════════
# run_backtest_1d
# ══════════════════════════════════════════════════════════════════

class TestRunBacktest1d:
    def test_shapes(self, sample_returns):
        n = len(sample_returns)
        wealth, pi_path = run_backtest_1d(
            log_returns=sample_returns,
            r_daily=R / 252,
            strategy_fn=lambda w, tau, **kw: 0.5,
            strategy_kwargs={},
        )
        assert wealth.shape  == (n + 1,)
        assert pi_path.shape == (n,)

    def test_starts_at_W0(self, sample_returns):
        wealth, _ = run_backtest_1d(
            log_returns=sample_returns,
            r_daily=R / 252,
            strategy_fn=lambda w, tau, **kw: 0.5,
            strategy_kwargs={},
            W0=2.0,
        )
        assert abs(wealth[0] - 2.0) < 1e-12

    def test_wealth_positive(self, sample_returns):
        wealth, _ = run_backtest_1d(
            log_returns=sample_returns,
            r_daily=R / 252,
            strategy_fn=lambda w, tau, **kw: 0.5,
            strategy_kwargs={},
        )
        assert np.all(wealth > 0)

    def test_pi_clipped(self, sample_returns):
        """Strategy output is clipped to [d, u]."""
        _, pi_path = run_backtest_1d(
            log_returns=sample_returns,
            r_daily=R / 252,
            strategy_fn=lambda w, tau, **kw: 99.0,  # out-of-range
            strategy_kwargs={},
            d=D, u=U,
        )
        assert np.all(pi_path <= U + 1e-9)


# ══════════════════════════════════════════════════════════════════
# compute_metrics
# ══════════════════════════════════════════════════════════════════

class TestComputeMetrics:
    def test_keys_present(self, random_wealth):
        m = compute_metrics(random_wealth)
        for key in ('ann_ret', 'ann_vol', 'sharpe', 'max_dd', 'goal_rate'):
            assert key in m, f"Missing key: {key}"

    def test_max_dd_nonpositive(self, random_wealth):
        m = compute_metrics(random_wealth)
        assert m['max_dd'] <= 0, f"max_dd should be <= 0, got {m['max_dd']}"

    def test_goal_rate_in_unit_interval(self, random_wealth):
        m = compute_metrics(random_wealth)
        assert 0.0 <= m['goal_rate'] <= 1.0

    def test_ann_vol_positive(self, random_wealth):
        m = compute_metrics(random_wealth)
        assert m['ann_vol'] > 0

    def test_flat_max_dd_zero(self, flat_wealth):
        """Monotone increasing wealth has zero drawdown."""
        m = compute_metrics(flat_wealth)
        assert abs(m['max_dd']) < 1e-6, f"max_dd = {m['max_dd']:.2e}"

    def test_goal_rate_one_for_always_rising(self, flat_wealth):
        """Wealth that rises monotonically always hits +10% goal."""
        m = compute_metrics(flat_wealth, target_return=1.05)  # < actual growth
        assert m['goal_rate'] == 1.0


# ══════════════════════════════════════════════════════════════════
# run_multi_year
# ══════════════════════════════════════════════════════════════════

class TestRunMultiYear:
    @pytest.mark.parametrize("strategy", ['fd_goalreach', 'fd_aspiration', 'browne', 'kelly'])
    def test_shapes_1asset(self, strategy, fd_gr_policy, fd_as_policy):
        n_years = 3
        res = run_multi_year(
            n_years=n_years, W0=1.0, annual_target=1.10, strategy=strategy,
            fd_gr_policy=fd_gr_policy, fd_as_policy=fd_as_policy,
            mu=MU, r=R, sigma=SIG,
            d=D, u=U, days=63, seed=0,
        )
        assert len(res['year_end_W']) == n_years
        assert len(res['goals'])      == n_years
        assert len(res['goal_hit'])   == n_years
        assert len(res['W'])          == n_years * 63 + 1

    def test_starts_at_W0(self, fd_gr_policy):
        res = run_multi_year(
            n_years=2, W0=1.5, annual_target=1.10, strategy='fd_goalreach',
            fd_gr_policy=fd_gr_policy,
            mu=MU, r=R, sigma=SIG, d=D, u=U, days=63, seed=1,
        )
        assert abs(res['W'][0] - 1.5) < 1e-12

    def test_goal_hit_is_bool(self, fd_gr_policy):
        res = run_multi_year(
            n_years=2, W0=1.0, annual_target=1.10, strategy='kelly',
            fd_gr_policy=fd_gr_policy,
            mu=MU, r=R, sigma=SIG, d=D, u=U, days=63, seed=2,
        )
        assert res['goal_hit'].dtype == bool

    def test_wealth_always_positive(self, fd_gr_policy):
        res = run_multi_year(
            n_years=3, W0=1.0, annual_target=1.10, strategy='browne',
            fd_gr_policy=fd_gr_policy,
            mu=MU, r=R, sigma=SIG, d=D, u=U, days=63, seed=3,
        )
        assert np.all(res['W'] > 0)

    def test_unknown_strategy_raises(self, fd_gr_policy):
        with pytest.raises(ValueError):
            run_multi_year(
                n_years=1, W0=1.0, annual_target=1.10, strategy='unicorn',
                fd_gr_policy=fd_gr_policy,
                mu=MU, r=R, sigma=SIG, d=D, u=U, days=63, seed=0,
            )
