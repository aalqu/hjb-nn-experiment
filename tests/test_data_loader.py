"""
tests/test_data_loader.py
-------------------------
Tests for real_data_loader.py.

Invariants tested
-----------------
Shape consistency  : log_ret.shape == (T, n), mu/sigma/rho shapes match n
Return sanity      : daily log-returns are small (no insane values)
Covariance PD      : omega is positive-definite (Cholesky succeeds)
Correlation diag   : rho diagonal == 1
Excess return      : mkt.mu == mkt.mu_ann - R_FREE_ANN
Date slicing       : start/end filters reduce T without changing n
agg_1d             : portfolio collapse is self-consistent
Portfolio lists    : PORTFOLIOS keys 5/10/20 have correct lengths
Load errors        : n_assets not in (5,10,20) raises ValueError
"""

import numpy as np
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from real_data_loader import load_portfolio, agg_1d, PORTFOLIOS, R_FREE_ANN, MarketData


# ── Shared fixtures ──────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def mkt5():
    return load_portfolio(5)

@pytest.fixture(scope="module")
def mkt10():
    return load_portfolio(10)

@pytest.fixture(scope="module")
def mkt20():
    return load_portfolio(20)


# ══════════════════════════════════════════════════════════════════
# Portfolio composition
# ══════════════════════════════════════════════════════════════════

class TestPortfolioLists:
    def test_keys(self):
        assert set(PORTFOLIOS.keys()) == {5, 10, 20}

    def test_lengths(self):
        for n, tickers in PORTFOLIOS.items():
            assert len(tickers) == n, f"PORTFOLIOS[{n}] has {len(tickers)} tickers"

    def test_no_duplicates(self):
        for n, tickers in PORTFOLIOS.items():
            assert len(set(tickers)) == len(tickers), f"Duplicate tickers in {n}-asset list"


# ══════════════════════════════════════════════════════════════════
# Shape consistency
# ══════════════════════════════════════════════════════════════════

class TestShapes:
    @pytest.mark.parametrize("n", [5, 10, 20])
    def test_shapes(self, n):
        mkt = load_portfolio(n)
        assert mkt.mu.shape     == (n,),     f"mu shape mismatch for n={n}"
        assert mkt.sigma.shape  == (n,),     f"sigma shape mismatch for n={n}"
        assert mkt.rho.shape    == (n, n),   f"rho shape mismatch for n={n}"
        assert mkt.omega.shape  == (n, n),   f"omega shape mismatch for n={n}"
        assert mkt.log_ret.shape[1] == n,    f"log_ret cols mismatch for n={n}"

    def test_tickers_length(self, mkt5):
        assert len(mkt5.tickers) == 5

    def test_dates_length_matches_log_ret(self, mkt5):
        assert len(mkt5.dates) == mkt5.log_ret.shape[0]

    def test_n_attribute(self, mkt5, mkt10, mkt20):
        assert mkt5.n  == 5
        assert mkt10.n == 10
        assert mkt20.n == 20


# ══════════════════════════════════════════════════════════════════
# Return sanity
# ══════════════════════════════════════════════════════════════════

class TestReturnSanity:
    def test_daily_log_returns_bounded(self, mkt5):
        """No daily log return should exceed ±50% (sanity filter)."""
        assert np.all(np.abs(mkt5.log_ret) < 0.5), "Insane daily log return detected"

    def test_no_nan(self, mkt5):
        assert not np.any(np.isnan(mkt5.log_ret)), "NaN in log_ret"

    def test_no_inf(self, mkt5):
        assert not np.any(np.isinf(mkt5.log_ret)), "Inf in log_ret"

    def test_sigma_positive(self, mkt5):
        assert np.all(mkt5.sigma > 0), "Non-positive sigma"

    def test_reasonable_annual_vol(self, mkt5):
        """Equity ETF vol should be between 2% and 80% annualised."""
        assert np.all(mkt5.sigma > 0.02), f"Unreasonably low sigma: {mkt5.sigma.min():.4f}"
        assert np.all(mkt5.sigma < 0.80), f"Unreasonably high sigma: {mkt5.sigma.max():.4f}"

    def test_sufficient_history(self, mkt5):
        """At least 10 years = 2520 trading days."""
        assert mkt5.log_ret.shape[0] >= 2520, f"Only {mkt5.log_ret.shape[0]} days of data"


# ══════════════════════════════════════════════════════════════════
# Covariance / correlation
# ══════════════════════════════════════════════════════════════════

class TestCovarianceMatrix:
    def test_omega_symmetric(self, mkt5):
        diff = np.abs(mkt5.omega - mkt5.omega.T).max()
        assert diff < 1e-12, f"omega not symmetric: max diff = {diff:.2e}"

    def test_omega_positive_definite(self, mkt5):
        """Cholesky should succeed for a PD matrix."""
        try:
            np.linalg.cholesky(mkt5.omega)
        except np.linalg.LinAlgError:
            pytest.fail("omega is not positive definite (Cholesky failed)")

    def test_rho_diagonal_ones(self, mkt5):
        diag = np.diag(mkt5.rho)
        np.testing.assert_allclose(diag, 1.0, atol=1e-10,
                                   err_msg="rho diagonal not all 1")

    def test_rho_bounded(self, mkt5):
        assert np.all(np.abs(mkt5.rho) <= 1.0 + 1e-10), "Correlations out of [-1, 1]"

    def test_omega_consistent_with_rho_sigma(self, mkt5):
        """omega should equal diag(sigma) @ rho @ diag(sigma)."""
        D   = np.diag(mkt5.sigma)
        expected = D @ mkt5.rho @ D
        np.testing.assert_allclose(mkt5.omega, expected, atol=1e-10)


# ══════════════════════════════════════════════════════════════════
# Excess return attribute
# ══════════════════════════════════════════════════════════════════

class TestExcessReturn:
    def test_mu_is_excess(self, mkt5):
        """mkt.mu should equal mkt.mu_ann - R_FREE_ANN."""
        np.testing.assert_allclose(mkt5.mu, mkt5.mu_ann - R_FREE_ANN, atol=1e-12)

    def test_r_attribute(self, mkt5):
        assert abs(mkt5.r - R_FREE_ANN) < 1e-12


# ══════════════════════════════════════════════════════════════════
# Date slicing
# ══════════════════════════════════════════════════════════════════

class TestDateSlicing:
    def test_start_filter_reduces_T(self):
        mkt_full = load_portfolio(5)
        mkt_late = load_portfolio(5, start='2020-01-01')
        assert mkt_late.log_ret.shape[0] < mkt_full.log_ret.shape[0]
        assert mkt_late.n == mkt_full.n

    def test_end_filter_reduces_T(self):
        mkt_full  = load_portfolio(5)
        mkt_early = load_portfolio(5, end='2015-12-31')
        assert mkt_early.log_ret.shape[0] < mkt_full.log_ret.shape[0]

    def test_start_end_window(self):
        mkt = load_portfolio(5, start='2015-01-01', end='2018-12-31')
        # All dates in window
        assert int(str(mkt.dates[0])[:4]) >= 2015
        assert int(str(mkt.dates[-1])[:4]) <= 2018

    def test_n_unchanged_after_slice(self):
        mkt = load_portfolio(10, start='2018-01-01')
        assert mkt.n == 10
        assert mkt.log_ret.shape[1] == 10


# ══════════════════════════════════════════════════════════════════
# agg_1d
# ══════════════════════════════════════════════════════════════════

class TestAgg1d:
    def test_returns_three_values(self, mkt5):
        result = agg_1d(mkt5)
        assert len(result) == 3

    def test_sigma_positive(self, mkt5):
        _, sig, _ = agg_1d(mkt5)
        assert sig > 0

    def test_log_ret_shape(self, mkt5):
        _, _, lr = agg_1d(mkt5)
        assert lr.shape == (mkt5.log_ret.shape[0],)

    def test_equal_weight_default(self, mkt5):
        mu_ew, _, lr_ew = agg_1d(mkt5)
        w = np.ones(mkt5.n) / mkt5.n
        _, _, lr_exp = agg_1d(mkt5, weights=w)
        np.testing.assert_allclose(lr_ew, lr_exp, atol=1e-12)

    def test_custom_weights_normalised(self, mkt5):
        """Custom weights that don't sum to 1 should be auto-normalised."""
        w = np.array([2.0, 1.0, 1.0, 1.0, 1.0])
        mu1, sig1, lr1 = agg_1d(mkt5, weights=w)
        mu2, sig2, lr2 = agg_1d(mkt5, weights=w / w.sum())
        np.testing.assert_allclose(lr1, lr2, atol=1e-12)

    def test_sigma_consistent(self, mkt5):
        """agg sigma should equal sqrt(w^T Omega w) for equal weights."""
        _, sig, _ = agg_1d(mkt5)
        w = np.ones(mkt5.n) / mkt5.n
        expected = float(np.sqrt(w @ mkt5.omega @ w))
        assert abs(sig - expected) < 1e-10


# ══════════════════════════════════════════════════════════════════
# Error handling
# ══════════════════════════════════════════════════════════════════

class TestErrors:
    def test_bad_n_assets(self):
        with pytest.raises(ValueError):
            load_portfolio(7)

    def test_bad_n_assets_negative(self):
        with pytest.raises(ValueError):
            load_portfolio(-1)

    def test_bad_n_assets_string(self):
        with pytest.raises((ValueError, TypeError)):
            load_portfolio("five")
