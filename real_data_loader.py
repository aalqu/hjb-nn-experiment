"""
real_data_loader.py
-------------------
Loads pre-extracted ETF data (real_etf_data.npz) and exposes
ready-to-use parameter sets for the HJB / NN notebooks.

Usage
-----
    from real_data_loader import load_portfolio, PORTFOLIOS

    # Drop-in for synthetic params
    mkt = load_portfolio(1)           # 1-asset (IVV only)
    mkt = load_portfolio(5)          # 5-asset
    mkt = load_portfolio(10)         # 10-asset
    mkt = load_portfolio(20)         # 20-asset

    mkt.mu        # annualised excess returns (shape: n,)
    mkt.sigma     # annualised vols           (shape: n,)
    mkt.rho       # correlation matrix        (shape: n,n)
    mkt.omega     # covariance matrix         (shape: n,n)
    mkt.log_ret   # daily log returns matrix  (shape: T, n)
    mkt.tickers   # list of ticker strings
    mkt.n         # number of assets

Portfolios
----------
  5-asset  (2010-2026): IVV, QQQ, TLT, GLD, VNQ
 10-asset  (2010-2026): IVV, QQQ, IWM, VEA, VNQ, TLT, IEF, LQD, GLD, XLK
 20-asset  (2010-2026): IVV, QQQ, IWM, VEA, VWO, VNQ, TLT, IEF, SHY, TIP,
                        LQD, HYG, GLD, SLV, DBC, USO, XLK, XLF, XLV, UUP
"""

from pathlib import Path
import numpy as np

_NPZ = Path(__file__).parent / 'real_etf_data.npz'

PORTFOLIOS = {
    1:  ['IVV'],
    5:  ['IVV', 'QQQ', 'TLT', 'GLD', 'VNQ'],
    10: ['IVV', 'QQQ', 'IWM', 'VEA', 'VNQ', 'TLT', 'IEF', 'LQD', 'GLD', 'XLK'],
    20: ['IVV', 'QQQ', 'IWM', 'VEA', 'VWO', 'VNQ', 'TLT', 'IEF', 'SHY', 'TIP',
         'LQD', 'HYG', 'GLD', 'SLV', 'DBC', 'USO', 'XLK', 'XLF', 'XLV', 'UUP'],
}

R_FREE_ANN = 0.03   # assumed risk-free rate (3%)


class MarketData:
    """Container for a portfolio's calibrated parameters."""
    def __init__(self, n, tickers, mu_ann, sig_ann, rho, log_ret, dates):
        self.n       = n
        self.tickers = list(tickers)
        self.mu      = mu_ann - R_FREE_ANN   # excess return
        self.mu_ann  = mu_ann                # total return
        self.sigma   = sig_ann
        self.rho     = rho
        self.omega   = np.outer(sig_ann, sig_ann) * rho
        self.log_ret = log_ret               # shape (T, n)
        self.dates   = dates                 # int YYYYMMDD
        self.r       = R_FREE_ANN

    def __repr__(self):
        lines = [f"MarketData({self.n}-asset, {len(self.dates)} days)"]
        for t, m, s in zip(self.tickers, self.mu_ann, self.sigma):
            lines.append(f"  {t:<6}: mu={m*100:+5.1f}%  sigma={s*100:4.1f}%")
        return '\n'.join(lines)


def load_portfolio(n_assets: int, start: str = None, end: str = None) -> MarketData:
    """
    Load calibrated parameters for the n_assets-asset portfolio.

    Parameters
    ----------
    n_assets : 5 | 10 | 20
    start    : 'YYYY-MM-DD' to slice the return history (optional)
    end      : 'YYYY-MM-DD' to slice the return history (optional)

    Returns
    -------
    MarketData object with mu, sigma, rho, omega, log_ret, tickers
    """
    if n_assets not in (1, 5, 10, 20):
        raise ValueError("n_assets must be 1, 5, 10, or 20")
    d = np.load(_NPZ, allow_pickle=True)
    # n=1: load IVV by slicing the first column of the 5-asset dataset
    if n_assets == 1:
        k = '5'
        mu      = d[f'mu{k}'][:1]
        sig     = d[f'sig{k}'][:1]
        rho     = d[f'rho{k}'][:1, :1]
        log_ret = d[f'log_ret{k}'][:, :1]   # (T, 1)
        dates   = d[f'dates{k}']
        tickers = ['IVV']
    else:
        k = str(n_assets)
        mu      = d[f'mu{k}']
        sig     = d[f'sig{k}']
        rho     = d[f'rho{k}']
        log_ret = d[f'log_ret{k}']    # (T, n)
        dates   = d[f'dates{k}']      # int YYYYMMDD
        tickers = list(d[f'tickers{k}'])

    # Optional date slicing
    if start is not None:
        s = int(start.replace('-', ''))
        mask = dates >= s
        log_ret = log_ret[mask]; dates = dates[mask]
    if end is not None:
        e = int(end.replace('-', ''))
        mask = dates <= e
        log_ret = log_ret[mask]; dates = dates[mask]

    return MarketData(n_assets, tickers, mu, sig, rho, log_ret, dates)


def agg_1d(mkt: MarketData, weights=None):
    """
    Collapse an n-asset MarketData to a 1D S&P-like aggregate.
    weights: (n,) array; default = equal-weight.
    Returns (mu_1d, sigma_1d, log_ret_1d).
    """
    n = mkt.n
    w = np.ones(n) / n if weights is None else np.asarray(weights, float)
    w /= w.sum()
    log_ret_1d = mkt.log_ret @ w
    mu_1d      = mkt.mu_ann @ w
    sig_1d     = np.sqrt(w @ mkt.omega @ w)
    return float(mu_1d), float(sig_1d), log_ret_1d


if __name__ == '__main__':
    for n in (1, 5, 10, 20):
        mkt = load_portfolio(n)
        print(mkt)
        print()
