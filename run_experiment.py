"""
run_experiment.py
-----------------
Full comparison: FD HJB solver vs all neural network architectures,
across multiple asset counts.

Outputs (all written to RESULTS_DIR)
─────────────────────────────────────
results/
├── summary.csv              ← one row per method × n_assets × seed
├── goal_probability.png     ← P(W_T ≥ goal) by method and asset count
├── terminal_wealth.png      ← distribution of W_T by method
├── train_time.png           ← solve / train time by method
├── convergence/
│   └── <arch>_n<k>.png     ← E[U] training curve per architecture
├── weights/
│   └── <method>_n<k>_<ticker>.png  ← weight time-series per ticker
└── weight_dist/
    └── <method>_n<k>.png   ← box-plot of weights across assets

Usage
─────
    cd "/path/to/Claude Code"
    python run_experiment.py              # full run
    python run_experiment.py --quick      # fast sanity check (1 seed, tiny NN)
    python run_experiment.py --no-nn      # FD + baselines only
"""

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "comparisons"))

RESULTS_DIR = ROOT / "results" / "experiment"

# ── Configuration ─────────────────────────────────────────────────────────────

def make_config(quick=False, no_nn=False, n_assets_list=None, seeds=None,
                nn_archs=None):
    """Build a BenchmarkConfig with sensible defaults."""
    from comparisons.core.config import BenchmarkConfig

    return BenchmarkConfig(
        n_assets_list  = n_assets_list or ([1, 5] if quick else [1, 5, 10, 20]),
        random_seeds   = seeds         or ([1]  if quick else [1, 2, 3]),
        start_date     = "2015-01-01",
        end_date       = "2024-12-31",
        initial_wealth_levels = [1.0],
        target_multiplier     = 1.10,

        # FD solver
        fd_wealth_max = 2.5,
        fd_nw         = 120,
        fd_nt         = 80,

        # NN training
        include_nn        = not no_nn,
        nn_architectures  = nn_archs or [
            "nn_mlp_small",
            "nn_mlp_deep",
            "nn_policy_net",
            "nn_ste_goalreach",
            "nn_digital_hedge",
            "nn_policy_long_only",
            "nn_ste_long_only",
            "nn_digital_hedge_long_only",
            "nn_historical_replay",
            "nn_historical_replay_long_only",
            "deep_bsde",
            "pinn",
            "actor_critic",
            "lstm",
            "transformer",
        ],
        nn_paths          = 64  if quick else 512,
        nn_iters          = 10  if quick else 200,
        nn_steps          = 16  if quick else 40,
        nn_pretrain_iters = 0   if quick else 100,
        nn_antithetic     = True,
        nn_p_curriculum   = 0.30,
        nn_patience       = 30  if quick else 60,
        nn_horizon_years  = 1.0,

        # Leverage
        weight_lower_bound = -5.0,
        weight_upper_bound =  3.0,
        max_long_leverage  =  3.0,
        max_short_leverage =  5.0,

        # Goal multiplier sweep
        goal_multipliers = [1.05, 1.10, 1.15, 1.20, 1.30],

        results_dir = RESULTS_DIR,
    )


# ── Data & evaluation imports ─────────────────────────────────────────────────

def _imports():
    from real_data_loader import load_portfolio
    from comparisons.core.evaluation import (
        evaluate_fd_benchmark,
        evaluate_nn_portfolio,
        evaluate_merton_benchmark,
        apply_leverage_constraint,
    )
    return load_portfolio, evaluate_fd_benchmark, evaluate_nn_portfolio, \
           evaluate_merton_benchmark, apply_leverage_constraint


# ── Single-run orchestrator ───────────────────────────────────────────────────

def _ckpt_path(results_dir, method, n, seed, gm):
    """Return the per-run checkpoint path (used for --resume)."""
    safe = method.replace("/", "_")
    return results_dir / "checkpoints" / f"{safe}_n{n}_s{seed}_gm{gm:.2f}.npz"


def _save_ckpt(results_dir, res, n, seed, gm):
    """Persist a lightweight checkpoint so --resume can skip completed runs."""
    path = _ckpt_path(results_dir, res["method_name"], n, seed, gm)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        method_name   = np.array(res["method_name"]),
        goal_hit      = np.array(res["goal_hit"]),
        wealth_path   = np.array(res.get("wealth_path", [])),
        train_time    = np.array(res.get("train_time_sec", 0.0)),
        solve_time    = np.array(res.get("solve_time_sec", 0.0)),
    )


def run_all(config, device=None, resume=False, compile_model=False):
    """
    Run every method for every (n_assets, seed) combination.

    Parameters
    ----------
    config        : BenchmarkConfig
    device        : str | None  — PyTorch device string ('cuda', 'cpu', …)
    resume        : bool        — skip runs that already have a checkpoint
    compile_model : bool        — pass compile=True to eval_nn (torch.compile)

    Returns
    -------
    results  : list[dict]  — one result dict per (method, n_assets, seed)
    histories: dict        — {key: {"train", "val", "val_iters", "test_u"}}
                             keyed by f"{arch}_n{n}_s{seed}_gm{gm:.2f}"
    """
    load_portfolio, eval_fd, eval_nn, eval_merton, _ = _imports()

    results   = []
    histories = {}    # train/val/test curves per arch×n×seed

    for n in config.n_assets_list:
        print(f"\n{'='*60}")
        print(f"  n_assets = {n}")
        print(f"{'='*60}")

        mkt = load_portfolio(n, start=config.start_date, end=config.end_date)
        print(f"  Tickers : {list(mkt.tickers)}")
        print(f"  Returns : {[f'{m:.1%}' for m in mkt.mu_ann]}")
        print(f"  Dates   : {mkt.dates[0]} → {mkt.dates[-1]}  ({len(mkt.dates)} days)")

        goal_multipliers = getattr(config, 'goal_multipliers', [config.target_multiplier])
        import copy

        # ── FD and Merton are deterministic (no seed dependency) ─────────────
        # Solve once per (n_assets, goal_mult) and reuse the result for every
        # seed.  This eliminates len(seeds) − 1 redundant FD solves per goal_mult
        # (50 solves → 10 for 5 seeds × 5 goal_mults).
        #
        # The result dict is shallow-copied per seed with only the "seed" field
        # updated so that downstream groupby(["method","n_assets","seed"]) works.
        fd_cache      = {}   # gm → result dict (seed-agnostic)
        merton_cache  = {}   # gm → result dict (seed-agnostic)

        for gm in goal_multipliers:
            cfg_gm = copy.copy(config)
            cfg_gm.target_multiplier = gm

            if config.include_fd_benchmark:
                _ck = _ckpt_path(config.results_dir, "fd_nd", n, 0, gm)
                if resume and _ck.exists():
                    print(f"  [FD n={n} goal={gm:.2f}] skipped (checkpoint exists)")
                else:
                    print(f"  [FD n={n} goal={gm:.2f}] solving HJB ...", end=" ", flush=True)
                    t0  = time.perf_counter()
                    res = eval_fd(mkt, cfg_gm, initial_wealth=1.0, seed=0)
                    print(f"done ({time.perf_counter()-t0:.1f}s)  "
                          f"goal_hit={res['goal_hit'][0]}")
                    fd_cache[gm] = {**res, "n_assets": n, "goal_mult": gm}

            if config.include_merton_benchmark:
                _ck = _ckpt_path(config.results_dir, "merton", n, 0, gm)
                if resume and _ck.exists():
                    print(f"  [Merton n={n} goal={gm:.2f}] skipped (checkpoint exists)")
                else:
                    print(f"  [Merton n={n} goal={gm:.2f}] ...", end=" ", flush=True)
                    res = eval_merton(mkt, cfg_gm, initial_wealth=1.0, seed=0)
                    print(f"done  goal_hit={res['goal_hit'][0]}")
                    merton_cache[gm] = {**res, "n_assets": n, "goal_mult": gm}

        for seed in config.random_seeds:
            print(f"\n  seed={seed}")

            for gm in goal_multipliers:
                cfg_gm = copy.copy(config)
                cfg_gm.target_multiplier = gm

                # Stamp cached FD/Merton results with this seed (shallow copy)
                if gm in fd_cache:
                    res_fd = {**fd_cache[gm], "seed": seed}
                    results.append(res_fd)
                    _save_ckpt(config.results_dir, res_fd, n, seed, gm)

                if gm in merton_cache:
                    res_m = {**merton_cache[gm], "seed": seed}
                    results.append(res_m)
                    _save_ckpt(config.results_dir, res_m, n, seed, gm)

                if config.include_nn:
                    for arch in config.nn_architectures:
                        _ck = _ckpt_path(config.results_dir, arch, n, seed, gm)
                        if resume and _ck.exists():
                            print(f"    [{arch} goal={gm:.2f}] skipped (checkpoint exists)")
                            continue
                        print(f"    [{arch} goal={gm:.2f}] training ...", end=" ", flush=True)
                        t0 = time.perf_counter()
                        try:
                            # Pass device and compile_model through if eval_nn accepts them
                            _nn_kwargs = dict(
                                architecture_name=arch,
                                initial_wealth=1.0,
                                seed=seed,
                            )
                            if device is not None:
                                _nn_kwargs["device"] = device
                            if compile_model:
                                _nn_kwargs["compile_model"] = True
                            res = eval_nn(mkt, cfg_gm, **_nn_kwargs)
                            elapsed = time.perf_counter() - t0
                            test_u  = res.get("test_u", float("nan"))
                            print(f"done ({elapsed:.1f}s)  "
                                  f"goal_hit={res['goal_hit'][0]}  "
                                  f"test_E[U]={test_u:.4f}")
                            res = {**res, "n_assets": n, "seed": seed, "goal_mult": gm}
                            results.append(res)
                            _save_ckpt(config.results_dir, res, n, seed, gm)

                            key = f"{arch}_n{n}_s{seed}_gm{gm:.2f}"
                            histories[key] = {
                                "arch"      : arch,
                                "n_assets"  : n,
                                "seed"      : seed,
                                "goal_mult" : gm,
                                "train"     : res.get("train_history", []),
                                "val"       : res.get("val_history",   []),
                                "val_iters" : res.get("val_iters",     []),
                                "test_u"    : test_u,
                            }

                        except Exception as exc:
                            import traceback
                            print(f"FAILED: {exc}")
                            if not resume:
                                traceback.print_exc()

    return results, histories


# ── Summary table ─────────────────────────────────────────────────────────────

def build_summary(results):
    """
    Build a tidy DataFrame with one row per (method, n_assets, seed).
    Columns: method, n_assets, seed, goal_probability, mean_terminal_wealth,
             p05_terminal_wealth, max_drawdown, mean_gross_leverage,
             mean_turnover, train_time_sec, solve_time_sec.
    """
    rows = []
    for r in results:
        wp      = np.asarray(r["wealth_path"])
        wt_path = np.asarray(r["weight_path"])
        goal    = r.get("target_wealth", 1.1)

        rows.append({
            "method"              : r["method_name"],
            "family"              : r.get("method_family", "?"),
            "n_assets"            : r.get("n_assets", 0),
            "seed"                : r.get("seed", 0),
            "goal_mult"           : float(r.get("goal_mult", 1.10)),
            "goal_probability"    : float(r["goal_hit"][0]),
            "terminal_wealth"     : float(wp[-1]),
            "shortfall"           : float(max(goal - wp[-1], 0.0)),
            "max_drawdown"        : float(np.min(r.get("drawdown_path", [0]))),
            "mean_gross_leverage" : float(np.mean(r.get("gross_leverage_path", [0]))),
            "mean_net_exposure"   : float(np.mean(r.get("net_exposure_path", [0]))),
            "wealth_vol"          : float(np.std(np.diff(np.log(wp + 1e-8))) * np.sqrt(252)),
            "train_time_sec"      : float(r.get("train_time_sec", 0.0)),
            "solve_time_sec"      : float(r.get("solve_time_sec", 0.0)),
            "nn_param_count"      : int(r.get("nn_param_count", 0)),
        })
    return pd.DataFrame(rows)


# ── Plots ──────────────────────────────────────────────────────────────────────

PALETTE = {
    "fd"       : "#1f77b4",
    "nn"       : "#ff7f0e",
    "baseline" : "#2ca02c",
}

METHOD_ORDER = [
    "fd_1d_proxy", "fd_nd", "fd_merton_multi",
    "nn_mlp_small", "nn_mlp_deep", "nn_policy_net", "nn_ste_goalreach",
    "nn_digital_hedge",
    "nn_policy_long_only", "nn_ste_long_only", "nn_digital_hedge_long_only",
    "nn_historical_replay", "nn_historical_replay_long_only",
    "deep_bsde", "pinn", "actor_critic", "lstm", "transformer",
    "equal_weight", "merton", "market_cap",
]


def _sorted_methods(df):
    present = df["method"].unique().tolist()
    return [m for m in METHOD_ORDER if m in present] + \
           [m for m in present if m not in METHOD_ORDER]


def plot_goal_vs_multiplier(df, out_dir):
    """
    Line chart: P(goal) vs goal multiplier for each method, one panel per n_assets.
    Shows how ambition (the difficulty of the target) affects each strategy.
    """
    if "goal_mult" not in df.columns:
        return
    mults = sorted(df["goal_mult"].unique())
    if len(mults) <= 1:
        return

    n_list = sorted(df["n_assets"].unique())
    fig, axes = plt.subplots(1, len(n_list), figsize=(7 * len(n_list), 5), sharey=True)
    if len(n_list) == 1:
        axes = [axes]

    # Highlight a small set of key methods to keep the plot readable
    KEY_METHODS = ["fd_nd", "nn_policy_net", "nn_ste_goalreach", "nn_digital_hedge",
                   "nn_historical_replay", "nn_policy_long_only", "equal_weight", "merton"]

    for ax, n in zip(axes, n_list):
        sub = df[df["n_assets"] == n]
        present = [m for m in KEY_METHODS if m in sub["method"].unique()]
        cmap = plt.get_cmap("tab10")
        for i, method in enumerate(present):
            ms = sub[sub["method"] == method]
            gp = ms.groupby("goal_mult")["goal_probability"]
            means = gp.mean().reindex(mults)
            stds  = gp.std().reindex(mults).fillna(0)
            color = cmap(i % 10)
            ax.plot(mults, means.values, marker='o', lw=1.8, label=method, color=color)
            ax.fill_between(mults,
                            (means - stds).values.clip(0, 1),
                            (means + stds).values.clip(0, 1),
                            alpha=0.12, color=color)
        ax.set_xlabel("Goal multiplier (target / initial wealth)")
        ax.set_ylabel("P(reach goal)" if n == n_list[0] else "")
        ax.set_title(f"n_assets = {n}")
        ax.set_ylim(0, 1.05)
        ax.set_xticks(mults)
        ax.set_xticklabels([f"{m:.2f}×" for m in mults])
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, ncol=2)

    fig.suptitle("P(goal) vs Goal Difficulty", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = out_dir / "goal_vs_multiplier.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")


def plot_goal_probability(df, out_dir):
    """
    Bar chart: P(goal) by method, one panel per n_assets.
    Aggregates across seeds with error bars.
    Only uses the default goal_mult=1.10 rows (or all rows if no sweep).
    """
    # Filter to base goal_mult if a sweep was run
    if "goal_mult" in df.columns and len(df["goal_mult"].unique()) > 1:
        df = df[df["goal_mult"] == 1.10]
    n_list = sorted(df["n_assets"].unique())
    fig, axes = plt.subplots(1, len(n_list), figsize=(6 * len(n_list), 5),
                             sharey=True)
    if len(n_list) == 1:
        axes = [axes]

    for ax, n in zip(axes, n_list):
        sub = df[df["n_assets"] == n]
        methods = _sorted_methods(sub)
        means = [sub[sub["method"] == m]["goal_probability"].mean() for m in methods]
        errs  = [sub[sub["method"] == m]["goal_probability"].std()  for m in methods]
        colors = [PALETTE.get(sub[sub["method"] == m]["family"].iloc[0]
                              if len(sub[sub["method"] == m]) else "baseline",
                              "#9467bd") for m in methods]

        bars = ax.bar(range(len(methods)), means, yerr=errs, color=colors,
                      alpha=0.85, capsize=4, ecolor="grey")
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=40, ha="right", fontsize=8)
        ax.set_title(f"n_assets = {n}", fontsize=11)
        ax.set_ylabel("P(W_T ≥ goal)" if n == n_list[0] else "")
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.axhline(0.5, color="grey", lw=0.8, ls="--")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Goal Probability by Method and Asset Count", fontsize=13)
    fig.tight_layout()
    path = out_dir / "goal_probability.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")


def plot_terminal_wealth(df, out_dir):
    """Box plot of terminal wealth by method, one panel per n_assets."""
    n_list = sorted(df["n_assets"].unique())
    fig, axes = plt.subplots(1, len(n_list), figsize=(6 * len(n_list), 5),
                             sharey=False)
    if len(n_list) == 1:
        axes = [axes]

    for ax, n in zip(axes, n_list):
        sub = df[df["n_assets"] == n]
        methods = _sorted_methods(sub)
        data = [sub[sub["method"] == m]["terminal_wealth"].values for m in methods]

        ax.boxplot(data, tick_labels=methods, patch_artist=True,
                   medianprops=dict(color="black", lw=2))
        ax.set_xticklabels(methods, rotation=40, ha="right", fontsize=8)
        ax.set_title(f"n_assets = {n}")
        ax.set_ylabel("Terminal wealth" if n == n_list[0] else "")
        ax.axhline(1.10, color="red", lw=1.0, ls="--", label="goal (1.10)")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Terminal Wealth Distribution", fontsize=13)
    fig.tight_layout()
    path = out_dir / "terminal_wealth.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")


def plot_train_time(df, out_dir):
    """Solve/train time vs method, stacked bars."""
    sub = df.groupby("method").agg(
        train=("train_time_sec", "mean"),
        solve=("solve_time_sec", "mean"),
        family=("family", "first"),
    ).reset_index()
    sub = sub.sort_values("train", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(sub))
    ax.bar(x, sub["solve"], label="FD solve", color="#1f77b4", alpha=0.85)
    ax.bar(x, sub["train"], bottom=sub["solve"], label="NN train",
           color="#ff7f0e", alpha=0.85)
    ax.set_xticks(list(x))
    ax.set_xticklabels(sub["method"].tolist(), rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("Wall-clock seconds")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Training / Solve Time by Method", fontsize=13)
    fig.tight_layout()
    path = out_dir / "train_time.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")


def plot_convergence(conv, out_dir):
    """E[U] training curves for each NN architecture × n_assets."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, history in conv.items():
        if not history:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(history, lw=1.5, color="#ff7f0e")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("E[U]  (goal-reaching)")
        ax.set_title(f"Convergence: {key}")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        path = out_dir / f"{key}.png"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
    if conv:
        print(f"  → {out_dir}/ ({len(conv)} curves)")


def plot_weight_paths(results, out_dir):
    """
    Per-ticker weight time-series for each method × n_assets.
    One file per (method, n_assets): all tickers overlaid.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for r in results:
        wt_path = np.asarray(r.get("weight_path", []))
        tickers = np.asarray(r.get("tickers", []))
        if wt_path.ndim != 2 or wt_path.shape[1] == 0:
            continue

        n      = wt_path.shape[1]
        method = r["method_name"]
        n_ast  = r.get("n_assets", n)
        seed   = r.get("seed", 0)

        fig, ax = plt.subplots(figsize=(12, 4))
        for i in range(n):
            label = tickers[i] if i < len(tickers) else f"asset_{i}"
            ax.plot(wt_path[:, i], lw=0.9, label=label, alpha=0.8)

        ax.axhline(0, color="grey", lw=0.6, ls="--")
        ax.set_xlabel("Trading day")
        ax.set_ylabel("Portfolio weight")
        ax.set_title(f"{method}  |  n={n_ast}  |  seed={seed}")
        ax.legend(ncol=min(n, 5), fontsize=7, loc="upper right")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        path = out_dir / f"{method}_n{n_ast}_s{seed}.png"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
    print(f"  → {out_dir}/")


def plot_weight_distribution(results, out_dir):
    """
    Box plot of time-averaged weights per asset, one panel per ticker.
    One file per (method, n_assets, seed).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for r in results:
        wt_path = np.asarray(r.get("weight_path", []))
        tickers = list(r.get("tickers", []))
        if wt_path.ndim != 2 or wt_path.shape[1] == 0:
            continue

        method = r["method_name"]
        n_ast  = r.get("n_assets", wt_path.shape[1])
        seed   = r.get("seed", 0)

        fig, ax = plt.subplots(figsize=(max(8, wt_path.shape[1] * 0.9), 4))
        ax.boxplot(
            [wt_path[:, i] for i in range(wt_path.shape[1])],
            tick_labels=tickers or [f"A{i}" for i in range(wt_path.shape[1])],
            patch_artist=True,
            medianprops=dict(color="black", lw=2),
        )
        ax.axhline(0, color="grey", lw=0.7, ls="--")
        ax.set_ylabel("Weight")
        ax.set_title(f"Weight distribution — {method}  n={n_ast}  seed={seed}")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        path = out_dir / f"{method}_n{n_ast}_s{seed}.png"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
    print(f"  → {out_dir}/")


def plot_training_curves(histories, out_dir):
    """
    For each NN run, plot training E[U], validation E[U], and the scalar
    test E[U] (horizontal dashed line) on the same axes.

    One file per (arch × n_assets): overlays all seeds with shading.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Group by (arch, n_assets)
    groups = {}
    for key, h in histories.items():
        gk = (h["arch"], h["n_assets"])
        groups.setdefault(gk, []).append(h)

    for (arch, n), runs in groups.items():
        fig, ax = plt.subplots(figsize=(9, 4))

        for h in runs:
            seed  = h["seed"]
            train = h["train"]
            val   = h["val"]
            vit   = h["val_iters"]
            test  = h["test_u"]

            if not train:
                continue

            iters = range(len(train))
            ax.plot(iters, train, lw=1.2, alpha=0.7,
                    color="#ff7f0e", label=f"train (s={seed})" if seed == runs[0]["seed"] else "_")
            if val and vit:
                ax.plot(vit, val, "o--", ms=4, lw=1.0, alpha=0.85,
                        color="#1f77b4", label=f"val (s={seed})" if seed == runs[0]["seed"] else "_")
            if not np.isnan(test):
                ax.axhline(test, lw=1.2, ls=":", alpha=0.7,
                           color="#2ca02c",
                           label=f"test (s={seed})" if seed == runs[0]["seed"] else "_")

        ax.set_xlabel("Training iteration")
        ax.set_ylabel("E[U]  (goal-reaching utility)")
        ax.set_title(f"{arch}  |  n_assets={n}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # Annotate gap between final train and test to show overfit
        for h in runs:
            if h["train"] and not np.isnan(h["test_u"]):
                gap = h["train"][-1] - h["test_u"]
                ax.annotate(
                    f"gap={gap:+.3f}",
                    xy=(len(h["train"]) - 1, h["train"][-1]),
                    xytext=(-60, 10), textcoords="offset points",
                    fontsize=7, color="grey",
                    arrowprops=dict(arrowstyle="->", color="grey", lw=0.8),
                )
                break   # annotate once per arch

        fig.tight_layout()
        path = out_dir / f"{arch}_n{n}.png"
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)

    if histories:
        print(f"  → {out_dir}/ ({len(groups)} curve files)")


def plot_weights_vs_wealth(results, config, out_dir):
    """
    Plot the policy function π*(w) as a function of wealth, at several
    fixed time-to-horizon values.

    For FD   : interpolate the saved policy grid at (w, τ).
    For NN   : forward-pass over a wealth grid with τ encoded in the input.

    One file per (method × n_assets × seed): one panel per asset, curves
    for τ ∈ {T, T/2, T/4, T/8}.  This directly reveals:
      - How aggressively each method bets near the goal boundary
      - Whether the NN learned the bang-bang behaviour near τ≈0
      - How leverage varies with wealth above vs below the goal
    """
    try:
        import torch
        from comparisons.core.torch_nn_models import policy_weights as torch_pw
        HAS_TORCH = True
    except ImportError:
        HAS_TORCH = False

    out_dir.mkdir(parents=True, exist_ok=True)

    T         = getattr(config, "nn_horizon_years", 1.0)
    goal_mult = config.target_multiplier
    goal      = 1.0 * goal_mult           # normalised: w0 = 1.0
    w_grid    = np.linspace(0.4 * goal, 1.8 * goal, 200)
    tau_vals  = [T, T / 2, T / 4, T / 8]
    tau_labels = [f"τ={t:.2f}y" for t in tau_vals]
    colors     = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for r in results:
        method  = r["method_name"]
        n_ast   = r.get("n_assets", 1)
        seed    = r.get("seed", 0)
        tickers = list(r.get("tickers", [f"A{i}" for i in range(n_ast)]))
        n       = len(tickers)

        # ── FD policy ─────────────────────────────────────────────────────
        if r.get("method_family") == "fd" and "_fd_artifact" in r:
            art = r["_fd_artifact"]
            # The artifact should contain grids; try to extract policy
            grids = art.get("grids", {})
            w_fd  = grids.get("w_grid")
            pi_fd = grids.get("Pi_grid")   # (Nw+1,) for 1D or (Nw+1, n) for nD
            if w_fd is None or pi_fd is None:
                continue
            pi_fd = np.atleast_2d(pi_fd) if pi_fd.ndim == 1 else pi_fd
            # FD is time-stationary at t=0; plot the single solved policy
            fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)
            for i, ticker in enumerate(tickers):
                ax = axes[0, i]
                col = pi_fd[:, i] if pi_fd.shape[1] > i else pi_fd[:, 0]
                ax.plot(w_fd, col, lw=1.8, color="#1f77b4", label="FD t=0")
                ax.axvline(goal, color="grey", lw=0.8, ls="--", label="goal")
                ax.set_xlabel("Wealth w")
                ax.set_ylabel(f"π* ({ticker})")
                ax.set_title(ticker)
                ax.legend(fontsize=7)
                ax.grid(alpha=0.25)
            fig.suptitle(f"{method}  |  n={n_ast}  |  seed={seed}  — π*(w) at t=0",
                         fontsize=11)
            fig.tight_layout()
            path = out_dir / f"{method}_n{n_ast}_s{seed}.png"
            fig.savefig(path, dpi=130, bbox_inches="tight")
            plt.close(fig)

        # ── NN policy ─────────────────────────────────────────────────────
        elif r.get("method_family") == "nn" and HAS_TORCH:
            model_art = r.get("_model_artifact")
            if model_art is None:
                continue
            net        = model_art["model"]
            meta       = model_art.get("metadata", {})
            n_steps    = meta.get("n_steps", 40)
            total_steps = n_steps

            fig, axes = plt.subplots(
                len(tau_vals), n,
                figsize=(4 * n, 3 * len(tau_vals)),
                squeeze=False,
            )

            for ti, (tau, tlabel, col) in enumerate(
                zip(tau_vals, tau_labels, colors)
            ):
                # Map tau to step_idx: tau = T * (1 - step_idx / total_steps)
                step_idx = max(0, int(round(total_steps * (1.0 - tau / T))))

                pi_grid = np.array([
                    torch_pw(net, w, goal,
                             step_idx=step_idx,
                             total_steps=total_steps)
                    for w in w_grid
                ])   # (200, n)

                for i, ticker in enumerate(tickers):
                    ax = axes[ti, i]
                    ax.plot(w_grid, pi_grid[:, i] if pi_grid.ndim > 1 else pi_grid,
                            lw=1.6, color=col)
                    ax.axvline(goal, color="grey", lw=0.8, ls="--")
                    ax.axhline(0,    color="grey", lw=0.5, ls=":")
                    ax.set_xlabel("Wealth w" if ti == len(tau_vals) - 1 else "")
                    ax.set_ylabel(f"π* ({ticker})" if i == 0 else "")
                    ax.set_title(f"{ticker} | {tlabel}" if ti == 0 else tlabel)
                    ax.grid(alpha=0.2)

            fig.suptitle(
                f"{method}  |  n={n_ast}  |  seed={seed}  — π*(w) at fixed τ",
                fontsize=11,
            )
            fig.tight_layout()
            path = out_dir / f"{method}_n{n_ast}_s{seed}.png"
            fig.savefig(path, dpi=130, bbox_inches="tight")
            plt.close(fig)

    print(f"  → {out_dir}/")


def plot_fd_vs_nn_scatter(df, out_dir):
    """
    Scatter: goal_probability vs mean_gross_leverage, coloured by family.
    One panel per n_assets.
    """
    n_list = sorted(df["n_assets"].unique())
    fig, axes = plt.subplots(1, len(n_list), figsize=(6 * len(n_list), 5))
    if len(n_list) == 1:
        axes = [axes]

    for ax, n in zip(axes, n_list):
        sub = df[df["n_assets"] == n]
        for fam, grp in sub.groupby("family"):
            ax.scatter(grp["mean_gross_leverage"], grp["goal_probability"],
                       label=fam, color=PALETTE.get(fam, "#9467bd"),
                       s=70, alpha=0.8)
            for _, row in grp.iterrows():
                ax.annotate(row["method"][:10],
                            (row["mean_gross_leverage"], row["goal_probability"]),
                            fontsize=6, alpha=0.75)

        ax.set_xlabel("Mean gross leverage")
        ax.set_ylabel("Goal probability")
        ax.set_title(f"n_assets = {n}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)

    fig.suptitle("Goal Probability vs Leverage (by family)", fontsize=13)
    fig.tight_layout()
    path = out_dir / "goal_vs_leverage.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run FD vs NN portfolio experiment")
    parser.add_argument("--quick",   action="store_true",
                        help="fast sanity check (1 seed, 10 NN iters)")
    parser.add_argument("--no-nn",  action="store_true",
                        help="skip NN training, FD and baselines only")
    parser.add_argument("--n-assets", default=None,
                        help="comma-separated list, e.g. '5,10' (default: 5,10,20)")
    parser.add_argument("--seeds",    default=None,
                        help="comma-separated list, e.g. '1,2' (default: 1,2,3)")
    parser.add_argument("--archs",    default=None,
                        help="comma-separated NN architectures to run")
    # ── GPU flags ────────────────────────────────────────────────────────────
    parser.add_argument("--device",   default=None,
                        help="PyTorch device string, e.g. 'cuda', 'cuda:0', 'cpu'")
    parser.add_argument("--resume",   action="store_true",
                        help="skip (method, n, seed, goal_mult) combos that already "
                             "have a checkpoint under results/experiment/checkpoints/")
    parser.add_argument("--compile",  action="store_true",
                        help="enable torch.compile() on NN models (PyTorch ≥ 2.0 + CUDA)")
    args = parser.parse_args()

    n_assets_list = (
        [int(x) for x in args.n_assets.split(",")]
        if args.n_assets else None
    )
    seeds = (
        [int(x) for x in args.seeds.split(",")]
        if args.seeds else None
    )
    nn_archs = (
        [x.strip() for x in args.archs.split(",")]
        if args.archs else None
    )

    config = make_config(
        quick=args.quick,
        no_nn=args.no_nn,
        n_assets_list=n_assets_list,
        seeds=seeds,
        nn_archs=nn_archs,
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Detect device and print GPU info ──────────────────────────────────────
    _device_str = args.device
    if _device_str is None:
        try:
            import torch
            if torch.cuda.is_available():
                _device_str = "cuda"
                print(f"\n  GPU detected : {torch.cuda.get_device_name(0)}")
                print(f"  VRAM         : "
                      f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            else:
                _device_str = "cpu"
        except ImportError:
            _device_str = "cpu"

    print("\n" + "="*60)
    print("  EXPERIMENT CONFIGURATION")
    print("="*60)
    print(f"  n_assets       : {config.n_assets_list}")
    print(f"  seeds          : {config.random_seeds}")
    print(f"  device         : {_device_str}")
    print(f"  resume         : {args.resume}")
    print(f"  compile        : {args.compile}")
    print(f"  FD grid        : Nw={config.fd_nw}, Nt={config.fd_nt}")
    print(f"  NN iters       : {config.nn_iters}")
    print(f"  NN paths       : {config.nn_paths}")
    print(f"  NN pretrain    : {config.nn_pretrain_iters}")
    print(f"  NN antithetic  : {config.nn_antithetic}")
    print(f"  NN curriculum  : {config.nn_p_curriculum:.0%}")
    print(f"  NN patience    : {config.nn_patience}")
    print(f"  Architectures  : {config.nn_architectures if config.include_nn else 'none'}")
    print(f"  Results → {RESULTS_DIR}")
    print("="*60)

    # ── Run ──────────────────────────────────────────────────────────────────
    t_start = time.perf_counter()
    results, histories = run_all(
        config,
        device        = _device_str,
        resume        = args.resume,
        compile_model = args.compile,
    )
    print(f"\nTotal run time: {time.perf_counter() - t_start:.1f}s")

    if not results:
        print("No results — check data loader and config.")
        return

    # ── Summary table ─────────────────────────────────────────────────────────
    df = build_summary(results)
    csv_path = RESULTS_DIR / "summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSummary table → {csv_path}")

    # Print quick console summary
    print("\n" + "="*60)
    print("  GOAL PROBABILITY (mean across seeds)")
    print("="*60)
    pivot = (
        df.groupby(["method", "n_assets"])["goal_probability"]
        .mean()
        .unstack("n_assets")
    )
    # Sort by first n_assets column
    first_col = pivot.columns[0]
    pivot = pivot.sort_values(first_col, ascending=False)
    with pd.option_context("display.float_format", "{:.1%}".format,
                           "display.max_rows", 50):
        print(pivot.to_string())

    print("\n" + "="*60)
    print("  TRAINING TIME (seconds)")
    print("="*60)
    time_df = df.groupby("method")[["train_time_sec", "solve_time_sec"]].mean()
    time_df["total_sec"] = time_df["train_time_sec"] + time_df["solve_time_sec"]
    print(time_df.sort_values("total_sec", ascending=False).to_string())

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nGenerating plots ...")
    plot_goal_probability(df,        RESULTS_DIR)
    plot_goal_vs_multiplier(df,      RESULTS_DIR)
    plot_terminal_wealth(df,         RESULTS_DIR)
    plot_train_time(df,              RESULTS_DIR)
    plot_fd_vs_nn_scatter(df,        RESULTS_DIR)
    plot_training_curves(histories,  RESULTS_DIR / "training_curves")
    plot_weights_vs_wealth(results,  config, RESULTS_DIR / "weights_vs_wealth")
    plot_weight_paths(results,       RESULTS_DIR / "weights")
    plot_weight_distribution(results, RESULTS_DIR / "weight_dist")

    print(f"\nAll outputs saved to: {RESULTS_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
