# Critique: Portfolio Optimization Comparison Experiment
*Against target spec: `# Portfolio Optimization Comparison Expe.md`*

---

## Executive Summary

The codebase is structurally sound and partially functional, but the experiment **has not achieved its targets** in several critical dimensions. There are severe numerical instabilities producing unrealistic results in most neural methods, the evaluation design is fundamentally misaligned with the MD spec (single backtest path vs. Monte Carlo), and large portions of the required deliverables are simply absent.

---

## 1. Unreasonable Numbers — Exploding Terminal Wealth

The most glaring problem. Starting wealth is `w0 = 1.0`, target is `1.1`. The following results are physically impossible in any credible portfolio simulation:

| Method        | n  | Mean Terminal Wealth |
|---------------|----|-----------------------|
| actor_critic  | 5  | **679** (seeds: 487–860) |
| actor_critic  | 10 | **26,886** (seeds: 4,891–58,440) |
| actor_critic  | 20 | **534,389** (seeds: 169,500–1,256,000) |
| pinn          | 10 | **44,128** (seeds: 19,670–62,850) |
| pinn          | 20 | **1,920,525** (seeds: 727,100–2,738,000) |
| transformer   | 20 | **2,501,265** (seeds: 243,700–3,776,000) |
| nn_mlp_deep   | 20 | **706,308** (seeds: 36,310–1,055,000) |
| nn_mlp_small  | 5  | **963** (seed 2: 1,751) |

These are not extreme outliers — they are the *mean* across 3 seeds. The FD benchmark gives a reasonable `W_T ≈ 1.64` (n=5) and the equal-weight baseline gives `W_T ≈ 5.6` (from a single bull-market backtest path). Getting 1,920,525 from a `target_multiplier=1.10` problem means the model has gone completely off-track.

**Root cause — leverage explosion**: The weight bounds are `[−5, 3]`. With 20 assets, the maximum gross leverage is `5×20 = 100`. The neural models are learning to apply extreme leverage during training (via the `target_vol`-based volatility scaling), but that scaling is **only applied during training**, not during evaluation. In `evaluate_nn_portfolio`, `infer_weights()` calls the network directly with no leverage cap. The result: the model has been implicitly trained to expect its leverage to be rescaled, but at inference time the raw unconstrained weights are used, causing compounding explosive growth on any period with positive returns.

---

## 2. Stopping / Portfolio Collapse (Near-Zero Wealth)

The opposite failure mode affects other architectures — the portfolio loses essentially all its value:

| Method       | n  | Terminal Wealth     | Notes |
|--------------|----|--------------------|-------|
| lstm         | 10 | **0.014–0.043** (all 3 seeds) | Min path wealth ≈ 0.002 |
| lstm         | 20 | **1.2e-5 – 8.0e-5** (all 3 seeds) | Fully collapsed, min wealth ≈ floor `1e-6` |
| deep_bsde    | 10 | **1.5e-4** (seed 1); seed 3 exploded to 4,974 | Inconsistent across seeds |
| deep_bsde    | 20 | **0.006–0.33** (all 3 seeds) | Near-collapse |
| nn_mlp_small | 20 | **0.0019 and 3e-6** (2 of 3 seeds) | Near-collapse |

The LSTM's `target_hit_rate = 0.0` at n=10 and n=20 is a complete failure signal. The collapse happens because the LSTM operates on a `seq_len=8` history window — when the model is poorly trained (only 40 torch iterations on 384 paths), it outputs unstable high-leverage positions that compound into ruin on real market data. The same is true for deep_bsde which uses per-step networks (`n_steps=32` networks trained together in 40 iterations).

---

## 3. The Training/Evaluation Mismatch — The Fundamental Design Error

The entire evaluation framework conflates two different problems:

- **Training**: on simulated GBM paths (`mu_ann`, `omega_mat` → Cholesky → Brownian motion), with a volatility-scaling heuristic applied to weights during rollout.
- **Evaluation**: on real historical daily returns, with no vol-scaling, using the *same* network.

The network has never seen real-market data during training, and the vol-scaling that constrained its effective leverage during training is absent at evaluation. The models are not being tested on what they were trained on. This is not a minor inconsistency — it means the evaluation results are invalid for the purposes of the experiment.

---

## 4. The Monte Carlo Evaluation Problem

The MD spec explicitly requires:
> *"2000 Monte Carlo evaluation paths"* for estimating target-hit probability.

**What was implemented instead**: a single backtest path per seed on real historical data. The "target_hit_rate" for each seed is therefore **binary (0 or 1)**, and the cross-seed mean can only be 0, 0.33, 0.67, or 1.0. This is not a probabilistic estimate of the policy's hit rate — it is a single realized outcome.

The consequence is that all baseline strategies (equal_weight, max_sharpe, market_cap) show `target_hit_rate = 1.0` because the historical period happens to be a bull market. This makes the metric useless for comparison: a coin-flip strategy would also show 1.0 if run on the same single upward-trending path.

---

## 5. FD Benchmark Is a 1D Proxy, Not Truly Multi-Dimensional

The FD benchmark (`fd_goalreach_proxy`) is computed by:
1. Aggregating all `n` assets into a single scalar via `agg_1d(market_data)`.
2. Solving the 1D HJB PDE on that single aggregate.
3. Using the 1D policy as a single "overall portfolio weight."

So the FD method stores `n_assets = 1` regardless of whether `n = 5, 10, or 20`. The policy it learns is how much to allocate to a single lumped asset — completely different from optimally allocating across `n` individual assets. This means:

- The FD results at n=5, 10, 20 are **all solving the same 1D problem** with different market data aggregations.
- The MD's core question — "does FD scale in dimension?" — is not answered because FD is never tested in dimension > 1.
- Comparison between FD and NN at n=10 or n=20 is not apples-to-apples: FD uses 1 asset, NN uses 10 or 20.

---

## 6. Missing Required Deliverables

### Experiment Coverage
| Requirement (MD) | Implemented | Notes |
|---|---|---|
| n = 1 | ❌ No | `n_assets_list = [5, 10, 20]` in config |
| n = 5 | ✅ | |
| n = 10 | ✅ | |
| n = 20 | ✅ | |
| w0 = 0.6, 0.8, 1.0, 1.2 | ❌ Only 1.0 | Config has `initial_wealth_levels = [1.0]` |
| 5 seeds | ❌ Only 3 | Config has `random_seeds = [1, 2, 3]` |
| 2000 MC evaluation paths | ❌ Single path | See §4 above |

### Output Files and Tables
| Required Output | Present | Notes |
|---|---|---|
| `results/summary/main_results.csv` | ✅ | |
| `results/summary/value_errors.csv` | ❌ Missing | Table 2 from MD |
| `results/summary/risk_metrics.csv` | ❌ Missing | Table 3 from MD |
| Neural architecture comparison table | ❌ Missing | Table 4 from MD |
| `value_grid` in `.npz` files | ❌ Missing | Required by MD schema |
| `policy_grid` in `.npz` files | ❌ Missing | Required by MD schema |

### Plots
The MD specifies 20 required plot categories across 6 groups. Only 4 were generated:
- `fd_vs_neural_target_hit_rate.png` ✅
- `fd_vs_neural_mean_terminal_wealth.png` ✅
- `fd_vs_neural_runtime.png` ✅
- `neural_risk_vs_performance.png` ✅

Missing entirely: scalability plots, V(0,w) comparison curves, value-function error vs. wealth, policy weight vs. wealth plots (n=1, 5, 10), gross leverage vs. wealth, wealth path samples, terminal wealth histograms, etc.

### Code Structure
| Required Module | Present | Notes |
|---|---|---|
| `core/config.py` | ✅ | |
| `core/metrics.py` | ✅ | |
| `core/evaluation.py` | ✅ | |
| `core/io.py` | ✅ | |
| `fd/hjb_solver.py` | ❌ | External import (`fd_core`) |
| `fd/policy_extraction.py` | ❌ | External import |
| `nn/architectures.py` | Partial | Embedded in `core/torch_nn_models.py` |
| `plots/benchmark_plots.py` | ❌ | Embedded in `core/reporting.py` |
| Notebooks (01–05) | ❌ All missing | 0 of 5 notebooks created |
| Tests | Partial | 4 test files exist but no `test_market.py`, `test_fd_solver.py` |

---

## 7. Additional Numerical and Logic Issues

### Gross Leverage Completely Unconstrained at Evaluation
The `weight_lower_bound = -5.0` and `weight_upper_bound = 3.0` are per-asset bounds, so for n=20 the theoretical max gross leverage is 5×20 = 100. The results confirm this:
- pinn n=20: mean gross leverage = 58.5, max = 64.8
- actor_critic n=20: mean = 50.1, max = 59.2

The MD's "fairness requirements" explicitly ask for a "same effective risk budget" comparison mode. This is never implemented. Comparing methods where one has mean_gross_leverage = 0.27 (FD proxy) against another with 58.5 (pinn n=20) is meaningless — the pinn is taking 200× more leverage.

### FD Turnover = 732 Per Year
The FD proxy shows `turnover = 732.4`. This is the sum of daily absolute weight changes over one year (252 trading days). A turnover of 732 means on average 2.9× the portfolio is turned over every single day — which is not realistic for a goal-reaching strategy and suggests the policy is oscillating erratically. The FD solve produces a policy `pi(w, tau)` which is re-evaluated daily as wealth changes; however, if the wealth is trending far from the goal (in this case it grows substantially), the policy may whipsaw back and forth near constraint boundaries.

### `solve_time_sec = 0.0` for All Neural Methods
All NN methods report `solve_time_sec = 0.0` while only `train_time_sec > 0`. This is correct by design but means the "Solve/Train Time" column in the benchmark table conflates two very different things depending on the method family. The FD has `solve_time_sec = 0.526` and `train_time_sec = 0.0`; the NN has the reverse. The reporting system adds them (or just takes one), potentially causing confusion.

### Baseline Strategies Have Unrealistically High Terminal Wealth
Equal weight n=5 shows `mean_terminal_wealth = 5.59` from a single path, starting at 1.0. This is a 459% gain over one year (or whatever the historical window is). Max Sharpe n=5 shows 10.63 (963% gain). These are not representative expected values — they are single realized paths from a particularly favorable period. This is further evidence that the evaluation needs Monte Carlo over simulated paths, not a single historical backtest.

---

## 8. What Is Working Well

Despite the above, several components are correctly implemented:

- **Metrics module** (`core/metrics.py`): The formulas for target_hit_rate, expected shortfall, gross leverage, max drawdown, turnover, etc. are correct.
- **Result schema validation**: `validate_result_schema()` correctly checks field presence and dimension consistency.
- **Architecture diversity**: 7 architectures (MLP small/deep, Deep BSDE, PINN, Actor-Critic, LSTM, Transformer) are all implemented with proper PyTorch modules, with sensible forward passes.
- **Config dataclass**: `BenchmarkConfig` is clean and extensible.
- **FD integration** (within its 1D scope): The FD solver (`fd_solve`) is called correctly, and the 1D policy extraction and backtest are plausible.
- **Test files**: Basic tests exist for metrics, evaluation pipeline, and NN results (though not covering the full TDD spec).
- **Saving/loading**: The `.npz` save/load infrastructure is correct and the result schema is internally consistent.

---

## 9. Priority Fixes

Listed in order of criticality:

1. **Fix the evaluation vs. training mismatch**: Either (a) evaluate on simulated MC paths using the same GBM as training, or (b) add the same vol-scaling heuristic to `evaluate_nn_portfolio`. The current code trains with vol-scaling but evaluates without it — this alone explains most of the explosions.

2. **Add a gross leverage cap at evaluation**: After computing `weights = infer_weights(wealth[idx])`, clip so that `sum(|w_i|) <= some_max_leverage` before applying returns. This is the "fairness" requirement from the MD.

3. **Implement proper Monte Carlo evaluation**: After training, evaluate each policy on `N=2000` fresh simulated paths (or a held-out historical bootstrap). The hit rate estimate must be based on a distribution, not a single path.

4. **Add n=1 to `n_assets_list`**: Required by the MD spec, and important for verifying that the FD and NN methods agree in the simplest case.

5. **Add remaining initial wealth levels and seeds**: `initial_wealth_levels = [0.6, 0.8, 1.0, 1.2]`, `random_seeds = [1, 2, 3, 4, 5]`.

6. **Implement true multi-dimensional FD** or clearly relabel the FD proxy as a 1D comparison only: The current "fd_goalreach_proxy at n=10" is misleading because it solves a 1D problem regardless.

7. **Increase training iterations substantially**: 18 CMA-ES iterations with population 20 (numpy) or 40 Adam steps (torch) on 384 paths is not enough for convergence. The MD implies serious training.

8. **Create the 5 required notebooks**: These are listed explicitly in the MD and are entirely absent.

9. **Generate the missing plots and tables**: Value-error tables, risk-profile table, policy-weight-vs-wealth plots, terminal wealth histograms.

10. **Store `value_grid` and `policy_grid`** in the result schema as required by the MD spec.
