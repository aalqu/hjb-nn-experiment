# Experiment Plan: Corrected HJB Viscosity FD vs Neural Network Architectures

## Objective

Build a reproducible experiment framework to compare two classes of methods for a target-return portfolio optimization problem:

1. **Finite-difference corrected HJB viscosity solution**
2. **Neural network methods with different architectures**

The comparison should be done across multiple asset dimensions:

- `n = 1`
- `n = 5`
- `n = 10`
- `n = 20`

The framework should support:

- benchmarking target-achievement performance
- benchmarking computational cost
- comparing portfolio weights and allocation behavior
- comparing risk-taking behavior
- testing multiple neural network architectures under the same setup
- implementing the codebase with clean notebooks, backend/core files, and red/green TDD

---

## Methods to Compare

## Method Family A: Finite Difference HJB

This method is the:

- **corrected HJB viscosity solution**
- solved by **finite-difference methods**
- treated as the PDE / control benchmark

This method should serve as the reference solution in low dimensions where it is computationally feasible.

Key role:
- benchmark for value function quality
- benchmark for policy structure
- reference point for comparison with neural methods

## Method Family B: Neural Networks

This method family includes multiple neural network architectures for solving or approximating the same target-return control problem.

Examples may include:
- feedforward network policy approximation
- Deep BSDE-style architecture
- time-sliced networks
- recurrent architecture if useful
- shared network vs per-time-step network
- shallow vs deep variants

Key role:
- scalable approximation in higher dimensions
- architecture study within the same optimization problem
- tradeoff between compute, accuracy, and policy behavior

---

## Core Research Questions

1. How closely do neural network methods match the corrected HJB viscosity FD solution in low dimensions?
2. Which neural architecture performs best on the target-return objective?
3. How do the methods compare in:
   - target-hit probability
   - shortfall
   - terminal wealth
4. How do they compare in:
   - computational cost
   - training / solve time
   - scalability with dimension
5. How do the portfolio policies differ in:
   - leverage
   - concentration
   - aggressiveness
   - stability over wealth and time
6. Are neural methods learning the same qualitative policy structure as the FD HJB benchmark?

---

## High-Level Deliverables

Codex should generate:

- clean notebooks for experiments and analysis
- backend/core modules for reusable logic
- tests using red/green TDD
- result-saving utilities
- plotting utilities
- reproducible configuration system

Suggested codebase outputs:

- experiment notebooks
- core solver interfaces
- FD HJB module
- neural architecture module
- evaluation module
- metrics module
- plotting module
- test suite

---

## Experimental Design

All methods must be compared under the same problem definition.

### Fixed Problem Inputs

Use the same for all methods:

- market model or dataset
- investment horizon
- target wealth / target return
- initial wealth grid
- rebalancing frequency
- interest rate
- portfolio constraints
- training / evaluation seeds where applicable

### Dimensions

Evaluate at:

- `n = 1`
- `n = 5`
- `n = 10`
- `n = 20`

### Initial Wealth Levels

Suggested values:

- `w0 = 0.6`
- `w0 = 0.8`
- `w0 = 1.0`
- `w0 = 1.2`

### Random Seeds

Suggested seeds:

- `1, 2, 3, 4, 5`

These should be used consistently across neural training and Monte Carlo evaluation.

---

## Comparison Structure

The comparison has two layers.

## Layer 1: FD HJB vs Neural Family

Compare:

- corrected HJB viscosity FD solution
- best-performing neural architectures
- baseline neural architecture

Purpose:
- determine whether neural methods reproduce FD behavior
- quantify the gap between PDE benchmark and NN approximation

## Layer 2: Neural Architecture Comparison

Within the neural family, compare architectures on:

- accuracy relative to FD
- target-hit performance
- runtime
- memory
- policy smoothness
- stability across seeds

Purpose:
- determine which NN architecture is best suited for this control problem

---

## Metrics

## 1. Target-Achievement Metrics

These measure how well the method solves the investment objective.

- Target-hit probability:
  - `P(W_T >= target)`
- Mean terminal wealth
- Median terminal wealth
- 5th percentile terminal wealth
- Expected shortfall:
  - `E[max(target - W_T, 0)]`

These should be reported for:
- FD HJB
- each neural architecture

## 2. Value-Function Metrics

These compare neural approximations to the FD benchmark.

- `V(0, w)` at selected wealth levels
- absolute error vs FD on wealth grid
- relative error vs FD on wealth grid
- optional norm over grid:
  - `L1`
  - `L2`
  - `L∞`

These are especially important for:
- `n = 1`
- `n = 5`

where FD is still a credible benchmark

## 3. Policy / Portfolio Metrics

These measure how the methods allocate capital.

- portfolio weights as function of wealth
- portfolio weights as function of time
- gross leverage:
  - `sum(abs(w_i))`
- net exposure:
  - `sum(w_i)`
- concentration:
  - `sum(w_i^2)`
- maximum single-name weight:
  - `max_i abs(w_i)`
- turnover:
  - `sum(abs(w_t - w_{t-1}))`

These are crucial to understanding whether a method wins by:
- smarter control
- more aggressive leverage
- more concentrated portfolios

## 4. Realised Risk Metrics

These measure pathwise realized outcomes.

- realised wealth volatility
- maximum drawdown
- downside tail of terminal wealth
- drawdown distribution
- pathwise leverage statistics

## 5. Computational Metrics

These measure scalability and practicality.

For FD:
- solve time
- memory usage if available

For NN:
- training time
- evaluation / inference time
- parameter count
- memory usage if available

For both:
- runtime scaling with `n`
- runtime scaling with number of paths / grid size
- runtime scaling with architecture size where relevant

---

## Neural Architecture Study

The experiment framework should support several architectures.

Suggested architecture comparison dimensions:

- shared network vs one-network-per-time-step
- shallow vs deep
- narrow vs wide
- different activation choices
- different batch sizes
- different number of time steps
- optional recurrent vs feedforward structure

Each neural architecture should be represented as a named experiment variant, for example:

- `nn_mlp_small`
- `nn_mlp_deep`
- `nn_bsde_shared`
- `nn_bsde_time_sliced`

The code should make it easy to add or remove architectures through configuration.

---

## Fairness Requirements

The comparison should be fair and consistent.

All methods must use:

- same target
- same horizon
- same constraints
- same market parameters
- same evaluation paths for Monte Carlo comparison

Where relevant, include two fairness modes:

## Mode A: Same Hard Constraints

All methods must obey identical box constraints:
- lower bound
- upper bound

## Mode B: Same Effective Risk Budget

Compare methods after normalizing or reporting:
- average gross leverage
- average net exposure

This helps determine whether one method is truly better, or simply taking more risk.

---

## Outputs to Save Per Run

For each combination of:

- method family
- specific method / architecture
- number of assets
- seed

save:

- solve/train time
- evaluation time
- value grid
- policy grid
- Monte Carlo terminal wealth
- goal-hit indicators
- pathwise weights
- gross leverage paths
- net exposure paths
- concentration paths
- turnover if available

Suggested raw result filename pattern:

- `results/raw/{method_name}_n{n}_seed{seed}.npz`

Suggested aggregated output:

- `results/summary/main_results.csv`
- `results/summary/value_errors.csv`
- `results/summary/risk_metrics.csv`

---

## Standard Result Schema

Each run should store fields such as:

- `method_family`
- `method_name`
- `n_assets`
- `seed`
- `train_time_sec`
- `solve_time_sec`
- `eval_time_sec`
- `wealth_grid`
- `value_grid`
- `policy_grid`
- `mc_terminal_wealth`
- `mc_goal_hit`
- `mc_gross_leverage`
- `mc_net_exposure`
- `mc_concentration`
- `mc_drawdown`

This schema should be used across FD and neural methods so downstream analysis is uniform.

---

## Tables to Include

## Table 1: Main Benchmark Table

Columns:

- `Method`
- `Family`
- `n`
- `Solve/Train Time (s)`
- `Eval Time (s)`
- `Target Hit Rate`
- `Mean W_T`
- `Median W_T`
- `Shortfall`
- `Mean Gross Leverage`
- `Mean Concentration`

## Table 2: FD vs Neural Value Error Table

Columns:

- `Method`
- `n`
- `Wealth Grid Error L1`
- `Wealth Grid Error L2`
- `Wealth Grid Error Linf`
- `V(0,0.8) Error`
- `V(0,1.0) Error`

This is mainly for neural methods against FD.

## Table 3: Risk Profile Table

Columns:

- `Method`
- `n`
- `Mean Net Exposure`
- `Max Gross Leverage`
- `Mean Max Position`
- `Turnover`
- `Wealth Volatility`
- `Max Drawdown`

## Table 4: Neural Architecture Comparison Table

Columns:

- `Architecture`
- `n`
- `Train Time`
- `Inference Time`
- `Target Hit Rate`
- `Value Error vs FD`
- `Mean Gross Leverage`
- `Max Drawdown`

---

## Plots to Include

## A. Scalability Plots

1. FD solve time vs number of assets
2. Neural training time vs number of assets
3. Neural inference time vs number of assets
4. Runtime comparison across methods

## B. Target-Achievement Plots

5. Target-hit probability vs number of assets
6. Mean terminal wealth vs number of assets
7. Expected shortfall vs number of assets

## C. Value-Function Comparison Plots

For low dimensions where FD is benchmark:

8. `V(0,w)` curves: FD vs neural architectures
9. Value-function absolute error vs wealth
10. Value-function heatmaps if time dimension is included

## D. Policy Structure Plots

For `n = 1`, `n = 5`, and possibly `n = 10`:

11. Portfolio weights vs wealth at `t = 0`
12. Portfolio weights vs wealth at different times
13. Gross leverage vs wealth
14. Concentration vs wealth
15. Maximum single-position size vs wealth

These plots should show whether the neural policy learns the same qualitative structure as FD.

## E. Realised Behavior Plots

16. Sample wealth paths under FD and neural methods
17. Gross leverage over time
18. Net exposure over time
19. Largest position over time
20. Histogram of terminal wealth

## F. Interpretation / Fairness Plots

21. Risk vs performance scatter:
- x-axis: mean gross leverage
- y-axis: target-hit probability
- color: method family
- marker shape: architecture
- marker size: number of assets

This is essential for interpreting whether better performance comes from more risk-taking.

---

## Proposed Codebase Structure

Codex should generate a clean modular structure.

Suggested folders and files:

- `core/config.py`
- `core/market.py`
- `core/metrics.py`
- `core/evaluation.py`
- `core/io.py`

- `fd/hjb_solver.py`
- `fd/policy_extraction.py`

- `nn/architectures.py`
- `nn/train.py`
- `nn/inference.py`
- `nn/losses.py`

- `plots/benchmark_plots.py`
- `plots/policy_plots.py`
- `plots/risk_plots.py`

- `experiments/run_benchmark.py`
- `experiments/run_architecture_sweep.py`

- `notebooks/01_problem_setup.ipynb`
- `notebooks/02_fd_reference.ipynb`
- `notebooks/03_nn_architecture_sweep.ipynb`
- `notebooks/04_benchmark_comparison.ipynb`
- `notebooks/05_analysis.ipynb`

- `tests/test_metrics.py`
- `tests/test_market.py`
- `tests/test_fd_solver.py`
- `tests/test_nn_interfaces.py`
- `tests/test_evaluation_pipeline.py`

---

## TDD Requirements

Use red/green TDD for the core components.

Codex should:

1. write failing tests first
2. implement the minimum code to pass
3. refactor after green
4. keep interfaces stable across FD and NN methods

Key test targets:

- market generation
- result schema consistency
- metric calculations
- FD solver interface
- neural model interface
- evaluation pipeline
- save/load logic

Important interface standardization:

```python
policy_fn(t, w, state=None) -> weights
and for solver outputs:

{
    "method_family": ...,
    "method_name": ...,
    "wealth_grid": ...,
    "value_grid": ...,
    "policy_grid": ...
}
Notebook-by-Notebook Plan
Notebook 1: Problem Setup
Purpose:

define the target-return problem
define market generation or dataset loading
define constraints
define evaluation metrics
verify the experiment configuration
Notebook 2: FD Reference
Purpose:

run the corrected HJB viscosity FD method
generate benchmark value functions
generate benchmark policies
verify low-dimensional correctness
save FD reference outputs
Notebook 3: Neural Architecture Sweep
Purpose:

train several neural architectures
compare convergence
compare runtime
compare value function approximation against FD where possible
save trained outputs and metadata
Notebook 4: Benchmark Comparison
Purpose:

compare FD and neural methods across n = 1, 5, 10, 20
run Monte Carlo evaluation
produce benchmark tables
produce runtime tables
Notebook 5: Analysis
Purpose:

create final plots
compare policy structure
compare leverage and concentration behavior
interpret whether neural policies match FD qualitatively
summarize tradeoffs between accuracy, compute, and risk
Interpretation Checklist
The final analysis should answer:

How well do neural methods approximate the corrected HJB viscosity FD benchmark?
Which neural architecture is best overall?
Which method achieves the target most reliably?
Is improved performance due to:
better approximation
higher leverage
more concentration
Do neural methods recover the same qualitative policy shape as FD?
At what dimension does FD become impractical?
At what dimension do neural methods become clearly preferable?
Are the gains economically meaningful, or only computational?
Minimal Credible Study
A strong minimal version should include:

FD HJB benchmark
at least 2 to 4 neural architectures
n = 1, 5, 10, 20
5 seeds
2000 Monte Carlo evaluation paths
2 to 4 summary tables
8 to 12 core plots
This is enough for a serious comparison without overcomplicating the implementation.

Recommended Figure Order
FD vs neural runtime scaling
Target-hit probability scaling
Shortfall scaling
V(0,w) comparison at low dimension
Weight-vs-wealth comparison at n = 5
Gross leverage-vs-wealth comparison
Terminal wealth histograms
Risk-vs-performance scatter
Neural architecture comparison table
Main benchmark summary table
Final Implementation Goal for Codex
Use this plan to generate:

notebooks
backend/core files
solver interfaces
neural architecture modules
experiment runners
plotting tools
tests using red/green TDD
The final system should make it easy to:

add new neural architectures
rerun the benchmark for new targets or dimensions
compare FD and neural methods on the same exact problem
inspect not only performance, but also the policy and risk mechanisms behind the results
Bottom Line
This experiment is not only about identifying the best-performing method.

It is about understanding:

how the corrected HJB viscosity FD benchmark behaves
whether neural architectures can reproduce that benchmark
when neural methods become necessary for scalability
whether gains come from better control or simply from more aggressive risk-taking
The framework should therefore be designed to compare:

accuracy
compute
policy structure
risk-taking
scalability
