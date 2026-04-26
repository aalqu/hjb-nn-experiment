# Notebook Workflow

## Start Jupyter

From the repo root:

```bash
python3 -m jupyter lab
```

or

```bash
python3 -m notebook
```

Open notebooks from the repository root so imports resolve cleanly.

## Recommended First Cell

```python
from pathlib import Path
import sys

ROOT = Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
```

## Load Comparison Tables

```python
from comparisons.core.notebook_api import load_summary_table

main_results = load_summary_table('main_results')
agg_results = load_summary_table('aggregated_results')
neural_results = load_summary_table('neural_family_results')
fd_vs_nn = load_summary_table('fd_vs_neural_results')
```

## Load FD Policy Bundle

```python
from comparisons.core.notebook_api import load_fd_policy_bundle

fd_bundle = load_fd_policy_bundle(n_assets=5, seed=1, initial_wealth=1.0)
fd_result = fd_bundle['result']
fd_policy = fd_bundle['policy']

fd_policy(1.0, 0.5)
```

## Load NN Model Bundle and Weights

```python
from comparisons.core.notebook_api import load_nn_model_bundle

nn_bundle = load_nn_model_bundle('transformer', n_assets=5, seed=1, initial_wealth=1.0)
model = nn_bundle['model']
weights_fn = nn_bundle['weights_fn']

weights_fn(1.0, 1.1)
```

For sequence models you can also pass history:

```python
weights_fn(1.0, 1.1, history=[0.98, 1.01, 1.03], step_idx=3, total_steps=252)
```

## Load Raw Run Arrays

```python
from comparisons.core.io import load_run_result
from pathlib import Path

raw = load_run_result(Path('comparisons/results/raw/transformer_n5_seed1_w1.00.npz'))
raw.keys()
```

## Reuse Existing Plot Files

```python
from IPython.display import Image, display

display(Image('comparisons/results/plots/fd_vs_neural_runtime.png'))
```

## Use Existing Core Modules Directly

```python
from real_data_loader import load_portfolio
from backtest_core import run_backtest_1d
from fd_core import fd_solve, make_fd_policy, goal_utility, asymp_goalreach
```

## Red-Green TDD Rule

When adding notebook-facing helpers:
1. add or update a test in `comparisons/tests/`
2. run pytest until it fails (red)
3. implement the helper (green)
4. refactor without breaking the tests
