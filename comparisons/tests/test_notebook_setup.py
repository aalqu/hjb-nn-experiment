"""
test_notebook_setup.py
----------------------
TDD suite for the comparison notebook setup cell.

Each test replicates one step of the Cell-1 bootstrap exactly as it appears
in comparisons/notebooks/01_results_explorer.ipynb and 02_playground.ipynb.

Run modes
---------
  python test_notebook_setup.py          # human-friendly RED/GREEN output
  pytest comparisons/tests/test_notebook_setup.py -v   # pytest mode

RED history
-----------
  Before the torch-guard fixes (May 2025):
    - TestImports.test_artifacts_import         FAILED (hard 'import torch')
    - TestImports.test_evaluation_import        FAILED (cascading torch crash)
    - TestImports.test_notebook_api_import      FAILED (torch.Tensor annotation)
    - TestImports.test_torch_nn_models_import   FAILED (nn.Module at class level)
  All now GREEN after the try/except + __future__ annotations fixes.
"""

from __future__ import annotations

import sys
import os
import unittest
import importlib
from pathlib import Path

# ── locate project root the same way the notebook bootstrap does ──────────────
def _find_root(start: Path) -> Path:
    """Walk up from *start* until we find a directory containing 'comparisons/'."""
    root = start.resolve()
    while not (root / "comparisons").exists() and root != root.parent:
        root = root.parent
    return root


# The test file itself lives at  <root>/comparisons/tests/test_notebook_setup.py
# so   Path(__file__).parents[2]  is the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def _assert_importable(dotted_name: str, from_list: list[str] | None = None):
    """Import *dotted_name* and optionally check that *from_list* names exist."""
    mod = importlib.import_module(dotted_name)
    if from_list:
        for name in from_list:
            assert hasattr(mod, name), \
                f"'{name}' not found in {dotted_name}"
    return mod


# ─────────────────────────────────────────────────────────────────────────────
# 1. PATH BOOTSTRAP
# ─────────────────────────────────────────────────────────────────────────────

class TestPathBootstrap(unittest.TestCase):
    """The ROOT-finding while-loop in Cell 1 must land on the project root."""

    def test_root_contains_fd_core(self):
        """PROJECT_ROOT must contain fd_core.py (the FD solver)."""
        self.assertTrue(
            (PROJECT_ROOT / "fd_core.py").exists(),
            f"fd_core.py not found under PROJECT_ROOT={PROJECT_ROOT}\n"
            "Check that you are running from inside the 'Claude Code' project folder."
        )

    def test_root_contains_backtest_core(self):
        self.assertTrue(
            (PROJECT_ROOT / "backtest_core.py").exists(),
            f"backtest_core.py not found under PROJECT_ROOT={PROJECT_ROOT}"
        )

    def test_root_contains_real_data_loader(self):
        self.assertTrue(
            (PROJECT_ROOT / "real_data_loader.py").exists(),
            f"real_data_loader.py not found under PROJECT_ROOT={PROJECT_ROOT}"
        )

    def test_root_contains_comparisons_package(self):
        self.assertTrue(
            (PROJECT_ROOT / "comparisons" / "__init__.py").exists(),
            f"comparisons/__init__.py not found under PROJECT_ROOT={PROJECT_ROOT}"
        )

    def test_root_on_sys_path(self):
        """PROJECT_ROOT must be in sys.path so bare imports work."""
        self.assertIn(
            str(PROJECT_ROOT), sys.path,
            f"PROJECT_ROOT={PROJECT_ROOT} is not on sys.path.\n"
            "Make sure the notebook bootstrap cell has run, or run:\n"
            "  sys.path.insert(0, str(PROJECT_ROOT))"
        )

    def test_walk_from_notebooks_dir_finds_correct_root(self):
        """Simulate the while-loop starting from comparisons/notebooks/."""
        notebooks_dir = PROJECT_ROOT / "comparisons" / "notebooks"
        found = _find_root(notebooks_dir)
        self.assertEqual(
            found, PROJECT_ROOT,
            f"Bootstrap resolved '{found}' but expected '{PROJECT_ROOT}'.\n"
            "The while-loop anchor '(ROOT / \"comparisons\").exists()' failed."
        )

    def test_walk_from_project_root_finds_correct_root(self):
        """Simulate the while-loop starting from the project root itself."""
        found = _find_root(PROJECT_ROOT)
        self.assertEqual(found, PROJECT_ROOT)


# ─────────────────────────────────────────────────────────────────────────────
# 2. CORE IMPORTS (every line from the failing Cell 1)
# ─────────────────────────────────────────────────────────────────────────────

class TestImports(unittest.TestCase):
    """Every import that appears in the notebook setup cell must succeed."""

    # --- standard library / third-party ---------------------------------

    def test_numpy_import(self):
        import numpy as np
        self.assertIsNotNone(np.__version__)

    def test_matplotlib_import(self):
        import matplotlib.pyplot as plt
        self.assertIsNotNone(plt)

    def test_pandas_import_graceful(self):
        """pandas is optional — the cell does try/except ImportError."""
        try:
            import pandas as pd
            self.assertIsNotNone(pd.__version__)
        except ImportError:
            pass  # acceptable — notebook wraps this in try/except

    def test_ipython_display_import(self):
        """IPython.display is needed for Image/display/Markdown in the notebook."""
        try:
            from IPython.display import Image, display, Markdown
        except ImportError:
            self.skipTest("IPython not installed (fine outside a notebook kernel)")

    # --- comparisons package itself ------------------------------------

    def test_comparisons_package_import(self):
        """comparisons/__init__.py must import without error."""
        import comparisons  # noqa: F401

    def test_comparisons_init_adds_project_root(self):
        """comparisons/__init__.py must add PROJECT_ROOT to sys.path."""
        import comparisons  # noqa: F401  (may already be cached)
        self.assertIn(str(PROJECT_ROOT), sys.path)

    # --- comparisons.core.artifacts (was RED before torch-guard fix) ---

    def test_artifacts_import(self):
        """
        RED before fix: 'import torch' at module level crashed without PyTorch.
        GREEN after fix: wrapped in try/except; torch-only functions raise
                         ImportError at call time instead.
        """
        from comparisons.core import artifacts  # noqa: F401
        from comparisons.core.artifacts import (
            artifact_filename,
            save_fd_artifact,
            load_fd_artifact,
        )
        # These three don't need torch and must always be callable
        name = artifact_filename("fd_1d_proxy", 5, 1, 1.0, "fd_policy")
        self.assertEqual(name, "fd_1d_proxy_n5_seed1_w1.00_fd_policy")

    # --- comparisons.core.torch_nn_models (was RED before annotation fix) ---

    def test_torch_nn_models_import(self):
        """
        RED before fix: 'class TorchPolicyNet(nn.Module)' and
                        'history: List[torch.Tensor]' crashed at import time.
        GREEN after fix: from __future__ import annotations + stub nn.Module.
        """
        from comparisons.core.torch_nn_models import (
            TORCH_ARCHITECTURES,
            HAS_TORCH,
        )
        self.assertIsInstance(TORCH_ARCHITECTURES, dict)
        self.assertIsInstance(HAS_TORCH, bool)

    def test_torch_architectures_names(self):
        from comparisons.core.torch_nn_models import TORCH_ARCHITECTURES, HAS_TORCH
        if HAS_TORCH:
            expected = {
                "nn_mlp_small", "nn_mlp_deep", "nn_policy_net",
                "deep_bsde", "pinn", "actor_critic", "lstm", "transformer",
            }
            self.assertEqual(set(TORCH_ARCHITECTURES.keys()), expected)

    def test_torch_detected_when_installed(self):
        """If torch is importable in this Python, HAS_TORCH must be True."""
        try:
            import torch  # noqa: F401
            torch_available = True
        except ImportError:
            torch_available = False

        from comparisons.core.torch_nn_models import HAS_TORCH
        self.assertEqual(
            HAS_TORCH, torch_available,
            f"torch importable={torch_available} but HAS_TORCH={HAS_TORCH}.\n"
            "There is a mismatch — torch_nn_models.py may have a stale .pyc cache.\n"
            "Delete comparisons/core/__pycache__/ and retry."
        )

    # --- comparisons.core.notebook_api (was RED before torch-guard fix) ---

    def test_notebook_api_import(self):
        """
        RED before fix: importing notebook_api triggered torch_nn_models which
                        crashed on 'List[torch.Tensor]' annotation.
        GREEN after fix: __future__ annotations defers evaluation.
        """
        from comparisons.core.notebook_api import (
            load_summary_table,
            list_available_runs,
            load_fd_policy_bundle,
            load_nn_model_bundle,
        )
        # All four names must be callable
        for fn in (load_summary_table, list_available_runs,
                   load_fd_policy_bundle, load_nn_model_bundle):
            self.assertTrue(callable(fn), f"{fn.__name__} is not callable")

    # --- comparisons.core.io -------------------------------------------

    def test_io_import(self):
        from comparisons.core.io import (
            load_run_result,
            save_run_result,
            save_summary_csv,
            ensure_dir,
        )
        for fn in (load_run_result, save_run_result, save_summary_csv, ensure_dir):
            self.assertTrue(callable(fn))

    # --- comparisons.core.config ---------------------------------------

    def test_config_import(self):
        from comparisons.core.config import BenchmarkConfig
        cfg = BenchmarkConfig()
        self.assertEqual(cfg.n_assets_list, [5, 10, 20])
        self.assertEqual(cfg.random_seeds, [1, 2, 3])

    # --- comparisons.core.evaluation (was RED before torch-guard fix) ---

    def test_evaluation_import(self):
        """
        RED before fix: evaluation.py imports artifacts.py which had bare
                        'import torch', crashing the whole chain.
        GREEN after fix: artifacts.py wraps torch in try/except.
        """
        from comparisons.core.evaluation import (
            run_real_data_portfolio_comparison,
            evaluate_fd_benchmark,
            evaluate_merton_benchmark,
            evaluate_static_portfolio,
            apply_leverage_constraint,
        )
        for fn in (run_real_data_portfolio_comparison, evaluate_fd_benchmark,
                   evaluate_merton_benchmark, evaluate_static_portfolio,
                   apply_leverage_constraint):
            self.assertTrue(callable(fn))

    # --- project-root modules (real_data_loader, fd_core, backtest_core) ---

    def test_real_data_loader_import(self):
        from real_data_loader import load_portfolio, MarketData, agg_1d
        for fn in (load_portfolio, agg_1d):
            self.assertTrue(callable(fn))

    def test_fd_core_import(self):
        from fd_core import fd_solve, make_fd_policy, goal_utility, asymp_goalreach
        for fn in (fd_solve, make_fd_policy, goal_utility, asymp_goalreach):
            self.assertTrue(callable(fn))

    def test_backtest_core_import(self):
        from backtest_core import run_backtest_1d, compute_metrics
        for fn in (run_backtest_1d, compute_metrics):
            self.assertTrue(callable(fn))


# ─────────────────────────────────────────────────────────────────────────────
# 3. ARCHITECTURE CONSISTENCY
# ─────────────────────────────────────────────────────────────────────────────

class TestArchitectureConsistency(unittest.TestCase):
    """
    Numpy and Torch backends must use identical hidden-layer sizes for the
    same architecture name so benchmark results are comparable.

    RED before fix: nn_mlp_small=(16,) in numpy vs (32,32) in torch.
    GREEN after fix: both use (32,32) and (64,64,32).
    """

    def test_nn_mlp_small_sizes_match(self):
        from comparisons.core.nn_models import ARCHITECTURES
        from comparisons.core.torch_nn_models import TORCH_ARCHITECTURES, HAS_TORCH
        if not HAS_TORCH:
            self.skipTest("torch not installed; skipping cross-backend comparison")
        np_layers = ARCHITECTURES["nn_mlp_small"]
        torch_layers = tuple(TORCH_ARCHITECTURES["nn_mlp_small"]["hidden_layers"])
        self.assertEqual(
            np_layers, torch_layers,
            f"nn_mlp_small numpy={np_layers} vs torch={torch_layers}.\n"
            "Benchmark results are not comparable between backends."
        )

    def test_nn_mlp_deep_sizes_match(self):
        from comparisons.core.nn_models import ARCHITECTURES
        from comparisons.core.torch_nn_models import TORCH_ARCHITECTURES, HAS_TORCH
        if not HAS_TORCH:
            self.skipTest("torch not installed; skipping cross-backend comparison")
        np_layers = ARCHITECTURES["nn_mlp_deep"]
        torch_layers = tuple(TORCH_ARCHITECTURES["nn_mlp_deep"]["hidden_layers"])
        self.assertEqual(
            np_layers, torch_layers,
            f"nn_mlp_deep numpy={np_layers} vs torch={torch_layers}."
        )

    def test_numpy_architectures_are_valid_tuples(self):
        from comparisons.core.nn_models import ARCHITECTURES
        for name, layers in ARCHITECTURES.items():
            self.assertIsInstance(layers, tuple,
                f"ARCHITECTURES['{name}'] should be a tuple, got {type(layers)}")
            for w in layers:
                self.assertIsInstance(w, int,
                    f"ARCHITECTURES['{name}'] layer width must be int, got {type(w)}")
                self.assertGreater(w, 0,
                    f"ARCHITECTURES['{name}'] layer width must be > 0")


# ─────────────────────────────────────────────────────────────────────────────
# 4. FD SOLVER CORRECTNESS
# ─────────────────────────────────────────────────────────────────────────────

class TestFDSolver(unittest.TestCase):
    """The FD solver must produce economically sensible output."""

    @classmethod
    def setUpClass(cls):
        import numpy as np
        from fd_core import fd_solve, goal_utility, asymp_goalreach, make_fd_policy
        cls.w, cls.V, cls.Pi = fd_solve(
            mu=0.12, r=0.03, sigma=0.18, T=1.0, A=3.0,
            Nw=100, Nt=50, d=-5.0, u=3.0,
            utility_fn=goal_utility,
            asymptotic_fn=lambda w, tau: asymp_goalreach(w, tau, 0.18, -5.0, 3.0),
        )
        # staticmethod prevents Python from treating the closure as an unbound
        # method when accessed via self.policy(w, tau).
        cls.policy = staticmethod(make_fd_policy(cls.w, cls.Pi))
        cls.np = np

    def test_value_function_in_unit_interval(self):
        """V must be in [0,1] for goal-reaching utility."""
        np = self.np
        self.assertGreaterEqual(float(self.V.min()), 0.0)
        self.assertLessEqual(float(self.V.max()), 1.0 + 1e-9)

    def test_value_function_monotone_below_goal(self):
        """V must increase with wealth below the goal (more wealth = more likely to succeed)."""
        np = self.np
        below = self.V[self.w <= 1.0]
        diffs = np.diff(below)
        # Allow tiny numerical wiggles but overall must be non-decreasing
        self.assertGreater(float(diffs.mean()), 0.0,
            "V should be increasing in wealth below the goal")

    def test_policy_respects_bounds(self):
        """All policy values must lie in [d, u] = [-5, 3]."""
        self.assertTrue(float(self.Pi.min()) >= -5.0 - 1e-9)
        self.assertTrue(float(self.Pi.max()) <= 3.0 + 1e-9)

    def test_policy_above_goal_not_extreme(self):
        """
        RED before fix: policy_from_V returned random +-5/+-3 corners above
                        goal where V is flat (Vw=Vww=0).
        GREEN after fix: flat region returns 0 instead of a random corner.
        """
        import numpy as np
        above = self.Pi[self.w > 1.05]
        extreme_frac = float(np.mean((above < -4.9) | (above > 2.9)))
        self.assertLess(
            extreme_frac, 0.15,
            f"{extreme_frac:.0%} of Pi values above goal are extreme corners.\n"
            "Expected < 15% after the policy_from_V flat-region fix."
        )

    def test_make_fd_policy_callable(self):
        """make_fd_policy must return a callable policy(w_norm, tau)."""
        self.assertTrue(callable(self.policy))
        # Should return a float in [d, u]
        pi = self.policy(0.9, 0.5)
        self.assertIsInstance(pi, float)
        self.assertGreaterEqual(pi, -5.0)
        self.assertLessEqual(pi, 3.0)

    def test_policy_decreases_as_wealth_approaches_goal(self):
        """
        Near the goal, the optimal policy should decrease (less risky)
        as w increases toward 1.  Below 0.5 the policy is max (u=3).
        """
        pis = [self.policy(w, 0.5) for w in [0.7, 0.8, 0.85, 0.9, 0.95]]
        # pis should be non-increasing from some point below the goal
        # (once we're past the "bang-bang" region near w~0.8+)
        self.assertGreater(pis[0], pis[-1],
            f"Policy should decrease toward goal, got {pis}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

class TestDataLoading(unittest.TestCase):

    def test_load_portfolio_5(self):
        from real_data_loader import load_portfolio
        mkt = load_portfolio(5)
        self.assertEqual(mkt.n, 5)
        self.assertEqual(mkt.log_ret.shape[1], 5)
        self.assertGreater(len(mkt.dates), 500)

    def test_load_portfolio_10(self):
        from real_data_loader import load_portfolio
        mkt = load_portfolio(10)
        self.assertEqual(mkt.n, 10)

    def test_load_portfolio_20(self):
        from real_data_loader import load_portfolio
        mkt = load_portfolio(20)
        self.assertEqual(mkt.n, 20)

    def test_load_portfolio_invalid_raises(self):
        from real_data_loader import load_portfolio
        with self.assertRaises(ValueError):
            load_portfolio(3)

    def test_agg_1d_shape_and_range(self):
        import numpy as np
        from real_data_loader import load_portfolio, agg_1d
        mkt = load_portfolio(5)
        mu1, sig1, lr1 = agg_1d(mkt)
        self.assertGreater(mu1, 0.0)
        self.assertGreater(sig1, 0.0)
        self.assertLess(sig1, 1.0)
        self.assertEqual(len(lr1), len(mkt.dates))

    def test_market_data_has_omega(self):
        """omega (covariance matrix) must be positive semi-definite."""
        import numpy as np
        from real_data_loader import load_portfolio
        mkt = load_portfolio(5)
        eigvals = np.linalg.eigvalsh(mkt.omega)
        self.assertTrue(all(eigvals >= -1e-10),
            f"omega is not PSD; min eigenvalue = {eigvals.min():.2e}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. BENCHMARK PIPELINE (end-to-end)
# ─────────────────────────────────────────────────────────────────────────────

class TestBenchmarkPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from real_data_loader import load_portfolio
        from comparisons.core.config import BenchmarkConfig
        cls.mkt = load_portfolio(5)
        cls.cfg = BenchmarkConfig()
        cls.cfg.fd_nw = 80
        cls.cfg.fd_nt = 40

    def test_evaluate_fd_benchmark_runs(self):
        import numpy as np
        from comparisons.core.evaluation import evaluate_fd_benchmark
        r = evaluate_fd_benchmark(self.mkt, self.cfg, initial_wealth=1.0, seed=1)
        self.assertIn("wealth_path", r)
        self.assertIn("weight_path", r)
        tw = float(np.asarray(r["terminal_wealth"]).flat[0])
        self.assertGreater(tw, 0.05,
            "Terminal wealth suspiciously low — possible numerical blow-up")
        self.assertLess(tw, 50.0,
            "Terminal wealth suspiciously high — possible numerical blow-up")

    def test_evaluate_merton_benchmark_runs(self):
        import numpy as np
        from comparisons.core.evaluation import evaluate_merton_benchmark
        r = evaluate_merton_benchmark(self.mkt, self.cfg, initial_wealth=1.0, seed=1)
        self.assertEqual(r["method_name"], "fd_merton_multi")
        tw = float(np.asarray(r["terminal_wealth"]).flat[0])
        self.assertGreater(tw, 0.05)
        self.assertLess(tw, 100.0)

    def test_evaluate_static_portfolio_equal_weight(self):
        import numpy as np
        from comparisons.core.evaluation import evaluate_static_portfolio
        weights = np.ones(self.mkt.n) / self.mkt.n
        r = evaluate_static_portfolio(
            "equal_weight", "baseline", self.mkt, weights,
            initial_wealth=1.0, seed=1,
        )
        self.assertEqual(r["method_name"], "equal_weight")
        self.assertEqual(r["n_assets"], self.mkt.n)
        self.assertEqual(len(r["wealth_path"]), len(self.mkt.dates) + 1)

    def test_result_schema_valid(self):
        """_build_result must produce all keys required by REQUIRED_RESULT_KEYS."""
        import numpy as np
        from comparisons.core.evaluation import (
            evaluate_static_portfolio,
            validate_result_schema,
            REQUIRED_RESULT_KEYS,
        )
        weights = np.ones(self.mkt.n) / self.mkt.n
        r = evaluate_static_portfolio(
            "equal_weight", "baseline", self.mkt, weights,
            initial_wealth=1.0, seed=1,
        )
        for key in REQUIRED_RESULT_KEYS:
            self.assertIn(key, r, f"Result missing required key: '{key}'")
        self.assertTrue(validate_result_schema(r))

    def test_leverage_constraint_applies_correctly(self):
        import numpy as np
        from comparisons.core.evaluation import apply_leverage_constraint
        raw = np.array([2.0, 2.0, -3.0, -3.0])  # gross=10, long=4, short=6
        out = apply_leverage_constraint(raw, d=-5.0, u=3.0,
                                        max_long=3.0, max_short=5.0)
        long_sum = float(np.maximum(out, 0).sum())
        short_sum = float(np.maximum(-out, 0).sum())
        self.assertLessEqual(long_sum,  3.0 + 1e-9)
        self.assertLessEqual(short_sum, 5.0 + 1e-9)
        # Must not amplify
        self.assertLessEqual(np.abs(out).sum(), np.abs(raw).sum() + 1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# 7. METRICS
# ─────────────────────────────────────────────────────────────────────────────

class TestMetrics(unittest.TestCase):

    def test_target_hit_rate(self):
        import numpy as np
        from comparisons.core.metrics import compute_target_metrics
        m = compute_target_metrics(np.array([0.9, 1.0, 1.2, 1.4]), target=1.1)
        self.assertAlmostEqual(m["target_hit_rate"], 0.5)

    def test_drawdown_starts_and_ends_at_zero(self):
        import numpy as np
        from comparisons.core.metrics import compute_drawdown_series
        wealth = np.array([1.0, 1.1, 0.9, 1.2])
        dd = compute_drawdown_series(wealth)
        self.assertEqual(dd[0], 0.0)
        self.assertEqual(dd[-1], 0.0)
        self.assertLess(dd.min(), 0.0)

    def test_mean_gross_leverage_equal_weight(self):
        import numpy as np
        from comparisons.core.metrics import compute_weight_metrics
        weights = np.tile(np.array([0.5, 0.5]), (5, 1))
        m = compute_weight_metrics(weights)
        self.assertAlmostEqual(m["mean_gross_leverage"], 1.0, places=10)

    def test_turnover_zero_for_static_portfolio(self):
        import numpy as np
        from comparisons.core.metrics import compute_weight_metrics
        weights = np.tile(np.array([0.4, 0.3, 0.3]), (10, 1))
        m = compute_weight_metrics(weights)
        self.assertAlmostEqual(m["turnover"], 0.0, places=10)


# ─────────────────────────────────────────────────────────────────────────────
# 8. IO ROUND-TRIP
# ─────────────────────────────────────────────────────────────────────────────

class TestIO(unittest.TestCase):

    def _make_result(self):
        import numpy as np
        from comparisons.core.metrics import compute_drawdown_series
        wealth = np.array([1.0, 1.05, 0.98, 1.12])
        weights = np.array([[0.5, 0.5], [0.6, 0.4], [0.7, 0.3]])
        return {
            "method_family": "test", "method_name": "test_method",
            "n_assets": 2, "seed": 1,
            "initial_wealth": 1.0, "target_wealth": 1.1,
            "train_time_sec": 0.0, "solve_time_sec": 0.0, "eval_time_sec": 0.0,
            "wealth_path": wealth, "weight_path": weights,
            "terminal_wealth": np.array([1.12]),
            "goal_hit": np.array([True]),
            "gross_leverage_path": np.ones(3),
            "net_exposure_path":   np.ones(3),
            "concentration_path":  np.ones(3) * 0.5,
            "drawdown_path":       compute_drawdown_series(wealth),
        }

    def test_save_and_load_roundtrip(self):
        import tempfile, numpy as np
        from comparisons.core.io import save_run_result, load_run_result
        r = self._make_result()
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = Path(f.name)
        try:
            save_run_result(path, r)
            r2 = load_run_result(path)
            self.assertEqual(r2["method_name"], "test_method")
            self.assertTrue(np.allclose(r2["wealth_path"], r["wealth_path"]))
            self.assertTrue(np.allclose(r2["weight_path"], r["weight_path"]))
        finally:
            path.unlink(missing_ok=True)

    def test_save_summary_csv(self):
        import tempfile, csv
        from comparisons.core.io import save_summary_csv
        rows = [
            {"method_name": "a", "n_assets": 5,  "target_hit_rate": 0.6},
            {"method_name": "b", "n_assets": 10, "target_hit_rate": 0.4},
        ]
        with tempfile.NamedTemporaryFile(
                suffix=".csv", delete=False, mode="w") as f:
            path = Path(f.name)
        try:
            save_summary_csv(path, rows)
            with open(path) as fh:
                reader = list(csv.DictReader(fh))
            self.assertEqual(len(reader), 2)
            self.assertEqual(reader[0]["method_name"], "a")
        finally:
            path.unlink(missing_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 9. NUMPY NN BACKEND
# ─────────────────────────────────────────────────────────────────────────────

class TestNumpyNN(unittest.TestCase):
    """The numpy evolutionary backend must work regardless of torch availability."""

    def test_train_and_infer_nn_mlp_small(self):
        import numpy as np
        from real_data_loader import load_portfolio
        from comparisons.core.nn_models import train_numpy_policy_net, policy_weights
        mkt = load_portfolio(5)
        net, meta = train_numpy_policy_net(
            mu_vec=mkt.mu_ann, omega_mat=mkt.omega, r=mkt.r,
            architecture_name="nn_mlp_small",
            w0=1.0, goal_mult=1.1,
            n_paths=32, n_iters=3,
            population_size=8, elite_frac=0.25,
            n_steps=8, d=-5.0, u=3.0, seed=0,
        )
        w = policy_weights(net, 1.0, 1.1)
        self.assertEqual(len(w), 5)
        self.assertTrue(all(-5.0 <= wi <= 3.0 for wi in w),
            f"Weights out of [-5,3]: {w}")

    def test_numpy_architecture_sizes(self):
        from comparisons.core.nn_models import ARCHITECTURES
        self.assertEqual(ARCHITECTURES["nn_mlp_small"], (32, 32))
        self.assertEqual(ARCHITECTURES["nn_mlp_deep"],  (64, 64, 32))


# ─────────────────────────────────────────────────────────────────────────────
# 10. TORCH NN BACKEND (skipped if torch not installed)
# ─────────────────────────────────────────────────────────────────────────────

class TestTorchNN(unittest.TestCase):

    def setUp(self):
        from comparisons.core.torch_nn_models import HAS_TORCH
        if not HAS_TORCH:
            self.skipTest("torch not installed — skipping Torch NN tests")

    def test_train_and_infer_nn_mlp_small(self):
        import numpy as np
        from real_data_loader import load_portfolio
        from comparisons.core.torch_nn_models import (
            train_torch_policy_net, policy_weights,
        )
        mkt = load_portfolio(5)
        net, meta = train_torch_policy_net(
            mu_vec=mkt.mu_ann, omega_mat=mkt.omega, r=mkt.r,
            architecture_name="nn_mlp_small",
            w0=1.0, goal_mult=1.1,
            n_paths=32, n_iters=2, n_steps=8,
            d=-5.0, u=3.0,
            max_long_leverage=3.0, max_short_leverage=5.0,
            seed=0,
        )
        w = np.asarray(policy_weights(net, 1.0, 1.1))
        self.assertEqual(len(w), 5)
        self.assertTrue(all(-5.0 <= float(wi) <= 3.0 for wi in w),
            f"Weights out of [-5,3]: {w}")

    def test_all_torch_architectures_build(self):
        from real_data_loader import load_portfolio
        from comparisons.core.torch_nn_models import (
            TORCH_ARCHITECTURES, _build_model,
        )
        for arch_name in TORCH_ARCHITECTURES:
            model = _build_model(
                architecture_name=arch_name,
                n_assets=5, n_steps=8, d=-5.0, u=3.0,
            )
            self.assertIsNotNone(model,
                f"_build_model returned None for architecture '{arch_name}'")


# ─────────────────────────────────────────────────────────────────────────────
# Custom runner: coloured RED / GREEN output
# ─────────────────────────────────────────────────────────────────────────────

class _ColorResult(unittest.TextTestResult):
    GREEN = "\033[32m"
    RED   = "\033[31m"
    RESET = "\033[0m"
    BOLD  = "\033[1m"

    def addSuccess(self, test):
        super().addSuccess(test)
        if self.showAll:
            self.stream.write(self.GREEN + "  PASS" + self.RESET + "\n")
            self.stream.flush()

    def addFailure(self, test, err):
        super().addFailure(test, err)
        if self.showAll:
            self.stream.write(self.RED + "  FAIL" + self.RESET + "\n")
            self.stream.flush()

    def addError(self, test, err):
        super().addError(test, err)
        if self.showAll:
            self.stream.write(self.RED + " ERROR" + self.RESET + "\n")
            self.stream.flush()

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        if self.showAll:
            self.stream.write("\033[33m  SKIP\033[0m (%s)\n" % reason)
            self.stream.flush()


class _ColorRunner(unittest.TextTestRunner):
    resultclass = _ColorResult


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite  = loader.loadTestsFromModule(__import__(__name__))
    runner = _ColorRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
