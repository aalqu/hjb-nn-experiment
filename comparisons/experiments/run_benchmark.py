import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from comparisons.core.config import BenchmarkConfig
from comparisons.core.evaluation import run_real_data_portfolio_comparison


if __name__ == "__main__":
    config = BenchmarkConfig(include_nn=True)
    outputs = run_real_data_portfolio_comparison(config)
    print(f"Completed {len(outputs['results'])} benchmark runs")
    print(f"Saved summary to {config.results_dir / 'summary' / 'main_results.csv'}")
