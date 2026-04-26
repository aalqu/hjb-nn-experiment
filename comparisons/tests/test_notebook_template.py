import json
from pathlib import Path


def _load_notebook(path_str):
    path = Path(path_str)
    assert path.exists(), f'Missing notebook: {path}'
    return json.loads(path.read_text())


def _all_source(nb):
    return '\n'.join(''.join(cell.get('source', [])) for cell in nb.get('cells', []))


def _first_code_source(nb):
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            return ''.join(cell.get('source', []))
    return ''


class TestNotebookTemplates:
    def test_results_explorer_exists_and_references_all_models(self):
        nb = _load_notebook('comparisons/notebooks/01_results_explorer.ipynb')
        sources = _all_source(nb)
        assert 'load_fd_policy_bundle' in sources
        assert 'load_nn_model_bundle' in sources
        for name in ['nn_mlp_small', 'nn_mlp_deep', 'deep_bsde', 'pinn', 'actor_critic', 'lstm', 'transformer']:
            assert name in sources

    def test_second_notebook_exists(self):
        nb = _load_notebook('comparisons/notebooks/02_playground.ipynb')
        sources = _all_source(nb)
        assert 'BenchmarkConfig' in sources
        assert 'run_real_data_portfolio_comparison' in sources
        assert 'NN_METHODS' in sources

    def test_notebooks_have_robust_repo_root_bootstrap(self):
        for path in [
            'comparisons/notebooks/01_results_explorer.ipynb',
            'comparisons/notebooks/02_playground.ipynb',
        ]:
            nb = _load_notebook(path)
            first_code = _first_code_source(nb)
            assert "while not (ROOT / 'comparisons').exists()" in first_code
            assert "sys.path.insert(0, str(ROOT))" in first_code
