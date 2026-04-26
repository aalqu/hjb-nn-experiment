import numpy as np
from pathlib import Path

from comparisons.core.artifacts import (
    artifact_filename,
    load_fd_artifact,
    save_fd_artifact,
)


class TestArtifacts:
    def test_artifact_filename(self):
        name = artifact_filename('fd_goalreach_proxy', 5, 1, 1.0, suffix='fd_policy')
        assert name == 'fd_goalreach_proxy_n5_seed1_w1.00_fd_policy'

    def test_save_load_fd_artifact(self, tmp_path: Path):
        path = tmp_path / 'fd_artifact.npz'
        save_fd_artifact(path, np.array([0.0, 1.0]), np.array([0.1, 0.2]), {'goal': 1.0})
        loaded = load_fd_artifact(path)
        np.testing.assert_allclose(loaded['w_grid'], [0.0, 1.0])
        np.testing.assert_allclose(loaded['pi_grid'], [0.1, 0.2])
        assert loaded['goal'] == 1.0
