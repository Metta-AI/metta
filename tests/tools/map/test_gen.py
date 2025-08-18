import os
import subprocess
from pathlib import Path

import pytest

MAZE_CONFIG = Path(__file__).parent / "maze.yaml"
MAP_MODULE = "tools.map"


@pytest.mark.slow
class TestMapGen:
    def test_gen_basic(self):
        subprocess.check_call(
            [
                "python",
                "-m",
                f"{MAP_MODULE}.gen",
                "--show-mode",
                "ascii",
                MAZE_CONFIG,
            ]
        )

    def test_gen_missing_config(self):
        exit_status = subprocess.call(
            [
                "python",
                "-m",
                f"{MAP_MODULE}.gen",
                "--show-mode",
                "ascii",
                "./NON_EXISTENT_CONFIG.yaml",
            ]
        )
        assert exit_status != 0

    def test_hydra(self):
        subprocess.check_call(
            [
                "python",
                "-m",
                f"{MAP_MODULE}.gen",
                "--show-mode",
                "ascii",
                "./configs/env/mettagrid/puffer.yaml",
            ]
        )

    def test_save(self, tmpdir):
        subprocess.check_call(
            [
                "python",
                "-m",
                f"{MAP_MODULE}.gen",
                "--output-uri",
                tmpdir,
                "--show-mode",
                "ascii",
                MAZE_CONFIG,
            ]
        )
        files = os.listdir(tmpdir)
        assert len(files) == 1
        assert files[0].endswith(".yaml")

    def test_save_exact_file(self, tmpdir):
        subprocess.check_call(
            [
                "python",
                "-m",
                f"{MAP_MODULE}.gen",
                "--output-uri",
                tmpdir + "/map.yaml",
                "--show-mode",
                "ascii",
                MAZE_CONFIG,
            ]
        )
        files = os.listdir(tmpdir)
        assert len(files) == 1
        assert files[0] == "map.yaml"

    def test_save_multiple(self, tmpdir):
        count = 3
        subprocess.check_call(
            [
                "python",
                "-m",
                f"{MAP_MODULE}.gen",
                "--output-uri",
                tmpdir,
                "--show-mode",
                "none",
                MAZE_CONFIG,
                "--count",
                str(count),
            ]
        )
        files = os.listdir(tmpdir)
        assert len(files) == count
        for file in files:
            assert file.endswith(".yaml")

    def test_view(self, tmpdir):
        subprocess.check_call(
            [
                "python",
                "-m",
                f"{MAP_MODULE}.gen",
                "--output-uri",
                tmpdir + "/map.yaml",
                "--show-mode",
                "none",
                MAZE_CONFIG,
            ]
        )
        subprocess.check_call(
            [
                "python",
                "-m",
                f"{MAP_MODULE}.view",
                "--show-mode",
                "ascii",
                tmpdir + "/map.yaml",
            ]
        )

    def test_view_random(self, tmpdir):
        subprocess.check_call(
            [
                "python",
                "-m",
                f"{MAP_MODULE}.gen",
                "--output-uri",
                tmpdir,
                "--show-mode",
                "none",
                "--count",
                "3",
                MAZE_CONFIG,
            ],
        )
        view_output = subprocess.check_output(
            ["python", "-m", f"{MAP_MODULE}.view", "--show-mode", "ascii", tmpdir],
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert "Loading random map" in view_output
