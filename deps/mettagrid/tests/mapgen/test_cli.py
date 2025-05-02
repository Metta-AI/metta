import os
import subprocess

from mettagrid.config.utils import mettagrid_configs_root

map_root = str(mettagrid_configs_root / "maps")


def test_gen_basic():
    subprocess.check_call(
        [
            "python",
            "-m",
            "tools.map.gen",
            f"{map_root}/maze.yaml",
        ]
    )


def test_gen_missing_config():
    exit_status = subprocess.call(
        [
            "python",
            "-m",
            "tools.map.gen",
            "./NON_EXISTENT_CONFIG.yaml",
        ]
    )
    assert exit_status != 0


def test_save(tmpdir):
    subprocess.check_call(
        [
            "python",
            "-m",
            "tools.map.gen",
            "--output-uri",
            tmpdir,
            f"{map_root}/maze.yaml",
        ]
    )
    files = os.listdir(tmpdir)
    assert len(files) == 1
    assert files[0].endswith(".yaml")


def test_save_exact_file(tmpdir):
    subprocess.check_call(
        [
            "python",
            "-m",
            "tools.map.gen",
            "--output-uri",
            tmpdir + "/map.yaml",
            f"{map_root}/maze.yaml",
        ]
    )
    files = os.listdir(tmpdir)
    assert len(files) == 1
    assert files[0] == "map.yaml"


def test_save_multiple(tmpdir):
    count = 3
    subprocess.check_call(
        [
            "python",
            "-m",
            "tools.map.gen",
            "--output-uri",
            tmpdir,
            f"{map_root}/maze.yaml",
            "--count",
            str(count),
        ]
    )
    files = os.listdir(tmpdir)
    assert len(files) == count
    for file in files:
        assert file.endswith(".yaml")


def test_view(tmpdir):
    subprocess.check_call(
        [
            "python",
            "-m",
            "tools.map.gen",
            "--output-uri",
            tmpdir + "/map.yaml",
            f"{map_root}/maze.yaml",
        ]
    )
    subprocess.check_call(
        [
            "python",
            "-m",
            "tools.map.view",
            tmpdir + "/map.yaml",
        ]
    )


def test_view_random(tmpdir):
    subprocess.check_call(
        [
            "python",
            "-m",
            "tools.map.gen",
            "--output-uri",
            tmpdir,
            "--count",
            "3",
            f"{map_root}/maze.yaml",
        ],
    )
    view_output = subprocess.check_output(
        ["python", "-m", "tools.map.view", tmpdir], stderr=subprocess.STDOUT, text=True
    )
    assert "Loading random map" in view_output
