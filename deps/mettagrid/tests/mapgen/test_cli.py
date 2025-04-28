import os
import subprocess


def test_gen_basic():
    subprocess.check_call(
        [
            "python",
            "-m",
            "tools.map.gen",
            "--show-mode",
            "ascii",
            "./configs/game/map_builder/mapgen_simple.yaml",
        ]
    )


def test_gen_missing_config():
    exit_status = subprocess.call(
        [
            "python",
            "-m",
            "tools.map.gen",
            "--show-mode",
            "ascii",
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
            "--show-mode",
            "ascii",
            "./configs/game/map_builder/mapgen_simple.yaml",
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
            "--show-mode",
            "ascii",
            "./configs/game/map_builder/mapgen_simple.yaml",
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
            "--show-mode",
            "none",
            "./configs/game/map_builder/mapgen_maze.yaml",
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
            "--show-mode",
            "none",
            "./configs/game/map_builder/mapgen_simple.yaml",
        ]
    )
    subprocess.check_call(
        [
            "python",
            "-m",
            "tools.map.view",
            "--show-mode",
            "ascii",
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
            "--show-mode",
            "none",
            "--count",
            "3",
            "./configs/game/map_builder/mapgen_simple.yaml",
        ],
    )
    view_output = subprocess.check_output(
        ["python", "-m", "tools.map.view", "--show-mode", "ascii", tmpdir], stderr=subprocess.STDOUT, text=True
    )
    assert "Loading random map" in view_output
