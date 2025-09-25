import os
import subprocess
import tempfile
from pathlib import Path

import pytest

from metta.common.util.fs import get_repo_root

repo_root = get_repo_root()
REPLAY_PATH = f"{repo_root}/mettascope/replays/replay.json.z"
MAP_PATH = f"{repo_root}/packages/mettagrid/tests/mapgen/scenes/fixtures/test.map"


def run_gen_thumb(file: str, output: Path):
    cmd = [
        "uv",
        "run",
        "python",
        "-m",
        "mettascope.tools.gen_thumb",
        "--file",
        file,
        "--step",
        "0",
        "--output",
        str(output),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=20, cwd=Path.cwd())
        assert result.returncode == 0, f"gen_thumb failed: {result.stderr}"
        assert output.exists(), f"gen_thumb output file {output} does not exist"
    except subprocess.TimeoutExpired:
        pytest.fail("gen_thumb timed out")
    except Exception as e:
        pytest.fail(f"gen_thumb failed: {e}")


class TestGenThumb:
    @pytest.mark.slow
    @pytest.mark.skipif(os.environ.get("CI") == "true", reason="Flaky timeout issues in CI")
    def test_gen_thumb(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            run_gen_thumb(REPLAY_PATH, tmp / "replay.png")
            run_gen_thumb(MAP_PATH, tmp / "map.png")
