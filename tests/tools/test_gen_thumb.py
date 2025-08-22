import subprocess
import tempfile
from pathlib import Path

import pytest

TESTS_PATH = Path(__file__).parent.parent
REPLAY_PATH = TESTS_PATH.parent / "mettascope" / "replays" / "replay.json.z"
MAP_PATH = TESTS_PATH / "map" / "scenes" / "fixtures" / "test.map"


def run_gen_thumb(file: Path, output: Path):
    cmd = [
        "python",
        "-m",
        "mettascope.tools.gen_thumb",
        "--file",
        str(file),
        "--step",
        "0",
        "--output",
        str(output),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5, cwd=Path.cwd())
        assert result.returncode == 0, f"gen_thumb failed: {result.stderr}"
        assert output.exists(), f"gen_thumb output file {output} does not exist"
    except subprocess.TimeoutExpired:
        pytest.fail("gen_thumb timed out")
    except Exception as e:
        pytest.fail(f"gen_thumb failed: {e}")


class TestGenThumb:
    @pytest.mark.slow
    def test_gen_thumb(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            run_gen_thumb(REPLAY_PATH, tmp / "replay.png")
            run_gen_thumb(MAP_PATH, tmp / "map.png")
