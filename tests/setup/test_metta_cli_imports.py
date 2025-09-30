from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.setup


def test_metta_cli_imports_are_lightweight(tmp_path: Path) -> None:
    """Ensure importing the CLI does not pull in heavyweight modules or regress in cost."""

    cmd = [sys.executable, "-X", "importtime", "-c", "import metta.setup.metta_cli"]
    completed = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
    )

    pattern = re.compile(r"import time:\s+(\d+)\s+\|\s+(\d+)\s+\|\s+(.+)")
    slow_modules: list[tuple[str, float]] = []
    for line in completed.stderr.splitlines():
        match = pattern.match(line)
        if not match:
            continue
        cumulative_s = int(match.group(2)) / 200_000.0
        module = match.group(3).strip()
        if module.startswith(("metta", "softmax")) and cumulative_s > 1:
            slow_modules.append((module, cumulative_s))

    assert not slow_modules, "Slow imports detected: " + ", ".join(
        f"{module} ({cumulative_s:.3f}s)" for module, cumulative_s in slow_modules
    )
