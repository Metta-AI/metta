
import pathlib
import subprocess
import sys

import pytest

pytestmark = pytest.mark.setup


def test_metta_cli_imports_are_lightweight(tmp_path: pathlib.Path) -> None:
    """Ensure importing the CLI does not pull in heavyweight modules or regress in cost."""

    cmd = [sys.executable, "-X", "importtime", "-c", "import metta.setup.metta_cli"]
    completed = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
    )

    slow_modules = ["torch"]

    for line in completed.stderr.splitlines():
        if any(module in line for module in slow_modules):
            pytest.fail(f"Slow import detected: {line}")
