import os
import random
import string
import subprocess

import pytest


@pytest.mark.slow
def test_basic():
    os.environ["COLUMNS"] = "200"
    run_name = "test_metta_script_" + "".join(random.choices(string.ascii_letters + string.digits, k=10))

    result = subprocess.check_output(["./tests/util/fixtures/script.py", f"run={run_name}"], text=True)

    # logging from script
    assert "Hello, world!" in result
    assert f"Run: {run_name}" in result

    # logging is configured correctly
    assert "script.py:14" in result

    # script name detection
    assert "Running tests/util/fixtures/script.py with config" in result

    # config is logged
    assert '"run": "test_metta_script_' in result
