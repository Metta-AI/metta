import os
import random
import shutil
import string
import subprocess

import pytest


@pytest.mark.slow
def test_basic():
    run_name = "test_metta_script_" + "".join(random.choices(string.ascii_letters + string.digits, k=10))

    result = subprocess.check_output(["./tests/util/fixtures/script.py", f"run={run_name}"], text=True)

    # logging from script
    assert "Hello, world!" in result

    # logging from metta_script
    assert "Environment setup completed" in result

    # logging is configured correctly
    assert "script.py:11" in result

    # check file log
    with open(f"train_dir/{run_name}/logs/script.log", "r") as f:
        content = f.read()
        assert "Environment setup completed" in content
        # logging format in file is different from stdout
        assert "test_script: Hello, world!" in content

    # clean up - comment out for debugging
    if os.path.exists(f"train_dir/{run_name}"):
        shutil.rmtree(f"train_dir/{run_name}")
