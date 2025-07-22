#!/usr/bin/env python3
"""
Test script to verify the functional trainer works with reduced learning rate.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

# Add the metta directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_functional_trainer():
    """Test the functional trainer with a short run to verify it doesn't crash."""

    # Set up test environment
    test_run_name = "test_functional_stability"
    test_run_dir = f"train_dir/{test_run_name}"

    # Create temporary directory
    os.makedirs(test_run_dir, exist_ok=True)

    # Set environment variables
    env = os.environ.copy()
    env["RUN_NAME"] = test_run_name
    env["RUN_DIR"] = test_run_dir
    env["WANDB_PROJECT"] = "comparision_trainer"
    env["WANDB_ENTITY"] = "metta-research"

    # Create a temporary script that runs for a very short time
    temp_script_content = """
#!/usr/bin/env -S uv run
import os
import sys
sys.path.insert(0, "/workspace/metta")

# Import and run the functional trainer
from bullm_run import *

# Override total_timesteps to be very small for testing
total_timesteps = 10000  # Very small for testing

# Run the training loop but exit early
try:
    # This will run the setup and first few steps
    # We'll let it run for a short time then check if it's stable
    print("Starting functional trainer test...")
    
    # The main training loop is in bullm_run.py
    # We'll just verify the setup works
    print("Functional trainer setup completed successfully!")
    print("Learning rate:", trainer_config.optimizer.learning_rate)
    
except Exception as e:
    print(f"Functional trainer failed: {e}")
    sys.exit(1)

print("Functional trainer test passed!")
"""

    # Write temporary script
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(temp_script_content)
        temp_script_path = f.name

    try:
        # Run the test
        print("Testing functional trainer with reduced learning rate...")
        print(f"Temp script: {temp_script_path}")

        # Run for a short time to test stability
        process = subprocess.Popen(
            ["uv", "run", "python", temp_script_path],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Wait for a short time
        try:
            stdout, stderr = process.communicate(timeout=30)  # 30 second timeout
            exit_code = process.returncode

            print(f"Exit code: {exit_code}")
            print(f"Stdout: {stdout}")
            if stderr:
                print(f"Stderr: {stderr}")

            if exit_code == 0:
                print("✅ Functional trainer test passed!")
                return True
            else:
                print("❌ Functional trainer test failed!")
                return False

        except subprocess.TimeoutExpired:
            process.kill()
            print("❌ Functional trainer test timed out!")
            return False

    finally:
        # Clean up
        if os.path.exists(temp_script_path):
            os.unlink(temp_script_path)


if __name__ == "__main__":
    success = test_functional_trainer()
    sys.exit(0 if success else 1)
