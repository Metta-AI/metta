import subprocess
import time
from pathlib import Path

import pytest
import requests


class TestMettascopeUI:
    """Test suite that orchestrates the full mettascope UI testing pipeline."""

    @classmethod
    def setup_class(cls):
        """Set up the testing environment: build, train, replay, start server."""
        cls.server_process = None
        cls.mettascope_root = Path(__file__).parent.parent
        cls.project_root = cls.mettascope_root.parent

        # Build Mettascope
        print("Building Mettascope...")
        result = subprocess.run(["./mettascope/install.sh"], cwd=cls.project_root, capture_output=True, text=True)
        if result.returncode != 0:
            pytest.fail(f"Failed to build Mettascope: {result.stderr}")

        # Run quick training job
        print("Running training job...")
        result = subprocess.run(
            [
                "uv",
                "run",
                "tools/train.py",
                "trainer.total_timesteps=10",
                "run=smoke_test",
                "+hardware=github",
                "wandb=off",
            ],
            cwd=cls.project_root,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            pytest.fail(f"Failed to run training: {result.stderr}")

        # Generate replay
        print("Generating replay...")
        result = subprocess.run(
            ["uv", "run", "tools/replay.py", "run=smoke_test", "+hardware=github", "wandb=off"],
            cwd=cls.project_root,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            pytest.fail(f"Failed to generate replay: {result.stderr}")

        # Start the server
        print("Starting server...")
        cls.server_process = subprocess.Popen(
            [
                "uv",
                "run",
                "tools/play.py",
                "run=smoke_test",
                "+hardware=github",
                "wandb=off",
                "replay_job.open_browser_on_start=false",
            ],
            cwd=cls.project_root,
        )

        # Wait for server to be ready
        print("Waiting for server to be ready...")
        for i in range(30):
            try:
                response = requests.get("http://localhost:8000", timeout=2)
                if response.status_code == 200:
                    print("Server is ready!")
                    break
            except requests.exceptions.RequestException:
                pass
            print(f"Attempt {i + 1}: Server not ready yet, waiting...")
            time.sleep(2)
        else:
            if cls.server_process:
                cls.server_process.terminate()
            pytest.fail("Server failed to start within 60 seconds")

    @classmethod
    def teardown_class(cls):
        """Clean up: stop the server."""
        if cls.server_process:
            print("Stopping server...")
            cls.server_process.terminate()
            try:
                cls.server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                cls.server_process.kill()
                cls.server_process.wait()

    def test_frontend_ui(self):
        """Run the frontend UI tests using npm."""
        print("Running frontend tests...")
        result = subprocess.run(["npm", "run", "test"], cwd=self.mettascope_root, capture_output=True, text=True)

        # Print output for debugging
        if result.stdout:
            print("Test output:", result.stdout)
        if result.stderr:
            print("Test errors:", result.stderr)

        assert result.returncode == 0, f"Frontend tests failed: {result.stderr}"
