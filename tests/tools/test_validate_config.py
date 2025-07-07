import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from hydra.core.global_hydra import GlobalHydra

from tools.validate_config import load_and_print_config


class TestValidateConfig(unittest.TestCase):
    """Test the validate_config script functionality."""

    def setUp(self):
        """Clear Hydra state before each test."""
        GlobalHydra.instance().clear()

    def tearDown(self):
        """Clear Hydra state after each test."""
        GlobalHydra.instance().clear()

    def test_load_and_print_config_arena_advanced(self):
        """Test loading env/mettagrid/arena/advanced.yaml config."""
        # Capture the output
        f = StringIO()
        try:
            with redirect_stdout(f):
                load_and_print_config("env/mettagrid/arena/advanced")
            output = f.getvalue()
            # Should have game configuration
            self.assertIn("game:", output)
        except SystemExit as e:
            if e.code != 0:
                self.fail(f"Failed to load env/mettagrid/arena/advanced.yaml: {e}")

    def test_load_and_print_config_agent_fast(self):
        """Test loading agent/fast.yaml config."""
        f = StringIO()
        try:
            with redirect_stdout(f):
                load_and_print_config("agent/fast")
            output = f.getvalue()
            # Should have agent configuration
            self.assertIn("agent:", output)
        except SystemExit as e:
            if e.code != 0:
                self.fail(f"Failed to load agent/fast.yaml: {e}")

    def test_load_and_print_config_trainer(self):
        """Test loading trainer/trainer.yaml config."""
        f = StringIO()
        try:
            with redirect_stdout(f):
                load_and_print_config("trainer/trainer")
            output = f.getvalue()
            # Should have trainer configuration
            self.assertIn("_target_:", output)
        except SystemExit as e:
            if e.code != 0:
                self.fail(f"Failed to load trainer/trainer.yaml: {e}")

    def test_script_execution_arena_advanced(self):
        """Test the script execution via command line for arena/advanced config."""
        result = subprocess.run(
            [sys.executable, "tools/validate_config.py", "env/mettagrid/arena/advanced"],
            cwd=Path(__file__).parent.parent.parent,  # metta root directory
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, f"Script failed: {result.stderr}")
        self.assertIn("game:", result.stdout)  # Simple validation that config was loaded

    def test_script_execution_agent_fast(self):
        """Test the script execution via command line for agent fast config."""
        result = subprocess.run(
            [sys.executable, "tools/validate_config.py", "agent/fast"],
            cwd=Path(__file__).parent.parent.parent,  # metta root directory
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, f"Script failed: {result.stderr}")
        self.assertIn("agent:", result.stdout)  # Simple validation that config was loaded

    def test_script_execution_trainer(self):
        """Test the script execution via command line for trainer config."""
        result = subprocess.run(
            [sys.executable, "tools/validate_config.py", "trainer/trainer"],
            cwd=Path(__file__).parent.parent.parent,  # metta root directory
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, f"Script failed: {result.stderr}")
        # Should contain trainer config elements
        self.assertTrue(
            any(x in result.stdout for x in ["_target_:", "trainer", "batch_size"]),
            f"Expected trainer config content not found in output: {result.stdout[:500]}",
        )

    def test_invalid_config_path(self):
        """Test that invalid config paths raise appropriate errors."""
        with self.assertRaises(SystemExit):
            load_and_print_config("nonexistent/config")


if __name__ == "__main__":
    unittest.main()
