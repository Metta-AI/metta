"""
Tests for the codeclip CLI functionality.
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from codebot.codeclip.cli import cli


class TestCodeclipCLI(unittest.TestCase):
    """Test cases for the codeclip CLI."""

    def setUp(self):
        self.runner = CliRunner()
        self.test_dir = tempfile.TemporaryDirectory()
        self.original_cwd = os.getcwd()
        os.chdir(self.test_dir.name)

    def tearDown(self):
        os.chdir(self.original_cwd)
        self.test_dir.cleanup()

    def _create_test_structure(self):
        """Create a test directory structure."""
        # Create test directories and files
        dirs = {
            "project1": {
                "README.md": "# Project 1\n",
                "main.py": "print('hello from project1')\n",
                "lib.py": "def helper(): pass\n",
                ".gitignore": "*.pyc\n__pycache__/\n",
            },
            "project2": {
                "README.md": "# Project 2\n",
                "app.py": "print('hello from project2')\n",
                "utils.py": "def util(): pass\n",
            },
            "shared": {"config.py": "CONFIG = {}\n", "helpers.py": "def help(): pass\n"},
        }

        for dir_name, files in dirs.items():
            dir_path = Path(self.test_dir.name) / dir_name
            dir_path.mkdir(parents=True, exist_ok=True)
            for file_name, content in files.items():
                file_path = dir_path / file_name
                file_path.write_text(content)

    def test_multiple_directory_inputs(self):
        """Test that multiple directory inputs are handled correctly."""
        self._create_test_structure()

        # Test with multiple directories (using -s for stdout output)
        result = self.runner.invoke(cli, ["project1", "project2", "-s"])
        self.assertEqual(result.exit_code, 0)

        # Check that files from both directories are included
        self.assertIn("project1/main.py", result.output)
        self.assertIn("project1/lib.py", result.output)
        self.assertIn("project2/app.py", result.output)
        self.assertIn("project2/utils.py", result.output)

    def test_pwd_relative_paths(self):
        """Test that paths are resolved relative to pwd."""
        self._create_test_structure()

        # Create a subdirectory and change to it
        subdir = Path(self.test_dir.name) / "subdir"
        subdir.mkdir()
        os.chdir(subdir)

        # Test with relative path (using -s for stdout output)
        result = self.runner.invoke(cli, ["../project1", "-s"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("project1/main.py", result.output)

    def test_process_current_directory(self):
        """Test processing current directory with explicit '.'."""
        # Create files in current directory
        Path("test.py").write_text("print('test')\n")
        Path("README.md").write_text("# Test\n")

        result = self.runner.invoke(cli, [".", "-s"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("test.py", result.output)
        self.assertIn("README.md", result.output)

    def test_extension_filtering(self):
        """Test file extension filtering."""
        self._create_test_structure()

        # Test filtering for Python files only
        result = self.runner.invoke(cli, ["-e", "py", "project1", "project2", "-s"])
        self.assertEqual(result.exit_code, 0)

        # Should include .py files
        self.assertIn("main.py", result.output)
        self.assertIn("app.py", result.output)

        # README.md files are always included even with extension filtering
        self.assertIn("README.md", result.output)

    def test_readmes_only_flag(self):
        """Test readmes-only flag (-r)."""
        Path("test.py").write_text("print('test')\n")
        Path("README.md").write_text("# Test Project\n")

        # Test with readmes-only flag (using -s for stdout output)
        result = self.runner.invoke(cli, [".", "-r", "-s"])
        self.assertEqual(result.exit_code, 0)

        # Should have XML format (we always use XML now)
        self.assertIn("<document index=", result.output)
        self.assertIn("<source>", result.output)

        # Should include README.md
        self.assertIn("README.md", result.output)
        self.assertIn("Test Project", result.output)

        # Should NOT include test.py (filtered out by readmes-only)
        self.assertNotIn("test.py", result.output)
        self.assertNotIn("print('test')", result.output)

    def test_xml_output_format(self):
        """Test XML output format (default)."""
        Path("test.py").write_text("print('test')\n")

        result = self.runner.invoke(cli, [".", "-s"])
        self.assertEqual(result.exit_code, 0)

        # Should have XML tags (new format)
        self.assertIn("<document index=", result.output)
        self.assertIn("<source>", result.output)
        self.assertIn("</document>", result.output)
        self.assertIn("<document_content>", result.output)

    def test_parent_readme_inclusion(self):
        """Test that parent README files are included."""
        # Create a parent README
        Path("README.md").write_text("# Parent README\n")

        # Create a subdirectory with files
        subdir = Path("subdir")
        subdir.mkdir()
        (subdir / "code.py").write_text("print('code')\n")

        result = self.runner.invoke(cli, ["subdir", "-s"])
        self.assertEqual(result.exit_code, 0)

        # Should include both the subdirectory file and parent README
        self.assertIn("subdir/code.py", result.output)
        self.assertIn("README.md", result.output)
        self.assertIn("Parent README", result.output)

    @patch("codebot.codeclip.cli.copy_to_clipboard")
    def test_clipboard_default(self, mock_copy):
        """Test clipboard integration on macOS (default behavior)."""
        Path("test.py").write_text("print('test')\n")

        # Clipboard is now the default behavior (no flags needed)
        result = self.runner.invoke(cli, ["."])
        self.assertEqual(result.exit_code, 0)

        # Check that clipboard copy was called
        mock_copy.assert_called_once()

        # Should see summary in stderr
        self.assertIn("Copied", result.output)
        self.assertIn("tokens", result.output)

    def test_stdout_flag(self):
        """Test that -s flag outputs to stdout."""
        Path("test.py").write_text("print('test')\n")

        result = self.runner.invoke(cli, [".", "-s"])
        self.assertEqual(result.exit_code, 0)

        # Should have file content in stdout
        self.assertIn("print('test')", result.output)
        # Should NOT have summary since -s was used
        self.assertNotIn("Copied", result.output)

    def test_help_contains_all_options(self):
        """Test that help output contains all expected options."""
        result = self.runner.invoke(cli, ["--help"])
        self.assertEqual(result.exit_code, 0)

        # Check that all options are mentioned in help
        self.assertIn("-s, --stdout", result.output)
        self.assertIn("-r, --readmes", result.output)
        self.assertIn("-e, --extension", result.output)
        self.assertIn("-p, --profile", result.output)
        self.assertIn("-f, --flamegraph", result.output)
        self.assertIn("-d, --diff", result.output)


if __name__ == "__main__":
    unittest.main()
