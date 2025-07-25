"""
Tests for the codeflow CLI functionality.
"""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner
from codeflow.cli import cli


class TestCodeflowCLI(unittest.TestCase):
    """Test cases for the codeflow CLI."""

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

        # Test with multiple directories
        result = self.runner.invoke(cli, ["project1", "project2"])
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

        # Test with relative path
        result = self.runner.invoke(cli, ["../project1"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("project1/main.py", result.output)

    def test_default_current_directory(self):
        """Test that no arguments defaults to current directory."""
        # Create files in current directory
        Path("test.py").write_text("print('test')\n")
        Path("README.md").write_text("# Test\n")

        result = self.runner.invoke(cli, [])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("test.py", result.output)
        self.assertIn("README.md", result.output)

    def test_extension_filtering(self):
        """Test file extension filtering."""
        self._create_test_structure()

        # Test filtering for Python files only
        result = self.runner.invoke(cli, ["-e", "py", "project1", "project2"])
        self.assertEqual(result.exit_code, 0)

        # Should include .py files
        self.assertIn("main.py", result.output)
        self.assertIn("app.py", result.output)

        # README.md files are always included even with extension filtering
        self.assertIn("README.md", result.output)

    def test_raw_output_format(self):
        """Test raw output format."""
        Path("test.py").write_text("print('test')\n")

        # Test with raw format
        result = self.runner.invoke(cli, ["-r"])
        self.assertEqual(result.exit_code, 0)

        # Should not have XML tags
        self.assertNotIn("<file path=", result.output)
        # Should have file content
        self.assertIn("print('test')", result.output)

    def test_xml_output_format(self):
        """Test XML output format (default)."""
        Path("test.py").write_text("print('test')\n")

        result = self.runner.invoke(cli, [])
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

        result = self.runner.invoke(cli, ["subdir"])
        self.assertEqual(result.exit_code, 0)

        # Should include both the subdirectory file and parent README
        self.assertIn("subdir/code.py", result.output)
        self.assertIn("README.md", result.output)
        self.assertIn("Parent README", result.output)

    @patch("subprocess.Popen")
    def test_pbcopy_integration(self, mock_popen):
        """Test clipboard integration on macOS."""
        Path("test.py").write_text("print('test')\n")

        # Mock the process and communicate method
        mock_process = MagicMock()
        mock_process.communicate = MagicMock()
        mock_popen.return_value = mock_process

        result = self.runner.invoke(cli, ["-p"])
        self.assertEqual(result.exit_code, 0)

        # Check that pbcopy was called
        mock_popen.assert_called_once_with(["pbcopy"], stdin=subprocess.PIPE)
        mock_process.communicate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
