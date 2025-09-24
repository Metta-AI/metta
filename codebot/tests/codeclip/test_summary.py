"""Test the summary output functionality of codeclip."""

import os
import shutil
import tempfile
import unittest
from pathlib import Path

from typer.testing import CliRunner

from codebot.codeclip.cli import app


class TestCodeclipSummary(unittest.TestCase):
    """Test suite for codeclip summary output - simplified tests that don't mock subprocess internals."""

    def setUp(self):
        """Set up test environment."""
        self.runner = CliRunner()
        self.original_dir = os.getcwd()
        # Create a temporary directory for tests
        self.test_dir = tempfile.mkdtemp()
        os.chdir(self.test_dir)

    def tearDown(self):
        """Clean up test environment."""
        os.chdir(self.original_dir)
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_stdout_flag_outputs_to_stdout(self):
        """Test that -s flag outputs to stdout instead of clipboard."""
        # Create test files
        Path("test1.py").write_text("# Test file 1\nprint('hello')")
        Path("test2.py").write_text("# Test file 2\nprint('world')")

        # Run with stdout flag
        result = self.runner.invoke(app, [".", "-s"])
        self.assertEqual(result.exit_code, 0)

        # Should see the document output
        self.assertIn("<documents>", result.output)
        self.assertIn("</documents>", result.output)
        self.assertIn("test1.py", result.output)
        self.assertIn("test2.py", result.output)

    def test_extensions_filter_limits_files(self):
        """Test that extension filters work correctly."""
        # Create files with different extensions
        Path("code.py").write_text("# Python file")
        Path("doc.md").write_text("# Markdown file")
        Path("style.css").write_text("/* CSS file */")

        # Test with Python files only
        result = self.runner.invoke(app, [".", "-s", "-e", "py"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("code.py", result.output)
        self.assertNotIn("doc.md", result.output)
        self.assertNotIn("style.css", result.output)

    def test_readmes_only_flag(self):
        """Test that --readmes flag only includes README files."""
        # Create various files
        Path("README.md").write_text("# Main readme")
        Path("code.py").write_text("# Code file")
        os.makedirs("subdir")
        Path("subdir/README.md").write_text("# Sub readme")
        Path("subdir/more_code.py").write_text("# More code")

        # Test with readmes only
        result = self.runner.invoke(app, [".", "-s", "--readmes"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("README.md", result.output)
        self.assertIn("subdir/README.md", result.output)
        self.assertNotIn("code.py", result.output)
        self.assertNotIn("more_code.py", result.output)

    def test_help_flag_shows_help(self):
        """Test that --help flag shows help text."""
        result = self.runner.invoke(app, ["--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Usage:", result.output)
        self.assertIn("Options", result.output)  # Typer uses "Options" in a box, not "Options:"


if __name__ == "__main__":
    unittest.main()
