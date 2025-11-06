"""Test the summary output functionality of codeclip."""

import os
import pathlib
import shutil
import tempfile
import unittest

import typer.testing

import codebot.codeclip.cli


class TestCodeclipSummary(unittest.TestCase):
    """Test suite for codeclip summary output - simplified tests that don't mock subprocess internals."""

    def setUp(self):
        """Set up test environment."""
        self.runner = typer.testing.CliRunner()
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
        pathlib.Path("test1.py").write_text("# Test file 1\nprint('hello')")
        pathlib.Path("test2.py").write_text("# Test file 2\nprint('world')")

        # Run with stdout flag
        result = self.runner.invoke(codebot.codeclip.cli.app, [".", "-s"])
        self.assertEqual(result.exit_code, 0)

        # Should see the document output
        self.assertIn("<documents>", result.output)
        self.assertIn("</documents>", result.output)
        self.assertIn("test1.py", result.output)
        self.assertIn("test2.py", result.output)

    def test_extensions_filter_limits_files(self):
        """Test that extension filters work correctly."""
        # Create files with different extensions
        pathlib.Path("code.py").write_text("# Python file")
        pathlib.Path("doc.md").write_text("# Markdown file")
        pathlib.Path("style.css").write_text("/* CSS file */")

        # Test with Python files only
        result = self.runner.invoke(codebot.codeclip.cli.app, [".", "-s", "-e", "py"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("code.py", result.output)
        self.assertNotIn("doc.md", result.output)
        self.assertNotIn("style.css", result.output)

    def test_readmes_only_flag(self):
        """Test that --readmes flag only includes README files."""
        # Create various files
        pathlib.Path("README.md").write_text("# Main readme")
        pathlib.Path("code.py").write_text("# Code file")
        os.makedirs("subdir")
        pathlib.Path("subdir/README.md").write_text("# Sub readme")
        pathlib.Path("subdir/more_code.py").write_text("# More code")

        # Test with readmes only
        result = self.runner.invoke(codebot.codeclip.cli.app, [".", "-s", "--readmes"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("README.md", result.output)
        self.assertIn("subdir/README.md", result.output)
        self.assertNotIn("code.py", result.output)
        self.assertNotIn("more_code.py", result.output)

    def test_help_flag_shows_help(self):
        """Test that --help flag shows help text."""
        result = self.runner.invoke(codebot.codeclip.cli.app, ["--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Usage:", result.output)
        self.assertIn("Options", result.output)  # Typer uses "Options" in a box, not "Options:"


if __name__ == "__main__":
    unittest.main()
