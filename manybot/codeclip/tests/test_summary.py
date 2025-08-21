"""Test the summary output functionality of codeclip."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner
from codeclip.cli import cli


class TestCodeclipSummary(unittest.TestCase):
    """Test suite for codeclip summary output."""

    def setUp(self):
        """Set up test environment."""
        self.runner = CliRunner()
        self.original_dir = os.getcwd()
        self.test_dir = tempfile.mkdtemp()
        os.chdir(self.test_dir)

    def tearDown(self):
        """Clean up test environment."""
        os.chdir(self.original_dir)
        import shutil

        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_no_args_defaults_to_current_directory(self):
        """Test that running codeclip with no arguments defaults to current directory."""
        # Create test files
        Path("test1.py").write_text("# Test file 1\nprint('hello')")
        Path("test2.py").write_text("# Test file 2\nprint('world')")

        # Mock clipboard
        with patch("subprocess.Popen") as mock_popen:
            mock_process = MagicMock()
            mock_process.communicate = MagicMock(return_value=(None, None))
            mock_popen.return_value = mock_process

            result = self.runner.invoke(cli, [])
            self.assertEqual(result.exit_code, 0)

            # Should have copied files (not shown help)
            self.assertIn("Copied", result.output)
            self.assertIn("tokens from", result.output)
            self.assertIn("files", result.output)

            # pbcopy should have been called
            mock_popen.assert_called_once()

    def test_single_directory_shows_subdirectory_breakdown(self):
        """Test that providing a single directory shows breakdown of its contents."""
        # Create directory structure
        os.makedirs("mydir/subdir1")
        os.makedirs("mydir/subdir2")
        Path("mydir/file1.py").write_text("# File in root\n" + "x = 1\n" * 100)
        Path("mydir/subdir1/file2.py").write_text("# File in subdir1\n" + "y = 2\n" * 50)
        Path("mydir/subdir2/file3.py").write_text("# File in subdir2\n" + "z = 3\n" * 75)

        # Mock clipboard
        with patch("subprocess.Popen") as mock_popen:
            mock_process = MagicMock()
            mock_process.communicate = MagicMock(return_value=(None, None))
            mock_popen.return_value = mock_process

            result = self.runner.invoke(cli, ["mydir"])
            self.assertEqual(result.exit_code, 0)

            # Should show breakdown of subdirectories/files
            self.assertIn("Top items:", result.output)
            # Should show aggregated by immediate children
            self.assertIn("mydir/file1.py", result.output)  # Direct file
            self.assertIn("mydir/subdir1", result.output)  # Directory aggregation
            self.assertIn("mydir/subdir2", result.output)  # Directory aggregation
            # Should NOT show deep files individually
            self.assertNotIn("mydir/subdir1/file2.py", result.output)

    def test_single_file_shows_just_that_file(self):
        """Test that providing a single file shows just that file in summary."""
        Path("single.py").write_text("# Single file\nprint('test')")

        with patch("subprocess.Popen") as mock_popen:
            mock_process = MagicMock()
            mock_process.communicate = MagicMock(return_value=(None, None))
            mock_popen.return_value = mock_process

            result = self.runner.invoke(cli, ["single.py"])
            self.assertEqual(result.exit_code, 0)

            # Should show the file was copied
            self.assertIn("Copied", result.output)
            self.assertIn("1 file", result.output)

    def test_multiple_paths_shows_path_breakdown(self):
        """Test that providing multiple paths shows breakdown by path."""
        # Create multiple directories
        os.makedirs("dir1")
        os.makedirs("dir2")
        Path("dir1/file1.py").write_text("# Dir1 file\n" + "a = 1\n" * 100)
        Path("dir2/file2.py").write_text("# Dir2 file\n" + "b = 2\n" * 50)

        with patch("subprocess.Popen") as mock_popen:
            mock_process = MagicMock()
            mock_process.communicate = MagicMock(return_value=(None, None))
            mock_popen.return_value = mock_process

            result = self.runner.invoke(cli, ["dir1", "dir2"])
            self.assertEqual(result.exit_code, 0)

            # Should show breakdown by path
            self.assertIn("By path:", result.output)
            self.assertIn("dir1:", result.output)
            self.assertIn("dir2:", result.output)
            self.assertIn("tokens", result.output)

    def test_deep_directory_aggregates_by_one_level(self):
        """Test that deep directory structures aggregate by immediate children only."""
        # Create deep directory structure
        os.makedirs("root/level1/level2/level3")
        Path("root/file_at_root.py").write_text("# Root file\n" + "a = 1\n" * 10)
        Path("root/level1/file_at_l1.py").write_text("# L1 file\n" + "b = 2\n" * 20)
        Path("root/level1/level2/file_at_l2.py").write_text("# L2 file\n" + "c = 3\n" * 30)
        Path("root/level1/level2/level3/file_at_l3.py").write_text("# L3 file\n" + "d = 4\n" * 40)

        with patch("subprocess.Popen") as mock_popen:
            mock_process = MagicMock()
            mock_process.communicate = MagicMock(return_value=(None, None))
            mock_popen.return_value = mock_process

            result = self.runner.invoke(cli, ["root"])
            self.assertEqual(result.exit_code, 0)

            # Should show only immediate children
            self.assertIn("Top items:", result.output)
            self.assertIn("root/level1", result.output)  # All files under level1 aggregated
            self.assertIn("root/file_at_root.py", result.output)  # Direct file

            # Should NOT show deeper paths
            self.assertNotIn("level1/level2", result.output)
            self.assertNotIn("level2/level3", result.output)
            self.assertNotIn("file_at_l2.py", result.output)
            self.assertNotIn("file_at_l3.py", result.output)


if __name__ == "__main__":
    unittest.main()
