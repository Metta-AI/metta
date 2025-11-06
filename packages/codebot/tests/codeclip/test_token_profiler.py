"""
Tests for token profiler functionality including common prefix detection.
"""

import pathlib
import unittest
import unittest.mock

import codebot.codeclip.token_profiler


class TestTokenProfiler(unittest.TestCase):
    """Test cases for the TokenProfiler class."""

    def test_process_file_stores_path(self):
        """Test that process_file stores the file path for common prefix calculation."""
        profiler = codebot.codeclip.token_profiler.TokenProfiler()

        # Mock the token counting functionality
        profiler.count_tokens = unittest.mock.MagicMock(return_value=100)

        # Process a file
        file_path = pathlib.Path("/Users/jack/src/metta/test.py")
        profiler.process_file(file_path, "content")

        # Check that the path was stored
        self.assertIn(str(file_path.parent), profiler.file_paths)


if __name__ == "__main__":
    unittest.main()
