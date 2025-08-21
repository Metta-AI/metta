"""
Tests for token profiler functionality including common prefix detection.
"""

import unittest
from pathlib import Path
from unittest.mock import MagicMock

from codeclip.token_profiler import TokenProfiler


class TestTokenProfiler(unittest.TestCase):
    """Test cases for the TokenProfiler class."""

    def test_process_file_stores_path(self):
        """Test that process_file stores the file path for common prefix calculation."""
        profiler = TokenProfiler()

        # Mock the token counting functionality
        profiler.count_tokens = MagicMock(return_value=100)

        # Process a file
        file_path = Path("/Users/jack/src/metta/test.py")
        profiler.process_file(file_path, "content")

        # Check that the path was stored
        self.assertIn(str(file_path.parent), profiler.file_paths)


if __name__ == "__main__":
    unittest.main()
