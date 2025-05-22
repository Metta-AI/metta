"""
Test suite for validating ASCII map files.

This module validates .map files used by the mettagrid system to ensure they:
1. Have consistent line lengths (required for NumPy array creation)
2. Only contain symbols defined in the SYMBOLS mapping
3. Can be successfully loaded as 2D NumPy arrays

The validation prevents InstantiationException errors that occur when
malformed maps are loaded at runtime.
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest

from mettagrid.room.ascii import SYMBOLS


def find_map_files(root_dir: str = ".") -> List[Path]:
    """
    Find all .map files in the repository, excluding development directories.

    Args:
        root_dir: Root directory to search from

    Returns:
        Sorted list of Path objects for .map files
    """
    root_path = Path(root_dir).resolve()

    # Exclude common development directories that might contain non-game .map files
    exclude_patterns = {
        ".venv",
        "venv",
        "node_modules",
        ".git",
        "dist",
        "build",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
    }

    map_files = []
    for map_file in root_path.rglob("*.map"):
        try:
            relative_path = map_file.relative_to(root_path)
            if not any(part in exclude_patterns for part in relative_path.parts):
                map_files.append(map_file)
        except ValueError:
            # Include files we can't make relative
            map_files.append(map_file)

    return sorted(map_files)


def validate_map_structure(file_path: Path) -> Tuple[bool, List[str]]:
    """
    Validate the structure and content of an ASCII map file.

    Args:
        file_path: Path to the .map file to validate

    Returns:
        Tuple of (is_valid, list_of_error_messages)
    """
    errors = []

    # Read file content
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        errors.append(f"Failed to read file: {e}")
        return False, errors

    lines = content.strip().splitlines()

    # Check for empty files
    if not lines:
        errors.append("File is empty")
        return False, errors

    # Validate line length consistency
    line_lengths = [len(line) for line in lines]
    if len(set(line_lengths)) > 1:
        min_len, max_len = min(line_lengths), max(line_lengths)
        errors.append(f"Inconsistent line lengths: min={min_len}, max={max_len}")

        # Report specific problematic lines
        expected_length = max(line_lengths)
        for i, line in enumerate(lines):
            if len(line) != expected_length:
                errors.append(f"Line {i + 1} has length {len(line)}, expected {expected_length}")

    # Validate symbols
    all_chars = set("".join(lines))
    unknown_chars = all_chars - set(SYMBOLS.keys())
    if unknown_chars:
        # Filter out acceptable whitespace characters
        truly_unknown = unknown_chars - {"\t", "\r", "\n"}
        if truly_unknown:
            errors.append(f"Unknown symbols found: {sorted(truly_unknown)}")

    # Test NumPy array creation (the critical operation that was failing)
    try:
        level_array = np.array([list(line) for line in lines], dtype="U6")
        np.vectorize(SYMBOLS.get)(level_array)
    except Exception as e:
        errors.append(f"Failed to create NumPy array: {e}")

    return len(errors) == 0, errors


class TestAsciiMaps:
    """Test suite for ASCII map validation."""

    @pytest.fixture(scope="class")
    def map_files(self):
        """Fixture providing all .map files found in the repository."""
        files = find_map_files()
        if not files:
            pytest.skip("No .map files found in repository")
        return files

    def test_map_files_discovered(self, map_files):
        """Verify that map files are found in the repository."""
        assert len(map_files) > 0, "Should discover at least one .map file"

    @pytest.mark.parametrize("map_file", find_map_files())
    def test_individual_map_structure(self, map_file):
        """Test that each map file has valid structure and content."""
        is_valid, errors = validate_map_structure(map_file)

        if not is_valid:
            # Create detailed error message for failed validation
            relative_path = map_file.relative_to(Path.cwd()) if Path.cwd() in map_file.parents else map_file
            error_details = "\n".join(f"  • {error}" for error in errors)
            pytest.fail(f"Map validation failed for {relative_path}:\n{error_details}")

    def test_all_maps_use_known_symbols(self, map_files):
        """Verify all maps only use symbols defined in SYMBOLS mapping."""
        files_with_unknown_symbols = {}

        for map_file in map_files:
            try:
                with open(map_file, "r", encoding="utf-8") as f:
                    content = f.read()

                all_chars = set(content)
                unknown_chars = all_chars - set(SYMBOLS.keys()) - {"\t", "\r", "\n"}

                if unknown_chars:
                    files_with_unknown_symbols[map_file] = sorted(unknown_chars)
            except Exception:
                # Individual file read errors will be caught by test_individual_map_structure
                continue

        if files_with_unknown_symbols:
            error_lines = []
            for file_path, symbols in files_with_unknown_symbols.items():
                relative_path = file_path.relative_to(Path.cwd()) if Path.cwd() in file_path.parents else file_path
                error_lines.append(f"  • {relative_path}: {symbols}")

            pytest.fail("Maps contain unknown symbols:\n" + "\n".join(error_lines))

    def test_all_maps_have_consistent_line_lengths(self, map_files):
        """Verify all maps have consistent line lengths within each file."""
        files_with_inconsistent_lengths = {}

        for map_file in map_files:
            try:
                with open(map_file, "r", encoding="utf-8") as f:
                    lines = f.read().strip().splitlines()

                if lines:
                    line_lengths = [len(line) for line in lines]
                    if len(set(line_lengths)) > 1:
                        min_len, max_len = min(line_lengths), max(line_lengths)
                        files_with_inconsistent_lengths[map_file] = (min_len, max_len, len(lines))

            except Exception:
                # Individual file read errors will be caught by test_individual_map_structure
                continue

        if files_with_inconsistent_lengths:
            error_lines = []
            for file_path, (min_len, max_len, total_lines) in files_with_inconsistent_lengths.items():
                relative_path = file_path.relative_to(Path.cwd()) if Path.cwd() in file_path.parents else file_path
                error_lines.append(f"  • {relative_path}: {min_len}-{max_len} chars ({total_lines} lines)")

            pytest.fail("Maps have inconsistent line lengths:\n" + "\n".join(error_lines))

    def test_all_maps_load_as_numpy_arrays(self, map_files):
        """Verify all maps can be loaded as NumPy arrays (critical for runtime)."""
        files_that_failed_loading = {}

        for map_file in map_files:
            try:
                with open(map_file, "r", encoding="utf-8") as f:
                    lines = f.read().strip().splitlines()

                if lines:
                    # This is the exact operation that causes InstantiationException
                    level_array = np.array([list(line) for line in lines], dtype="U6")
                    mapped_array = np.vectorize(SYMBOLS.get)(level_array)

                    # Basic structure validation
                    assert level_array.ndim == 2, "Should be 2D array"
                    assert level_array.shape[0] == len(lines), "Should have correct row count"

            except Exception as e:
                files_that_failed_loading[map_file] = str(e)

        if files_that_failed_loading:
            error_lines = []
            for file_path, error in files_that_failed_loading.items():
                relative_path = file_path.relative_to(Path.cwd()) if Path.cwd() in file_path.parents else file_path
                error_lines.append(f"  • {relative_path}: {error}")

            pytest.fail("Maps failed to load as NumPy arrays:\n" + "\n".join(error_lines))


def print_validation_summary():
    """Print a comprehensive validation summary (useful for manual inspection)."""
    print("ASCII Map Validation Summary")
    print("=" * 50)

    map_files = find_map_files()
    print(f"Found {len(map_files)} .map files\n")

    if not map_files:
        print("No .map files found to validate.")
        return

    valid_count = 0
    invalid_files = []

    for map_file in map_files:
        is_valid, errors = validate_map_structure(map_file)
        relative_path = map_file.relative_to(Path.cwd()) if Path.cwd() in map_file.parents else map_file

        if is_valid:
            print(f"✓ {relative_path}")
            valid_count += 1
        else:
            print(f"✗ {relative_path}")
            for error in errors:
                print(f"    • {error}")
            invalid_files.append(relative_path)
            print()

    print("\nValidation Results:")
    print(f"  Valid:   {valid_count}")
    print(f"  Invalid: {len(invalid_files)}")

    if invalid_files:
        print("\nFiles requiring manual correction:")
        for file_path in invalid_files:
            print(f"  • {file_path}")


if __name__ == "__main__":
    print_validation_summary()
