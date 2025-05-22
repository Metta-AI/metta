from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest

# Import SYMBOLS from the actual source file
from mettagrid.room.ascii import SYMBOLS


def find_map_files(root_dir: str = ".") -> List[Path]:
    """Find all .map files in the repository."""
    root_path = Path(root_dir)
    return sorted(root_path.rglob("*.map"))


def validate_map_structure(file_path: Path) -> Tuple[bool, List[str]]:
    """
    Validate a .map file structure.

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        errors.append(f"Failed to read file: {e}")
        return False, errors

    lines = content.strip().splitlines()

    # Check if file is empty
    if not lines:
        errors.append("File is empty")
        return False, errors

    # Check line length consistency
    line_lengths = [len(line) for line in lines]
    if len(set(line_lengths)) > 1:
        min_len, max_len = min(line_lengths), max(line_lengths)
        errors.append(f"Inconsistent line lengths: min={min_len}, max={max_len}")

        # Show which lines are problematic
        expected_length = max(line_lengths)
        for i, line in enumerate(lines):
            if len(line) != expected_length:
                errors.append(f"Line {i + 1} has length {len(line)}, expected {expected_length}")

    # Check for unknown symbols
    all_chars = set("".join(lines))
    unknown_chars = all_chars - set(SYMBOLS.keys())
    if unknown_chars:
        # Filter out common whitespace/control characters
        truly_unknown = unknown_chars - {"\t", "\r", "\n"}
        if truly_unknown:
            errors.append(f"Unknown symbols found: {sorted(truly_unknown)}")

    # Test NumPy array creation (the actual failure point in the original code)
    try:
        level_array = np.array([list(line) for line in lines], dtype="U6")
        # Test symbol mapping
        np.vectorize(SYMBOLS.get)(level_array)
    except Exception as e:
        errors.append(f"Failed to create NumPy array: {e}")

    return len(errors) == 0, errors


class TestAsciiMaps:
    """Test suite for ASCII map validation."""

    @pytest.fixture(scope="class")
    def map_files(self):
        """Fixture to find all .map files."""
        files = find_map_files()
        if not files:
            pytest.skip("No .map files found in repository")
        return files

    def test_map_files_found(self, map_files):
        """Test that we can find .map files in the repository."""
        assert len(map_files) > 0, "Should find at least one .map file"

    @pytest.mark.parametrize("map_file", find_map_files())
    def test_map_file_structure(self, map_file):
        """Test that each .map file has valid structure."""
        is_valid, errors = validate_map_structure(map_file)

        if not is_valid:
            error_msg = f"Map file validation failed for {map_file}:\n"
            for error in errors:
                error_msg += f"  - {error}\n"
            pytest.fail(error_msg)

    def test_all_maps_use_known_symbols(self, map_files):
        """Test that all maps only use symbols defined in SYMBOLS."""
        unknown_symbols_by_file = {}

        for map_file in map_files:
            try:
                with open(map_file, "r", encoding="utf-8") as f:
                    content = f.read()

                all_chars = set(content)
                unknown_chars = all_chars - set(SYMBOLS.keys()) - {"\t", "\r", "\n"}

                if unknown_chars:
                    unknown_symbols_by_file[map_file] = sorted(unknown_chars)
            except Exception:
                # Individual file errors will be caught by test_map_file_structure
                continue

        if unknown_symbols_by_file:
            error_msg = "Unknown symbols found in map files:\n"
            for file_path, symbols in unknown_symbols_by_file.items():
                error_msg += f"  {file_path}: {symbols}\n"
            pytest.fail(error_msg)

    def test_all_maps_have_consistent_line_lengths(self, map_files):
        """Test that all maps have consistent line lengths."""
        inconsistent_files = {}

        for map_file in map_files:
            try:
                with open(map_file, "r", encoding="utf-8") as f:
                    lines = f.read().strip().splitlines()

                if not lines:
                    continue

                line_lengths = [len(line) for line in lines]
                if len(set(line_lengths)) > 1:
                    min_len, max_len = min(line_lengths), max(line_lengths)
                    inconsistent_files[map_file] = (min_len, max_len, len(lines))

            except Exception:
                # Individual file errors will be caught by test_map_file_structure
                continue

        if inconsistent_files:
            error_msg = "Maps with inconsistent line lengths found:\n"
            for file_path, (min_len, max_len, total_lines) in inconsistent_files.items():
                error_msg += (
                    f"  {file_path}: lines vary from {min_len} to {max_len} chars ({total_lines} lines total)\n"
                )
            pytest.fail(error_msg)

    def test_maps_can_be_loaded_as_numpy_arrays(self, map_files):
        """Test that all maps can be successfully loaded as NumPy arrays."""
        failed_files = {}

        for map_file in map_files:
            try:
                with open(map_file, "r", encoding="utf-8") as f:
                    lines = f.read().strip().splitlines()

                if not lines:
                    continue

                # This is the exact operation that was failing in the original error
                level_array = np.array([list(line) for line in lines], dtype="U6")
                mapped_array = np.vectorize(SYMBOLS.get)(level_array)

                # Basic sanity checks
                assert level_array.ndim == 2, "Should be a 2D array"
                assert level_array.shape[0] == len(lines), "Should have correct number of rows"

            except Exception as e:
                failed_files[map_file] = str(e)

        if failed_files:
            error_msg = "Maps that failed to load as NumPy arrays:\n"
            for file_path, error in failed_files.items():
                error_msg += f"  {file_path}: {error}\n"
            pytest.fail(error_msg)


# Utility functions for manual testing and debugging
def print_map_summary():
    """Print a summary of all maps found (useful for debugging)."""
    map_files = find_map_files()

    print(f"Found {len(map_files)} .map files:")

    valid_count = 0
    invalid_count = 0

    for map_file in map_files:
        is_valid, errors = validate_map_structure(map_file)
        status = "✓" if is_valid else "✗"
        print(f"  {status} {map_file}")

        if not is_valid:
            invalid_count += 1
            for error in errors[:2]:  # Show first 2 errors
                print(f"    - {error}")
            if len(errors) > 2:
                print(f"    ... and {len(errors) - 2} more errors")
        else:
            valid_count += 1

    print(f"\nSummary: {valid_count} valid, {invalid_count} invalid")


def fix_line_lengths(file_path: Path, dry_run: bool = True) -> bool:
    """
    Fix inconsistent line lengths by padding shorter lines with spaces.

    Args:
        file_path: Path to the .map file
        dry_run: If True, only print what would be changed

    Returns:
        True if file was (or would be) modified
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return False

    lines = content.strip().splitlines()
    line_lengths = [len(line) for line in lines]

    if len(set(line_lengths)) <= 1:
        return False  # All lines same length

    max_length = max(line_lengths)
    fixed_lines = []
    changes_made = False

    for i, line in enumerate(lines):
        if len(line) < max_length:
            fixed_line = line.ljust(max_length)
            fixed_lines.append(fixed_line)
            changes_made = True
            if dry_run:
                print(f"  Line {i + 1}: would pad from {len(line)} to {max_length} chars")
        else:
            fixed_lines.append(line)

    if changes_made:
        if dry_run:
            print(f"Would fix {file_path} ({sum(1 for l in line_lengths if l < max_length)} lines need padding)")
        else:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("\n".join(fixed_lines) + "\n")
            print(f"Fixed {file_path}")

    return changes_made


if __name__ == "__main__":
    # When run directly, show map summary
    print_map_summary()
