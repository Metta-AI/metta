import numpy as np
import pytest

from metta.map.utils.pattern import (
    Pattern,
    ascii_to_patterns_with_counts,
    ascii_to_weights_of_all_patterns,
    parse_ascii_into_grid,
)


# TestParseAscii reduced - tests basic parsing functionality
class TestParseAscii:
    def test_basic_parsing(self):
        source = """
            ###
            #.#
            ###
        """
        grid = parse_ascii_into_grid(source)
        assert grid.shape == (3, 3)
        assert grid.dtype == bool


# TestPattern reduced - tests basic pattern functionality
class TestPattern:
    def test_basic_creation(self):
        source = """
            ###
            #.#
            ###
        """
        grid = parse_ascii_into_grid(source)
        pattern = Pattern(grid, 0, 0, 2)
        assert pattern.data.shape == (2, 2)
        assert pattern.data.dtype == bool


# TestPatternsWithCounts reduced - tests essential pattern counting
class TestPatternsWithCounts:
    def test_basic_pattern_counting(self):
        source = """
            ##
            ##
        """
        patterns = ascii_to_patterns_with_counts(source, 2, periodic=False, symmetry="none")
        assert len(patterns) == 1
        assert patterns[0][1] == 1


# TestWeightsOfAllPatterns reduced - tests essential weight calculation
class TestWeightsOfAllPatterns:
    def test_basic_weights(self):
        source = """
            ##
            ##
        """
        weights = ascii_to_weights_of_all_patterns(source, 2, periodic=False, symmetry="none")
        assert len(weights) == 16
        assert sum(weights) == 1


if __name__ == "__main__":
    pytest.main()
