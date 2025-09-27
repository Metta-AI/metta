import numpy as np
import pytest

from mettagrid.mapgen.utils.pattern import (
    Pattern,
    ascii_to_patterns_with_counts,
    ascii_to_weights_of_all_patterns,
    parse_ascii_into_grid,
)


class TestParseAscii:
    def test_basic_parsing(self):
        source = """
            ###
            #.#
            ###
        """
        grid = parse_ascii_into_grid(source)

        expected = np.array(
            [
                [True, True, True],
                [True, False, True],
                [True, True, True],
            ],
            dtype=bool,
        )

        assert np.array_equal(grid, expected)

    def test_error_cases(self):
        with pytest.raises(ValueError):
            parse_ascii_into_grid("""
                ###
                #.
                ###
            """)  # Inconsistent width

        with pytest.raises(ValueError):
            parse_ascii_into_grid("""
                123
                4 5
                678
            """)  # Invalid characters


class TestPattern:
    def test_basic_creation(self):
        source = """
            ###
            #.#
            ###
        """
        grid = parse_ascii_into_grid(source)

        # Create pattern from top-left corner
        pattern = Pattern(grid, 0, 0, 2)
        expected = np.array(
            [
                [True, True],
                [True, False],
            ],
            dtype=bool,
        )
        assert np.array_equal(pattern.data, expected)

        # Create pattern from bottom-right corner with wrapping
        pattern = Pattern(grid, 2, 2, 2)
        expected = np.array(
            [
                [True, True],
                [True, True],
            ],
            dtype=bool,
        )
        assert np.array_equal(pattern.data, expected)

    def test_index(self):
        # Create a simple pattern and verify its index
        source = """
            ##
            ##
        """
        grid = parse_ascii_into_grid(source)
        pattern = Pattern(grid, 0, 0, 2)

        # Index should be 15 (1+2+4+8) for a 2x2 grid of all True
        assert pattern.index() == 15

        # Test another pattern
        source = """
            #.
            .#
        """
        grid = parse_ascii_into_grid(source)
        pattern = Pattern(grid, 0, 0, 2)

        # Index should be 9 (1+8) for a 2x2 grid with True at (0,0) and (1,1)
        assert pattern.index() == 9

    def test_pattern_variations(self):
        # Test the variations method for different symmetry settings
        source = """
            #..
            ##.
            .#.
        """
        grid = parse_ascii_into_grid(source)
        pattern = Pattern(grid, 0, 0, 3)

        # Test with symmetry="none"
        variations = pattern.variations("none")
        assert len(variations) == 1
        assert len(set([p.index() for p in variations])) == 1
        assert np.array_equal(variations[0].data, pattern.data)

        # Test with symmetry="horizontal"
        variations = pattern.variations("horizontal")
        assert len(variations) == 2
        assert len(set([p.index() for p in variations])) == 2
        assert np.array_equal(variations[0].data, pattern.data)
        assert np.array_equal(variations[1].data, pattern.reflected().data)

        # Test with symmetry="all"
        variations = pattern.variations("all")
        assert len(variations) == 8
        # Check for uniqueness in the generated patterns
        assert len(set([p.index() for p in variations])) == 8


class TestPatternsWithCounts:
    def test_nonperiodic(self):
        # Simple 2x2 grid with only one pattern
        source = """
            ##
            ##
        """
        patterns = ascii_to_patterns_with_counts(source, 2, periodic=False, symmetry="none")

        # There should be 1 pattern (only one 2x2 pattern in a 2x2 source)
        assert len(patterns) == 1

        # The pattern index should be 15 (all true)
        assert patterns[0][0].index() == 15
        assert patterns[0][1] == 1

    def test_periodic(self):
        source = """
            ##
            ##
        """
        patterns = ascii_to_patterns_with_counts(source, 2, periodic=True, symmetry="none")

        assert len(patterns) == 1

        assert patterns[0][0].index() == 15
        assert patterns[0][1] == 4  # generated from all 4 possible positions - wrapping is allowed

    def test_multiple_patterns(self):
        # More complex source with multiple patterns
        source = """
            ###
            #.#
            ###
        """
        patterns = ascii_to_patterns_with_counts(source, 2, periodic=False, symmetry="none")

        # Check we have the expected number of unique patterns
        # 2x2 window in a 3x3 grid gives 4 possible positions
        assert len(patterns) == 4

        # Sum of counts should be number of possible patterns in the source
        total_count = sum(info[1] for info in patterns)
        assert total_count == 4

    def test_with_symmetry(self):
        # Test with symmetry="all"
        source = """
            ###
            #.#
            ###
        """
        patterns = ascii_to_patterns_with_counts(source, 2, periodic=False, symmetry="all")
        assert len(patterns) == 4

        # With symmetry, there might be more patterns, but the total count should be the same
        total_count = sum(info[1] for info in patterns)
        assert total_count == 4 * 8  # 4 patterns * 8 variations per pattern


class TestWeightsOfAllPatterns:
    def test_simple_source(self):
        # Simple source with one pattern
        source = """
            ##
            ##
        """
        weights = ascii_to_weights_of_all_patterns(source, 2, periodic=False, symmetry="none")

        # For a 2x2 pattern, there are 2^4 = 16 possible patterns (0-15)
        assert len(weights) == 16

        # Check the weight of the pattern in the source
        assert weights[15] == 1  # All true pattern

        # All other weights should be 0
        for i in range(15):
            assert weights[i] == 0

    def test_complex_source(self):
        # Test with a more complex source
        source = """
            ###
            #.#
            ###
        """
        weights = ascii_to_weights_of_all_patterns(source, 2, periodic=False, symmetry="none")

        # For a 2x2 pattern, there are still 16 possible patterns
        assert len(weights) == 16

        # Sum of weights should equal number of patterns in source
        assert sum(weights) == 4

    def test_with_symmetry(self):
        # Test with symmetry="all"
        source = """
            ###
            #.#
            ###
        """
        weights = ascii_to_weights_of_all_patterns(source, 2, periodic=False, symmetry="all")

        # With symmetry, the total weight should increase
        assert sum(weights) == 4 * 8  # 4 patterns * 8 variations per pattern


if __name__ == "__main__":
    pytest.main()
