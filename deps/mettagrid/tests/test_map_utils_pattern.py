import numpy as np
import pytest

from mettagrid.map.utils.pattern import (
    Pattern,
    ascii_to_patterns_with_counts,
    ascii_to_weights_of_all_patterns,
    parse_ascii_into_grid,
)


# Common test fixtures
@pytest.fixture
def simple_grid():
    """Fixture providing a simple 2x2 grid of all True values."""
    source = """
    |##|
    |##|
    """
    return parse_ascii_into_grid(source)


@pytest.fixture
def complex_grid():
    """Fixture providing a 3x3 grid with a hole in the middle."""
    source = """
    |###|
    |# #|
    |###|
    """
    return parse_ascii_into_grid(source)


@pytest.fixture
def asymmetric_grid():
    """Fixture providing an asymmetric 3x3 grid for symmetry tests."""
    source = """
    |#  |
    |## |
    | # |
    """
    return parse_ascii_into_grid(source)


# Test cases with parametrization
class TestParseAscii:
    def test_basic_parsing(self, complex_grid):
        expected = np.array(
            [
                [True, True, True],
                [True, False, True],
                [True, True, True],
            ],
            dtype=bool,
        )
        assert np.array_equal(complex_grid, expected)

    @pytest.mark.parametrize(
        "source,error_type,error_message",
        [
            (
                """
                ###
                # #
                ###
                """,
                ValueError,
                "Pattern must be enclosed in | characters",
            ),
            (
                """
                |###|
                |# |
                |###|
                """,
                ValueError,
                "All lines must be the same width",
            ),
            (
                """
                |123|
                |4 5|
                |678|
                """,
                ValueError,
                "Pattern must be composed of # and space characters",
            ),
        ],
    )
    def test_error_cases(self, source, error_type, error_message):
        with pytest.raises(error_type, match=error_message):
            parse_ascii_into_grid(source)


class TestPattern:
    def test_basic_creation(self, complex_grid):
        # Create pattern from top-left corner
        pattern = Pattern(complex_grid, 0, 0, 2)
        expected = np.array(
            [
                [True, True],
                [True, False],
            ],
            dtype=bool,
        )
        assert np.array_equal(pattern.data, expected)

        # Create pattern from bottom-right corner with wrapping
        pattern = Pattern(complex_grid, 2, 2, 2)
        expected = np.array(
            [
                [True, True],
                [True, True],
            ],
            dtype=bool,
        )
        assert np.array_equal(pattern.data, expected)

    @pytest.mark.parametrize(
        "source,position,expected_index",
        [
            (
                """
            |##|
            |##|
            """,
                (0, 0),
                15,
            ),  # 1+2+4+8 for all True
            (
                """
            |# |
            | #|
            """,
                (0, 0),
                9,
            ),  # 1+8 for True at (0,0) and (1,1)
        ],
    )
    def test_index(self, source, position, expected_index):
        grid = parse_ascii_into_grid(source)
        x, y = position
        pattern = Pattern(grid, x, y, 2)
        assert pattern.index() == expected_index

    @pytest.mark.parametrize(
        "symmetry,expected_variations,expected_unique_indices",
        [
            ("none", 1, 1),
            ("horizontal", 2, 2),
            ("all", 8, 8),
        ],
    )
    def test_pattern_variations(self, asymmetric_grid, symmetry, expected_variations, expected_unique_indices):
        pattern = Pattern(asymmetric_grid, 0, 0, 3)
        variations = pattern.variations(symmetry)

        assert len(variations) == expected_variations
        assert len(set(p.index() for p in variations)) == expected_unique_indices
        assert np.array_equal(variations[0].data, pattern.data)

        if symmetry == "horizontal":
            assert np.array_equal(variations[1].data, pattern.reflected().data)


class TestPatternsWithCounts:
    @pytest.mark.parametrize(
        "periodic,expected_count",
        [
            (False, 1),
            (True, 4),  # 4 possible positions with wrapping
        ],
    )
    def test_simple_pattern_counts(self, simple_grid, periodic, expected_count):
        source = """
        |##|
        |##|
        """
        patterns = ascii_to_patterns_with_counts(source, 2, periodic=periodic, symmetry="none")

        assert len(patterns) == 1
        assert patterns[0][0].index() == 15  # all true pattern
        assert patterns[0][1] == expected_count

    def test_multiple_patterns(self, complex_grid):
        source = """
        |###|
        |# #|
        |###|
        """
        patterns = ascii_to_patterns_with_counts(source, 2, periodic=False, symmetry="none")

        # 2x2 window in a 3x3 grid gives 4 possible positions
        assert len(patterns) == 4

        # Sum of counts should be number of possible patterns in the source
        total_count = sum(info[1] for info in patterns)
        assert total_count == 4

    @pytest.mark.parametrize(
        "symmetry,expected_total_count",
        [
            ("none", 4),
            ("all", 32),  # 4 patterns * 8 variations per pattern
        ],
    )
    def test_with_symmetry(self, complex_grid, symmetry, expected_total_count):
        source = """
        |###|
        |# #|
        |###|
        """
        patterns = ascii_to_patterns_with_counts(source, 2, periodic=False, symmetry=symmetry)

        total_count = sum(info[1] for info in patterns)
        assert total_count == expected_total_count


class TestWeightsOfAllPatterns:
    def test_simple_source(self, simple_grid):
        source = """
        |##|
        |##|
        """
        weights = ascii_to_weights_of_all_patterns(source, 2, periodic=False, symmetry="none")

        # For a 2x2 pattern, there are 2^4 = 16 possible patterns (0-15)
        assert len(weights) == 16
        assert weights[15] == 1  # All true pattern

        # All other weights should be 0
        for i in range(15):
            assert weights[i] == 0

    def test_complex_source(self, complex_grid):
        source = """
        |###|
        |# #|
        |###|
        """
        weights = ascii_to_weights_of_all_patterns(source, 2, periodic=False, symmetry="none")

        # For a 2x2 pattern, there are still 16 possible patterns
        assert len(weights) == 16

        # Sum of weights should equal number of patterns in source
        assert sum(weights) == 4

    @pytest.mark.parametrize(
        "symmetry,expected_total_weight",
        [
            ("none", 4),
            ("all", 32),  # 4 patterns * 8 variations per pattern
        ],
    )
    def test_with_symmetry(self, complex_grid, symmetry, expected_total_weight):
        source = """
        |###|
        |# #|
        |###|
        """
        weights = ascii_to_weights_of_all_patterns(source, 2, periodic=False, symmetry=symmetry)

        assert sum(weights) == expected_total_weight
