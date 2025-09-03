"""Tests for ASCII map transformation utilities."""

import pytest

from metta.mettagrid.mapgen.utils.ascii_transform import (
    mirror_ascii_map,
    rotate_ascii_map,
    stretch_ascii_map,
    transform_ascii_map,
)


class TestRotateAsciiMap:
    """Test rotation transformations."""

    def test_rotate_90_simple(self):
        """Test 90 degree rotation on simple map."""
        original = "#.@\n#._\n###"

        rotated = rotate_ascii_map(original, 90)
        expected = "###\n#..\n#_@"

        assert rotated == expected

    def test_rotate_180_simple(self):
        """Test 180 degree rotation on simple map."""
        original = "#.@\n#._\n###"

        rotated = rotate_ascii_map(original, 180)
        expected = "###\n_.#\n@.#"

        assert rotated == expected

    def test_rotate_270_simple(self):
        """Test 270 degree rotation on simple map."""
        original = "#.@\n#._\n###"

        rotated = rotate_ascii_map(original, 270)
        expected = "@_#\n..#\n###"

        assert rotated == expected

    def test_four_rotations_return_original(self):
        """Test that rotating 4 times by 90 degrees returns to original."""
        original = "#####\n#@._#\n#nmC#\n#RBG#\n#####"

        result = original
        for _ in range(4):
            result = rotate_ascii_map(result, 90)

        assert result == original

    def test_two_180_rotations_return_original(self):
        """Test that rotating twice by 180 degrees returns to original."""
        original = "#####\n#@._#\n#nmC#\n#RBG#\n#####"

        result = rotate_ascii_map(original, 180)
        result = rotate_ascii_map(result, 180)

        assert result == original

    def test_rotation_preserves_special_characters(self):
        """Test that rotation preserves converter chain characters."""
        # Map with various special characters
        original = "####\n#@c#\n#nR#\n####"

        rotated = rotate_ascii_map(original, 90)

        # Check that all characters are still present
        assert "@" in rotated
        assert "c" in rotated
        assert "n" in rotated
        assert "R" in rotated

    def test_invalid_degrees_raises_error(self):
        """Test that invalid rotation angles raise ValueError."""
        original = "###\n#.#\n###"

        with pytest.raises(ValueError, match="Degrees must be 90, 180, or 270"):
            rotate_ascii_map(original, 45)

        with pytest.raises(ValueError):
            rotate_ascii_map(original, 360)


class TestMirrorAsciiMap:
    """Test mirroring transformations."""

    def test_horizontal_mirror_simple(self):
        """Test horizontal mirroring (left-right flip)."""
        original = "#.@\n#._\n###"

        mirrored = mirror_ascii_map(original, "horizontal")
        expected = "@.#\n_.#\n###"

        assert mirrored == expected

    def test_vertical_mirror_simple(self):
        """Test vertical mirroring (top-bottom flip)."""
        original = "#.@\n#._\n###"

        mirrored = mirror_ascii_map(original, "vertical")
        expected = "###\n#._\n#.@"

        assert mirrored == expected

    def test_double_horizontal_mirror_returns_original(self):
        """Test that mirroring horizontally twice returns to original."""
        original = "#####\n#@._#\n#nmC#\n#RBG#\n#####"

        result = mirror_ascii_map(original, "horizontal")
        result = mirror_ascii_map(result, "horizontal")

        assert result == original

    def test_double_vertical_mirror_returns_original(self):
        """Test that mirroring vertically twice returns to original."""
        original = "#####\n#@._#\n#nmC#\n#RBG#\n#####"

        result = mirror_ascii_map(original, "vertical")
        result = mirror_ascii_map(result, "vertical")

        assert result == original

    def test_mirror_preserves_special_characters(self):
        """Test that mirroring preserves converter chain characters."""
        original = "####\n#@c#\n#nR#\n####"

        h_mirrored = mirror_ascii_map(original, "horizontal")
        v_mirrored = mirror_ascii_map(original, "vertical")

        # Check that all characters are still present
        for mirrored in [h_mirrored, v_mirrored]:
            assert "@" in mirrored
            assert "c" in mirrored
            assert "n" in mirrored
            assert "R" in mirrored

    def test_invalid_axis_raises_error(self):
        """Test that invalid axis raises ValueError."""
        original = "###\n#.#\n###"

        with pytest.raises(ValueError, match="Axis must be 'horizontal' or 'vertical'"):
            mirror_ascii_map(original, "diagonal")


class TestTransformAsciiMap:
    """Test combined transformations."""

    def test_rotate_and_mirror(self):
        """Test combining rotation and mirroring."""
        original = "#.@\n#._\n###"

        # Rotate 90 then mirror horizontally
        result = transform_ascii_map(original, rotate=90, mirror_horizontal=True)

        # Should be same as doing operations separately
        expected = rotate_ascii_map(original, 90)
        expected = mirror_ascii_map(expected, "horizontal")

        assert result == expected

    def test_all_transformations(self):
        """Test combining all transformation types."""
        original = "#####\n#@._#\n#nmC#\n#RBG#\n#####"

        result = transform_ascii_map(original, rotate=180, mirror_horizontal=True, mirror_vertical=True)

        # Should be same as doing operations separately in order
        expected = rotate_ascii_map(original, 180)
        expected = mirror_ascii_map(expected, "horizontal")
        expected = mirror_ascii_map(expected, "vertical")

        assert result == expected

    def test_no_transformations_returns_original(self):
        """Test that no transformations returns original."""
        original = "#####\n#@._#\n#nmC#\n#RBG#\n#####"

        result = transform_ascii_map(original)

        assert result == original

    def test_transformation_order_matters(self):
        """Test that transformation order affects result."""
        original = "#.@\n#._\n###"

        # Rotate then mirror horizontal
        result1 = transform_ascii_map(original, rotate=90, mirror_horizontal=True)

        # Mirror horizontal then rotate (done manually)
        temp = mirror_ascii_map(original, "horizontal")
        result2 = rotate_ascii_map(temp, 90)

        # Results should be different (proving order matters)
        assert result1 != result2


class TestComplexMaps:
    """Test with more complex, realistic maps."""

    def test_corridors_map_rotation_cycle(self):
        """Test rotation cycle on a corridors-style map."""
        # Simplified version of corridors map
        original = (
            "##########\n"
            "#####....#\n"
            "#####....#\n"
            "#....#...#\n"
            "#....#...#\n"
            "#@.......#\n"
            "#........#\n"
            "#...._...#\n"
            "#........#\n"
            "##########"
        )

        result = original
        # Rotate 4 times should return to original
        for _ in range(4):
            result = rotate_ascii_map(result, 90)

        assert result == original

    def test_converter_chain_map_transformations(self):
        """Test transformations on map with converter chains."""
        original = "#######\n#@...c#\n#....n#\n#m...R#\n#B.G._#\n#######"

        # Test that all converter chain characters survive transformations
        rotated_90 = rotate_ascii_map(original, 90)
        rotated_180 = rotate_ascii_map(original, 180)
        rotated_270 = rotate_ascii_map(original, 270)
        h_mirrored = mirror_ascii_map(original, "horizontal")
        v_mirrored = mirror_ascii_map(original, "vertical")

        all_transformed = [rotated_90, rotated_180, rotated_270, h_mirrored, v_mirrored]

        # Check all special characters are preserved in all transformations
        special_chars = ["@", "c", "n", "m", "R", "B", "G", "_"]
        for transformed in all_transformed:
            for char in special_chars:
                assert char in transformed, f"Character {char} missing after transformation"

    def test_asymmetric_map_dimensions(self):
        """Test transformations on non-square maps."""
        # 5x3 map
        original = "#####\n#@._#\n#####"

        # Rotate 90 degrees - should become 3x5
        rotated = rotate_ascii_map(original, 90)
        lines = rotated.split("\n")
        assert len(lines) == 5  # Height becomes 5
        assert all(len(line) == 3 for line in lines)  # Width becomes 3

        # Rotate back 3 more times
        result = rotated
        for _ in range(3):
            result = rotate_ascii_map(result, 90)

        assert result == original


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_single_character_map(self):
        """Test transformations on minimal 1x1 map."""
        original = "@"

        # All transformations should return the same
        assert rotate_ascii_map(original, 90) == original
        assert rotate_ascii_map(original, 180) == original
        assert rotate_ascii_map(original, 270) == original
        assert mirror_ascii_map(original, "horizontal") == original
        assert mirror_ascii_map(original, "vertical") == original


class TestStretchAsciiMap:
    """Tests for stretching ASCII maps with selective duplication."""

    def test_no_op_scale_returns_original(self):
        original = "#.@\n#._\n###"
        assert stretch_ascii_map(original, 1, 1) == original

    def test_stretch_width_only_duplicates_walls_and_empties(self):
        original = "#.@\n#._\n###"
        # scale_x=2, scale_y=1
        stretched = stretch_ascii_map(original, 2, 1)
        expected = (
            "##.@@".replace("@@", ".@")
            + "\n"  # '#', '.', '@' -> '#' duplicated, '.' duplicated, '@' once + '.'
            "##.__".replace("__", "._")
            + "\n"  # '#', '.', '_' -> '#', '.', '_' once + '.'
            "######"  # '###' -> '######'
        )
        # Explicit expected string for clarity
        expected = "##..@.\n##.._.\n######"
        assert stretched == expected

    def test_stretch_height_only_duplicates_walls_and_empties(self):
        original = "#.@\n#._\n###"
        # scale_x=1, scale_y=2
        stretched = stretch_ascii_map(original, 1, 2)
        expected = (
            "#.@\n"  # first row unchanged
            "#..\n"  # second copy: '@' replaced by '.'
            "#._\n"  # third row unchanged
            "#..\n"  # second copy: '_' replaced by '.'
            "###\n"  # bottom row unchanged
            "###"  # second copy identical
        )
        expected = "#.@\n#..\n#._\n#..\n###\n###"
        assert stretched == expected

    def test_stretch_both_dimensions(self):
        original = "#.@\n..#"
        # scale_x=2, scale_y=2
        stretched = stretch_ascii_map(original, 2, 2)
        expected = (
            "##.@.\n"  # top-left row: '#' duplicated, '.' duplicated, '@' once + '.'
            "##.. .\n"  # second row (vertical duplicate): '@' replaced by '.'
            "....##\n"  # third original row duplicated horizontally
            "....##"  # fourth vertical duplicate identical (only dots and '#')
        )
        # Cleanup expected with exact content
        expected = "##..@.\n##....\n....##\n....##"
        assert stretched == expected

    def test_objects_not_duplicated(self):
        # Ensure equal width lines to pass ASCII map validation
        original = "#c@R\n._G."
        stretched = stretch_ascii_map(original, 3, 2)
        # First row: '#', 'c', '@', 'R' with scale_x=3
        # '#' -> '###'
        # 'c' -> 'c..'
        # '@' -> '@..'
        # 'R' -> 'R..'
        # Second vertical row: objects replaced by '.'
        # Final exact expected based on rules:
        # Row1: '#c@R' -> '###c..@..R..'
        # Row1 copy: '###.........'
        # Row2: '._G.' -> '..._..G.....'
        # Row2 copy: '............'
        expected = "###c..@..R..\n###.........\n..._..G.....\n............"
        # Explanation:
        # Row1: '#c@R' -> '###c..@..R..'
        # Row1 copy: '###.........' (objects turned to '.')
        # Row2: '._G.' -> '..._..G.....'
        # Row2 copy: '............'
        assert stretched == expected

    def test_empty_lines_preserved(self):
        """Test that empty space characters are preserved."""
        original = "###\n# #\n###"

        rotated = rotate_ascii_map(original, 90)
        # Space should still be present
        assert " " in rotated

    def test_uniform_map(self):
        """Test transformations on map with all same character."""
        original = "###\n###\n###"

        # All transformations should return the same
        assert rotate_ascii_map(original, 90) == original
        assert rotate_ascii_map(original, 180) == original
        assert rotate_ascii_map(original, 270) == original
        assert mirror_ascii_map(original, "horizontal") == original
        assert mirror_ascii_map(original, "vertical") == original
