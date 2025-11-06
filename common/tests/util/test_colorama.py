"""Tests for metta.common.util.colorama module."""

import colorama

import metta.common.util.text_styles


class TestColoramaUtils:
    """Test cases for colorama utility functions."""

    def setup_method(self):
        """Reset color settings before each test."""
        metta.common.util.text_styles.use_colors(True)

    def test_colorize_with_colors_enabled(self):
        """Test colorize function when colors are enabled."""
        text = "test text"
        color = colorama.Fore.RED
        result = metta.common.util.text_styles.colorize(text, color)
        expected = f"{color}{text}{colorama.Style.RESET_ALL}"
        assert result == expected

    def test_colorize_with_colors_disabled(self):
        """Test colorize function when colors are disabled."""
        metta.common.util.text_styles.use_colors(False)
        text = "test text"
        color = colorama.Fore.RED
        result = metta.common.util.text_styles.colorize(text, color)
        assert result == text

    def test_red_function(self):
        """Test red color function."""
        text = "error message"
        result = metta.common.util.text_styles.red(text)
        expected = f"{colorama.Fore.RED}{text}{colorama.Style.RESET_ALL}"
        assert result == expected

    def test_green_function(self):
        """Test green color function."""
        text = "success message"
        result = metta.common.util.text_styles.green(text)
        expected = f"{colorama.Fore.GREEN}{text}{colorama.Style.RESET_ALL}"
        assert result == expected

    def test_yellow_function(self):
        """Test yellow color function."""
        text = "warning message"
        result = metta.common.util.text_styles.yellow(text)
        expected = f"{colorama.Fore.YELLOW}{text}{colorama.Style.RESET_ALL}"
        assert result == expected

    def test_cyan_function(self):
        """Test cyan color function."""
        text = "info message"
        result = metta.common.util.text_styles.cyan(text)
        expected = f"{colorama.Fore.CYAN}{text}{colorama.Style.RESET_ALL}"
        assert result == expected

    def test_blue_function(self):
        """Test blue color function."""
        text = "blue message"
        result = metta.common.util.text_styles.blue(text)
        expected = f"{colorama.Fore.BLUE}{text}{colorama.Style.RESET_ALL}"
        assert result == expected

    def test_magenta_function(self):
        """Test magenta color function."""
        text = "magenta message"
        result = metta.common.util.text_styles.magenta(text)
        expected = f"{colorama.Fore.MAGENTA}{text}{colorama.Style.RESET_ALL}"
        assert result == expected

    def test_bold_function(self):
        """Test bold style function."""
        text = "bold message"
        result = metta.common.util.text_styles.bold(text)
        expected = f"{colorama.Style.BRIGHT}{text}{colorama.Style.RESET_ALL}"
        assert result == expected

    def test_use_colors_true(self):
        """Test enabling colors."""
        metta.common.util.text_styles.use_colors(True)
        text = "test"
        result = metta.common.util.text_styles.red(text)
        expected = f"{colorama.Fore.RED}{text}{colorama.Style.RESET_ALL}"
        assert result == expected

    def test_use_colors_false(self):
        """Test disabling colors."""
        metta.common.util.text_styles.use_colors(False)
        text = "test"
        result = metta.common.util.text_styles.red(text)
        assert result == text

    def test_color_functions_with_empty_string(self):
        """Test color functions with empty string."""
        empty_text = ""

        assert (
            metta.common.util.text_styles.red(empty_text)
            == f"{colorama.Fore.RED}{empty_text}{colorama.Style.RESET_ALL}"
        )
        assert (
            metta.common.util.text_styles.green(empty_text)
            == f"{colorama.Fore.GREEN}{empty_text}{colorama.Style.RESET_ALL}"
        )
        assert (
            metta.common.util.text_styles.yellow(empty_text)
            == f"{colorama.Fore.YELLOW}{empty_text}{colorama.Style.RESET_ALL}"
        )
        assert (
            metta.common.util.text_styles.cyan(empty_text)
            == f"{colorama.Fore.CYAN}{empty_text}{colorama.Style.RESET_ALL}"
        )
        assert (
            metta.common.util.text_styles.blue(empty_text)
            == f"{colorama.Fore.BLUE}{empty_text}{colorama.Style.RESET_ALL}"
        )
        assert (
            metta.common.util.text_styles.magenta(empty_text)
            == f"{colorama.Fore.MAGENTA}{empty_text}{colorama.Style.RESET_ALL}"
        )
        assert (
            metta.common.util.text_styles.bold(empty_text)
            == f"{colorama.Style.BRIGHT}{empty_text}{colorama.Style.RESET_ALL}"
        )

    def test_color_functions_with_special_characters(self):
        """Test color functions with special characters."""
        special_text = "!@#$%^&*()_+-=[]{}|;':\",./<>?"

        result = metta.common.util.text_styles.red(special_text)
        expected = f"{colorama.Fore.RED}{special_text}{colorama.Style.RESET_ALL}"
        assert result == expected

    def test_color_functions_with_unicode(self):
        """Test color functions with unicode characters."""
        unicode_text = "Hello 世界 🌍"

        result = metta.common.util.text_styles.green(unicode_text)
        expected = f"{colorama.Fore.GREEN}{unicode_text}{colorama.Style.RESET_ALL}"
        assert result == expected

    def test_color_functions_with_multiline_text(self):
        """Test color functions with multiline text."""
        multiline_text = "Line 1\nLine 2\nLine 3"

        result = metta.common.util.text_styles.blue(multiline_text)
        expected = f"{colorama.Fore.BLUE}{multiline_text}{colorama.Style.RESET_ALL}"
        assert result == expected

    def test_color_functions_when_disabled(self):
        """Test all color functions return plain text when colors are disabled."""
        metta.common.util.text_styles.use_colors(False)
        text = "test message"

        assert metta.common.util.text_styles.red(text) == text
        assert metta.common.util.text_styles.green(text) == text
        assert metta.common.util.text_styles.yellow(text) == text
        assert metta.common.util.text_styles.cyan(text) == text
        assert metta.common.util.text_styles.blue(text) == text
        assert metta.common.util.text_styles.magenta(text) == text
        assert metta.common.util.text_styles.bold(text) == text

    def test_use_colors_toggle(self):
        """Test toggling colors on and off multiple times."""
        text = "test"

        # Start with colors enabled
        metta.common.util.text_styles.use_colors(True)
        colored_result = metta.common.util.text_styles.red(text)
        assert colored_result == f"{colorama.Fore.RED}{text}{colorama.Style.RESET_ALL}"

        # Disable colors
        metta.common.util.text_styles.use_colors(False)
        plain_result = metta.common.util.text_styles.red(text)
        assert plain_result == text

        # Re-enable colors
        metta.common.util.text_styles.use_colors(True)
        colored_again = metta.common.util.text_styles.red(text)
        assert colored_again == f"{colorama.Fore.RED}{text}{colorama.Style.RESET_ALL}"

    def test_colorize_with_different_styles(self):
        """Test colorize function with different color and style combinations."""
        text = "test"

        # Test with foreground colors
        assert (
            metta.common.util.text_styles.colorize(text, colorama.Fore.RED)
            == f"{colorama.Fore.RED}{text}{colorama.Style.RESET_ALL}"
        )
        assert (
            metta.common.util.text_styles.colorize(text, colorama.Fore.GREEN)
            == f"{colorama.Fore.GREEN}{text}{colorama.Style.RESET_ALL}"
        )

        # Test with styles
        assert (
            metta.common.util.text_styles.colorize(text, colorama.Style.BRIGHT)
            == f"{colorama.Style.BRIGHT}{text}{colorama.Style.RESET_ALL}"
        )
        assert (
            metta.common.util.text_styles.colorize(text, colorama.Style.DIM)
            == f"{colorama.Style.DIM}{text}{colorama.Style.RESET_ALL}"
        )
