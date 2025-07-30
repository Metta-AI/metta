"""Tests for metta.common.util.colorama module."""

from colorama import Fore, Style

from metta.common.util.text_styles import (
    blue,
    bold,
    colorize,
    cyan,
    green,
    magenta,
    red,
    use_colors,
    yellow,
)


class TestColoramaUtils:
    """Test cases for colorama utility functions."""

    def setup_method(self):
        """Reset color settings before each test."""
        use_colors(True)

    def test_colorize_with_colors_enabled(self):
        """Test colorize function when colors are enabled."""
        text = "test text"
        color = Fore.RED
        result = colorize(text, color)
        expected = f"{color}{text}{Style.RESET_ALL}"
        assert result == expected

    def test_colorize_with_colors_disabled(self):
        """Test colorize function when colors are disabled."""
        use_colors(False)
        text = "test text"
        color = Fore.RED
        result = colorize(text, color)
        assert result == text

    def test_red_function(self):
        """Test red color function."""
        text = "error message"
        result = red(text)
        expected = f"{Fore.RED}{text}{Style.RESET_ALL}"
        assert result == expected

    def test_green_function(self):
        """Test green color function."""
        text = "success message"
        result = green(text)
        expected = f"{Fore.GREEN}{text}{Style.RESET_ALL}"
        assert result == expected

    def test_yellow_function(self):
        """Test yellow color function."""
        text = "warning message"
        result = yellow(text)
        expected = f"{Fore.YELLOW}{text}{Style.RESET_ALL}"
        assert result == expected

    def test_cyan_function(self):
        """Test cyan color function."""
        text = "info message"
        result = cyan(text)
        expected = f"{Fore.CYAN}{text}{Style.RESET_ALL}"
        assert result == expected

    def test_blue_function(self):
        """Test blue color function."""
        text = "blue message"
        result = blue(text)
        expected = f"{Fore.BLUE}{text}{Style.RESET_ALL}"
        assert result == expected

    def test_magenta_function(self):
        """Test magenta color function."""
        text = "magenta message"
        result = magenta(text)
        expected = f"{Fore.MAGENTA}{text}{Style.RESET_ALL}"
        assert result == expected

    def test_bold_function(self):
        """Test bold style function."""
        text = "bold message"
        result = bold(text)
        expected = f"{Style.BRIGHT}{text}{Style.RESET_ALL}"
        assert result == expected

    def test_use_colors_true(self):
        """Test enabling colors."""
        use_colors(True)
        text = "test"
        result = red(text)
        expected = f"{Fore.RED}{text}{Style.RESET_ALL}"
        assert result == expected

    def test_use_colors_false(self):
        """Test disabling colors."""
        use_colors(False)
        text = "test"
        result = red(text)
        assert result == text

    def test_color_functions_with_empty_string(self):
        """Test color functions with empty string."""
        empty_text = ""

        assert red(empty_text) == f"{Fore.RED}{empty_text}{Style.RESET_ALL}"
        assert green(empty_text) == f"{Fore.GREEN}{empty_text}{Style.RESET_ALL}"
        assert yellow(empty_text) == f"{Fore.YELLOW}{empty_text}{Style.RESET_ALL}"
        assert cyan(empty_text) == f"{Fore.CYAN}{empty_text}{Style.RESET_ALL}"
        assert blue(empty_text) == f"{Fore.BLUE}{empty_text}{Style.RESET_ALL}"
        assert magenta(empty_text) == f"{Fore.MAGENTA}{empty_text}{Style.RESET_ALL}"
        assert bold(empty_text) == f"{Style.BRIGHT}{empty_text}{Style.RESET_ALL}"

    def test_color_functions_with_special_characters(self):
        """Test color functions with special characters."""
        special_text = "!@#$%^&*()_+-=[]{}|;':\",./<>?"

        result = red(special_text)
        expected = f"{Fore.RED}{special_text}{Style.RESET_ALL}"
        assert result == expected

    def test_color_functions_with_unicode(self):
        """Test color functions with unicode characters."""
        unicode_text = "Hello 世界 🌍"

        result = green(unicode_text)
        expected = f"{Fore.GREEN}{unicode_text}{Style.RESET_ALL}"
        assert result == expected

    def test_color_functions_with_multiline_text(self):
        """Test color functions with multiline text."""
        multiline_text = "Line 1\nLine 2\nLine 3"

        result = blue(multiline_text)
        expected = f"{Fore.BLUE}{multiline_text}{Style.RESET_ALL}"
        assert result == expected

    def test_color_functions_when_disabled(self):
        """Test all color functions return plain text when colors are disabled."""
        use_colors(False)
        text = "test message"

        assert red(text) == text
        assert green(text) == text
        assert yellow(text) == text
        assert cyan(text) == text
        assert blue(text) == text
        assert magenta(text) == text
        assert bold(text) == text

    def test_use_colors_toggle(self):
        """Test toggling colors on and off multiple times."""
        text = "test"

        # Start with colors enabled
        use_colors(True)
        colored_result = red(text)
        assert colored_result == f"{Fore.RED}{text}{Style.RESET_ALL}"

        # Disable colors
        use_colors(False)
        plain_result = red(text)
        assert plain_result == text

        # Re-enable colors
        use_colors(True)
        colored_again = red(text)
        assert colored_again == f"{Fore.RED}{text}{Style.RESET_ALL}"

    def test_colorize_with_different_styles(self):
        """Test colorize function with different color and style combinations."""
        text = "test"

        # Test with foreground colors
        assert colorize(text, Fore.RED) == f"{Fore.RED}{text}{Style.RESET_ALL}"
        assert colorize(text, Fore.GREEN) == f"{Fore.GREEN}{text}{Style.RESET_ALL}"

        # Test with styles
        assert colorize(text, Style.BRIGHT) == f"{Style.BRIGHT}{text}{Style.RESET_ALL}"
        assert colorize(text, Style.DIM) == f"{Style.DIM}{text}{Style.RESET_ALL}"
