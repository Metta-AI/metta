"""Tests for the CLI formatter module."""

from typing import List, Optional

from pydantic import BaseModel

from experiments.cli_formatter import TreeHelpFormatter, format_help_with_defaults


class NestedConfig(BaseModel):
    """Nested configuration for testing."""

    enabled: bool = True
    threshold: float = 0.5


class TestConfig(BaseModel):
    """Test configuration model."""

    name: str = "test_experiment"
    gpus: int = 1
    nodes: int = 1
    spot: bool = True
    tags: List[str] = ["test", "example"]
    nested: Optional[NestedConfig] = NestedConfig()


class TestTreeHelpFormatter:
    """Test the TreeHelpFormatter class."""

    def test_color_codes(self):
        """Test that color codes are defined correctly."""
        formatter = TreeHelpFormatter(prog="test")
        assert formatter.COLORS["option"] == "\033[94m"  # Blue
        assert formatter.COLORS["default"] == "\033[92m"  # Green
        assert formatter.COLORS["override"] == "\033[1;31m"  # Bold red
        assert formatter.COLORS["reset"] == "\033[0m"

    def test_no_colors_in_non_tty(self, monkeypatch):
        """Test that colors are disabled when not in a TTY."""
        monkeypatch.setattr("sys.stdout.isatty", lambda: False)
        formatter = TreeHelpFormatter(prog="test")
        assert not formatter.use_colors

    def test_user_overrides_initialization(self):
        """Test that user overrides are properly initialized."""
        overrides = {"gpus": 4, "spot": False}
        formatter = TreeHelpFormatter(prog="test", user_overrides=overrides)
        assert formatter.user_overrides == overrides

        # Test empty initialization
        formatter_empty = TreeHelpFormatter(prog="test")
        assert formatter_empty.user_overrides == {}


class TestFormatHelpWithDefaults:
    """Test the format_help_with_defaults function."""

    def test_basic_help_output(self):
        """Test basic help output without overrides."""
        help_text = format_help_with_defaults(
            TestConfig, prog_name="test_prog", has_positional_name=True, collapse=False
        )

        # Check that key elements are present
        assert "TEST-PROG - Experiment Runner" in help_text
        assert "Usage: test_prog [name] [options]" in help_text
        assert "Positional arguments:" in help_text
        assert "Experiment name (optional, defaults to recipe name)" in help_text
        assert "--gpus=1 {int}" in help_text
        assert "--spot=true {bool}" in help_text
        assert "--tags=['test', 'example'] {List[str]}" in help_text

    def test_nested_options_expanded(self):
        """Test that nested options are shown in expanded mode."""
        help_text = format_help_with_defaults(TestConfig, prog_name="test_prog", collapse=False)

        # Check for nested options with tree structure
        assert "--nested {object,null}" in help_text
        assert "├─ --nested.enabled=true {bool}" in help_text
        assert "├─ --nested.threshold=0.5 {float}" in help_text

    def test_nested_options_collapsed(self):
        """Test that nested options are hidden in collapsed mode."""
        help_text = format_help_with_defaults(TestConfig, prog_name="test_prog", collapse=True)

        # Check that nested options are NOT shown
        assert "--nested {object,null}" in help_text
        assert "--nested.enabled" not in help_text
        assert "--nested.threshold" not in help_text

    def test_user_overrides_highlighting(self, monkeypatch):
        """Test that user overrides are highlighted in red."""
        # Ensure colors are enabled
        monkeypatch.setattr("sys.stdout.isatty", lambda: True)

        user_overrides = {
            "gpus": 4,
            "spot": False,
            "nested.threshold": 0.8,  # Use dots for nested fields
        }

        help_text = format_help_with_defaults(TestConfig, prog_name="test_prog", user_overrides=user_overrides)

        # Check for bold red color codes around overridden values
        assert "\033[1;31m" in help_text  # Bold red color code
        assert "\033[1;31m=4\033[0m" in help_text  # Overridden gpus value in bold red
        assert "\033[1;31m=false\033[0m" in help_text  # Overridden spot value in bold red
        assert "\033[1;31m=0.8\033[0m" in help_text  # Overridden nested value in bold red

        # Check that non-overridden values are still in green
        assert "\033[92m=1\033[0m" in help_text  # Default nodes value in green

    def test_examples_section(self):
        """Test that examples section exists."""
        help_text = format_help_with_defaults(TestConfig, prog_name="test_prog")
        # Just verify the section exists, not the exact content
        assert "Examples:" in help_text

    def test_compact_help_note(self):
        """Test that compact help note appears in expanded mode."""
        help_text = format_help_with_defaults(TestConfig, prog_name="test_prog", collapse=False)

        assert "Note:" in help_text
        assert "For compact help without nested options" in help_text

        # Should not appear in collapsed mode
        help_text_collapsed = format_help_with_defaults(TestConfig, prog_name="test_prog", collapse=True)

        assert "For compact help without nested options" not in help_text_collapsed

    def test_no_positional_name(self):
        """Test help output without positional name argument."""
        help_text = format_help_with_defaults(TestConfig, prog_name="test_prog", has_positional_name=False)

        assert "Positional arguments:" not in help_text
        assert "Experiment name (optional" not in help_text
        assert "Usage: test_prog [name]" not in help_text


class TestFormattingEdgeCases:
    """Test edge cases in formatting."""

    def test_long_nested_paths(self):
        """Test formatting of very long nested configuration paths."""

        class VeryDeepConfig(BaseModel):
            very_long_configuration_option_name: str = "test"
            another_extremely_long_option: int = 42

        class DeepConfig(BaseModel):
            deep_settings: VeryDeepConfig = VeryDeepConfig()

        class RootConfig(BaseModel):
            advanced: DeepConfig = DeepConfig()

        help_text = format_help_with_defaults(RootConfig, prog_name="test_prog", collapse=False)

        # Check that long paths are handled properly
        assert "--advanced.deep-settings.very-long-configuration-option-name" in help_text
        assert "--advanced.deep-settings.another-extremely-long-option" in help_text

    def test_optional_types(self):
        """Test handling of optional types."""

        class OptionalConfig(BaseModel):
            optional_str: Optional[str] = None
            optional_int: Optional[int] = None
            optional_list: Optional[List[str]] = None

        help_text = format_help_with_defaults(OptionalConfig, prog_name="test_prog")

        # Check that optional types show ',null' suffix
        assert "{str,null}" in help_text
        assert "{int,null}" in help_text
        assert "{List[str],null}" in help_text

    def test_empty_defaults(self):
        """Test handling of empty defaults."""

        class EmptyDefaultsConfig(BaseModel):
            empty_list: List[str] = []
            empty_string: str = ""
            none_value: Optional[str] = None

        help_text = format_help_with_defaults(EmptyDefaultsConfig, prog_name="test_prog")

        # Check handling of empty values
        assert "--empty-list=[] {List[str]}" in help_text
        assert '--empty-string="" {str}' in help_text
        # None values should not show a default
        assert "--none-value {str,null}" in help_text
        assert "default: None" not in help_text
