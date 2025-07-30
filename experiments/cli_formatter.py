"""Custom CLI formatter for better help display with defaults and tree layout."""

import argparse
import sys
from typing import Dict, Any, Optional, List, Type
from pydantic import BaseModel


class TreeHelpFormatter(argparse.HelpFormatter):
    """Custom help formatter that displays arguments in a tree structure with defaults."""

    # ANSI color codes
    COLORS = {
        "option": "\033[94m",  # Blue for option names
        "default": "\033[92m",  # Green for default values
        "type": "\033[90m",  # Gray for types
        "tree": "\033[90m",  # Gray for tree symbols
        "reset": "\033[0m",  # Reset
        "header": "\033[1m",  # Bold for headers
        "override": "\033[1;31m",  # Bold red for user overrides (like git diff)
    }

    def __init__(
        self,
        *args,
        collapse: bool = False,
        use_colors: bool = True,
        user_overrides: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        # Increase width for better tree display
        if "max_help_position" not in kwargs:
            kwargs["max_help_position"] = 50
        if "width" not in kwargs:
            kwargs["width"] = 120
        super().__init__(*args, **kwargs)
        self.collapse = collapse
        self.use_colors = (
            use_colors and sys.stdout.isatty()
        )  # Only use colors in terminal
        self.user_overrides = user_overrides or {}
        self._current_indent = 0
        self._tree_lines = []
        self._grouped_actions = {}

    def _format_action(self, action):
        """Format a single action (argument) with tree structure."""
        if action.dest == "help":
            return super()._format_action(action)

        # Skip if in collapse mode and this is a nested option
        if self.collapse and "." in action.dest:
            return ""

        parts = action.dest.split(".")
        level = len(parts) - 1

        # Helper to add color
        def color(text, color_name):
            if self.use_colors and color_name in self.COLORS:
                return f"{self.COLORS[color_name]}{text}{self.COLORS['reset']}"
            return text

        # Build the tree prefix with proper nesting
        prefix = ""
        indent = ""
        if level > 0:
            # For first level, add two spaces before the tree
            indent = "  "
            # Add tree branches for each level
            prefix_parts = []
            for i in range(level):
                prefix_parts.append(color("├─", "tree"))
            prefix = "".join(prefix_parts) + " "

        # Get the option string
        option_strings = action.option_strings
        if not option_strings:
            return ""

        # Format option name - always show the full path
        opt_name = (
            option_strings[0]
            if option_strings
            else "--" + action.dest.replace("_", "-")
        )
        opt_name = color(opt_name, "option")

        # Get the type from metavar
        type_str = ""
        if action.metavar:
            # Preserve the full metavar including any ,null suffix
            metavar = action.metavar.strip("{}")
            type_str = color(f" {{{metavar}}}", "type")

        # Get default value from help text
        default_val = None
        help_text = action.help or ""
        if "default:" in help_text:
            # Extract default value from help text
            parts = help_text.split("default:")
            if len(parts) > 1:
                default_val = parts[1].strip()
                # Remove the default from help text since we'll show it differently
                help_text = parts[0].strip()

        # Check if this option has been overridden by the user
        # Handle nested keys by converting dots to underscores
        override_key = action.dest
        is_overridden = override_key in self.user_overrides

        # Also check for nested keys with dots
        if not is_overridden and "." in override_key:
            # Try with dots preserved (for nested fields)
            dotted_key = override_key.replace("_", ".")
            is_overridden = dotted_key in self.user_overrides
            if is_overridden:
                override_key = dotted_key

        if is_overridden:
            # Use the user-provided value instead of the default
            override_val = self.user_overrides[override_key]
            # Format the override value
            if isinstance(override_val, bool):
                display_val = str(override_val).lower()
            elif isinstance(override_val, str):
                display_val = f'"{override_val}"'
            elif isinstance(override_val, list):
                display_val = "[]" if not override_val else str(override_val)
            else:
                display_val = str(override_val)
        else:
            display_val = default_val

        # Format the main option part
        if display_val:
            # For better alignment, split into option and value parts
            option_display = f"{opt_name}"
            # Use red color for overrides, green for defaults
            value_color = "override" if is_overridden else "default"
            default_display = color(f"={display_val}", value_color)
            option_part = f"{indent}{prefix}{option_display}{default_display}{type_str}"
        else:
            option_part = f"{indent}{prefix}{opt_name}{type_str}"

        # Calculate actual display length (without ANSI codes)
        def strip_ansi(text):
            import re

            return re.sub(r"\033\[[0-9;]*m", "", text)

        display_len = len(strip_ansi(option_part))

        # Calculate spacing for alignment
        spacing = max(2, self._max_help_position - display_len)

        # Add any additional help text
        additional_text = help_text if help_text else ""

        # For long paths, put help text on next line
        if display_len > self._max_help_position - 5:
            if additional_text:
                line = f"{option_part}\n{' ' * (len(indent) + 4)}{additional_text}\n"
            else:
                line = f"{option_part}\n"
        else:
            # Normal tabular layout
            if additional_text:
                import textwrap

                wrapper = textwrap.TextWrapper(
                    width=self._width - self._max_help_position,
                    initial_indent="",
                    subsequent_indent=" " * (self._max_help_position + 2),
                )
                wrapped_text = wrapper.fill(additional_text)
                line = f"{option_part}{' ' * spacing}{wrapped_text}\n"
            else:
                line = f"{option_part}\n"

        return line

    def _format_usage(self, usage, actions, groups, prefix):
        """Format usage line."""
        # Suppress the default usage line - we show our own custom one
        return ""

    def add_usage(self, usage, actions, groups, prefix=None):
        """Add usage line."""
        if usage is not argparse.SUPPRESS:
            args = usage, actions, groups, prefix
            self._add_item(self._format_usage, args)


def format_help_with_defaults(
    config_class: Type[BaseModel],
    prog_name: str,
    has_positional_name: bool = True,
    collapse: bool = False,
    user_overrides: Optional[Dict[str, Any]] = None,
    patched_defaults: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate help text with defaults from a Pydantic model.

    Args:
        config_class: The Pydantic config class
        prog_name: Program name for help
        has_positional_name: Whether the first positional arg is a name
        collapse: Whether to collapse nested options
        user_overrides: Dictionary of user-provided values to highlight in red
        patched_defaults: Dictionary of defaults to override the model defaults

    Returns:
        Formatted help string
    """

    # Create a parser with our custom formatter
    parser = argparse.ArgumentParser(
        prog=prog_name,
        formatter_class=lambda prog: TreeHelpFormatter(
            prog, collapse=collapse, user_overrides=user_overrides
        ),
        add_help=False,
    )

    # Add help action
    parser.add_argument(
        "-h", "--help", action="help", help="show this help message and exit"
    )

    # Add collapse option
    parser.add_argument(
        "--help-compact",
        action="store_true",
        help="show compact help without nested options",
    )

    # Extract fields from the config class and add them as arguments
    def add_fields(model_class: Type[BaseModel], prefix: str = "", level: int = 0):
        """Recursively add fields from a Pydantic model."""
        # Get all parent classes to collect fields from inheritance chain
        mro = model_class.__mro__
        all_fields = {}

        # Collect fields from all parent classes
        for cls in reversed(mro):
            if hasattr(cls, "model_fields"):
                all_fields.update(cls.model_fields)

        for field_name, field_info in all_fields.items():
            full_name = f"{prefix}{field_name}" if prefix else field_name

            # Skip if in collapse mode and this is a nested field
            if collapse and level > 0:
                continue

            # Get field type and default
            field_type = field_info.annotation
            default = field_info.default

            # Skip if default is a special pydantic marker
            if (
                hasattr(default, "__class__")
                and default.__class__.__name__ == "PydanticUndefined"
            ):
                default = None

            # Check if we have a patched default for this field
            if patched_defaults and full_name in patched_defaults:
                default = patched_defaults[full_name]

            # Handle optional types
            import typing

            type_origin = typing.get_origin(field_type)
            type_args = typing.get_args(field_type)

            # Check if it's Optional (Union with None)
            is_optional = False
            if type_origin is typing.Union:
                if type(None) in type_args:
                    is_optional = True
                    # Get the non-None type
                    non_none_types = [t for t in type_args if t != type(None)]
                    if non_none_types:
                        field_type = non_none_types[0]
                        type_origin = typing.get_origin(field_type)
                        type_args = typing.get_args(field_type)

            # If default is None, it's implicitly optional
            if default is None:
                is_optional = True

            # Determine the metavar based on type
            if type_origin in (list, typing.List) if "typing" in locals() else (list,):
                metavar = "List[str]" + (",null" if is_optional else "")
            elif (
                type_origin in (dict, typing.Dict) if "typing" in locals() else (dict,)
            ):
                metavar = "JSON" + (",null" if is_optional else "")
            elif field_type == bool:
                metavar = "bool" + (",null" if is_optional else "")
            elif field_type == int:
                metavar = "int" + (",null" if is_optional else "")
            elif field_type == float:
                metavar = "float" + (",null" if is_optional else "")
            elif field_type == str or field_type is str:
                metavar = "str" + (",null" if is_optional else "")
            elif hasattr(field_type, "model_fields"):
                # This is a nested Pydantic model
                metavar = "object" + (",null" if is_optional else "")
            else:
                metavar = "value"

            # Format default value for display
            default_str = ""
            if default is not None:
                if isinstance(default, bool):
                    default_str = str(default).lower()
                elif isinstance(default, str):
                    default_str = f'"{default}"'
                elif isinstance(default, list):
                    default_str = "[]" if not default else str(default)
                elif hasattr(default.__class__, "model_fields"):
                    # For nested Pydantic models, don't show the full repr
                    default_str = None
                else:
                    default_str = str(default)

            # Get description
            description = field_info.description or ""
            if default_str is not None and default_str != "":
                # Always add default to description so _format_action can extract it
                if description:
                    description += f" (default: {default_str})"
                else:
                    description = f"default: {default_str}"

            # Add the argument
            parser.add_argument(
                f"--{full_name.replace('_', '-')}",
                metavar=f"{{{metavar}}}",
                default=argparse.SUPPRESS,  # Don't show in help
                help=description,
            )

            # If this field is a Pydantic model, recurse
            if hasattr(field_type, "model_fields"):
                add_fields(field_type, prefix=f"{full_name}.", level=level + 1)

    # Add fields from the config class
    add_fields(config_class)

    # Mark that we have a positional name argument
    if has_positional_name:
        parser.formatter_class._has_positional_name = True

    # Generate help
    help_text = parser.format_help()

    # Helper for colors
    def color(text, color_name):
        if sys.stdout.isatty() and color_name in TreeHelpFormatter.COLORS:
            return f"{TreeHelpFormatter.COLORS[color_name]}{text}{TreeHelpFormatter.COLORS['reset']}"
        return text

    # Add a header section
    header = f"\n{color(prog_name.upper().replace('_', '-'), 'header')} - Experiment Runner\n"
    if has_positional_name:
        header += f"\n{color('Usage:', 'header')} {prog_name} [name] [options]\n"
        header += f"\n{color('Positional arguments:', 'header')}\n"
        header += f"  {color('name', 'option')}                    Experiment name (optional, defaults to recipe name)\n"

    # Add examples section
    examples = f"\n{color('Examples:', 'header')}\n"
    examples += "  metta notebook arena my_experiment --gpus 2\n"
    examples += "  metta notebook arena --gpus 4 --nodes 2 --launch\n"
    examples += "  metta notebook arena test_run --output_dir=None  # Run without creating notebook\n"

    # Add note about compact help
    if not collapse:
        examples += f"\n{color('Note:', 'header')} For compact help without nested options, use: metta notebook arena --help-compact\n"

    return header + "\n" + help_text + examples


def setup_cli_parser(
    config_class: Type[BaseModel],
    prog_name: str,
    args_to_parse: List[str],
    has_positional_name: bool = True,
) -> Dict[str, Any]:
    """Set up CLI parser with custom help handling.

    Returns:
        Parsed arguments as a dictionary
    """
    # Check if user wants help
    if "--help" in args_to_parse or "-h" in args_to_parse:
        # Check if --help-full is also present
        collapse = "--help-full" not in args_to_parse
        help_text = format_help_with_defaults(
            config_class,
            prog_name,
            has_positional_name=has_positional_name,
            collapse=collapse,
        )
        print(help_text)
        raise SystemExit(0)

    # Otherwise, use pydantic-settings for actual parsing
    # This will be handled by the existing runner code
    return {}
