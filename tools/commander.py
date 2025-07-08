import ast
import json
import re
from typing import Any, List, Tuple, Union


def commander(arguments: str, tree: Any) -> Any:
    """
    This function takes command line arguments and applies them to a possibly very nested and complex tree of objects.
    See the spec at docs/commander.md for more details.
    """
    # If arguments is actually a list (like sys.argv), join them
    if isinstance(arguments, list):
        arguments = " ".join(arguments)

    # Split arguments into tokens
    args = parse_arguments(arguments)

    # Handle help
    if any(arg in ["-h", "--help"] for arg in args):
        print_help(tree)
        return tree

    # Process each argument
    i = 0
    while i < len(args):
        arg = args[i]

        # Check if it's a flag/option
        if arg.startswith("-"):
            # Parse the key and potential value
            key, value, consumed = parse_key_value(args, i)

            # Apply the value to the tree
            try:
                apply_value(tree, key, value)
            except Exception as e:
                # Enhanced error message
                error_msg = f"Error setting '{key}' to {repr(value)}: {str(e)}"
                raise ValueError(error_msg) from e

            i += consumed
        else:
            # Skip non-flag arguments
            i += 1

    return tree


def parse_arguments(arguments: str) -> List[str]:
    """Parse command line string into tokens, respecting quotes and escapes."""
    tokens = []
    current_token = ""
    in_quotes = None
    escape_next = False
    brace_depth = 0  # Track { } nesting
    bracket_depth = 0  # Track [ ] nesting
    i = 0

    while i < len(arguments):
        char = arguments[i]

        if escape_next:
            # Handle escape sequences
            escape_map = {
                "n": "\n",
                "t": "\t",
                "r": "\r",
                "b": "\b",
                "f": "\f",
                "\\": "\\",
                '"': '"',
                "'": "'",
                "0": "\0",
            }
            current_token += escape_map.get(char, char)
            escape_next = False
        elif char == "\\" and in_quotes:
            escape_next = True
        elif char in ['"', "'"] and not in_quotes:
            in_quotes = char
            current_token += char  # Keep the quote
        elif char == in_quotes:
            in_quotes = None
            current_token += char  # Keep the quote
        elif not in_quotes:
            # Track brace/bracket depth when not in quotes
            if char == "{":
                brace_depth += 1
                current_token += char
            elif char == "}":
                brace_depth -= 1
                current_token += char
            elif char == "[":
                bracket_depth += 1
                current_token += char
            elif char == "]":
                bracket_depth -= 1
                current_token += char
            elif char.isspace() and brace_depth == 0 and bracket_depth == 0:
                # Only split on whitespace when not inside braces/brackets
                if current_token:
                    tokens.append(current_token)
                    current_token = ""
            else:
                current_token += char
        else:
            current_token += char

        i += 1

    if current_token:
        tokens.append(current_token)

    if in_quotes:
        raise ValueError(f"Unclosed quote: {in_quotes}")
    if brace_depth != 0:
        raise ValueError(f"Unmatched braces: depth={brace_depth}")
    if bracket_depth != 0:
        raise ValueError(f"Unmatched brackets: depth={bracket_depth}")

    return tokens


def parse_key_value(args: List[str], index: int) -> Tuple[str, Any, int]:
    """Parse a key-value pair from arguments starting at index.
    Returns (key, value, number_of_args_consumed)"""
    arg = args[index]

    # Remove leading dashes
    if arg.startswith("--"):
        arg = arg[2:]
    elif arg.startswith("-"):
        arg = arg[1:]
    else:
        raise ValueError(f"Expected flag starting with - or --, got: {arg}")

    # Check for = or : separator
    if "=" in arg:
        key, value_str = arg.split("=", 1)
        value = parse_value(value_str)
        return key, value, 1
    elif ":" in arg:
        key, value_str = arg.split(":", 1)
        value = parse_value(value_str)
        return key, value, 1
    else:
        # Key is the whole arg, check if there's a value after it
        key = arg
        if index + 1 < len(args):
            next_arg = args[index + 1]
            # Check if next arg is a value (not another flag)
            # Special handling for negative numbers which start with -
            if not next_arg.startswith("-") or (
                next_arg[1:].replace(".", "").replace("e", "").replace("+", "").lstrip("-").isdigit()
            ):
                # There's a value after the key
                value = parse_value(next_arg)
                return key, value, 2

        # No value, treat as boolean true
        return key, True, 1


def parse_value(value_str: str) -> Any:
    """Parse a value string into the appropriate type."""
    value_str = value_str.strip()

    if not value_str:
        return ""

    # Handle quoted strings first
    if (value_str.startswith('"') and value_str.endswith('"')) or (
        value_str.startswith("'") and value_str.endswith("'")
    ):
        # Remove quotes and handle escapes
        return parse_quoted_string(value_str[1:-1])

    # Handle JSON-like values
    if value_str.startswith("{") or value_str.startswith("["):
        return parse_json5(value_str)

    # Handle boolean literals
    if value_str.lower() == "true":
        return True
    elif value_str.lower() == "false":
        return False
    elif value_str.lower() == "null" or value_str.lower() == "none":
        return None

    # Handle numbers
    if is_number(value_str):
        return parse_number(value_str)

    # Handle plain words as strings (if they start with a letter)
    if value_str and value_str[0].isalpha():
        return value_str

    # Special case: negative numbers were already handled above
    # Everything else that starts with non-letter should have been quoted
    if value_str and not value_str[0].isalpha():
        raise ValueError(f"Values starting with '{value_str[0]}' must be quoted: {value_str}")

    return value_str


def parse_quoted_string(s: str) -> str:
    """Parse a quoted string handling escape sequences."""
    result = ""
    escape_next = False

    for char in s:
        if escape_next:
            escape_map = {
                "n": "\n",
                "t": "\t",
                "r": "\r",
                "b": "\b",
                "f": "\f",
                "\\": "\\",
                '"': '"',
                "'": "'",
                "0": "\0",
            }
            result += escape_map.get(char, char)
            escape_next = False
        elif char == "\\":
            escape_next = True
        else:
            result += char

    return result


def is_number(s: str) -> bool:
    """Check if a string represents a number."""
    # Handle empty string
    if not s:
        return False

    # Handle signs
    if s.startswith(("+", "-")):
        if len(s) == 1:
            return False
        s = s[1:]

    # Check for valid number patterns
    try:
        float(s)
        return True
    except ValueError:
        return False


def parse_number(s: str) -> Union[int, float]:
    """Parse a number string."""
    num = float(s)
    # If it's a whole number, return as int
    if num.is_integer():
        return int(num)
    return num


def parse_json5(s: str) -> Any:
    """Parse JSON5-like syntax (relaxed JSON)."""
    # First try standard JSON
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    # JSON5 relaxations:
    # 1. Unquoted keys
    # 2. Single quotes
    # 3. Trailing commas
    # 4. Comments (not implemented here)

    # Handle single quotes by replacing with double quotes (careful with escapes)
    relaxed = s

    # Convert single quotes to double quotes (simple approach)
    # This is not perfect but handles most cases
    if "'" in relaxed:
        # Replace single quotes that are not escaped
        parts = []
        i = 0
        in_double_quotes = False
        while i < len(relaxed):
            if relaxed[i] == '"' and (i == 0 or relaxed[i - 1] != "\\"):
                in_double_quotes = not in_double_quotes
                parts.append(relaxed[i])
            elif relaxed[i] == "'" and not in_double_quotes and (i == 0 or relaxed[i - 1] != "\\"):
                parts.append('"')
            else:
                parts.append(relaxed[i])
            i += 1
        relaxed = "".join(parts)

    # Convert unquoted keys to quoted keys
    # Match word characters followed by colon
    relaxed = re.sub(r"([{,]\s*)(\w+)(\s*:)", r'\1"\2"\3', relaxed)
    relaxed = re.sub(r"^(\s*)(\w+)(\s*:)", r'\1"\2"\3', relaxed)

    # Remove trailing commas
    relaxed = re.sub(r",(\s*[}\]])", r"\1", relaxed)

    # Try parsing again
    try:
        return json.loads(relaxed)
    except Exception as e:
        # As a fallback, try using ast.literal_eval for simple cases
        try:
            return ast.literal_eval(s)
        except Exception:
            raise ValueError(f"Unable to parse JSON5 value: {s}. Error: {str(e)}") from e


def apply_value(tree: Any, key_path: str, value: Any) -> None:
    """Apply a value to the tree at the given key path."""
    keys = key_path.split(".")
    current = tree
    path = []

    # Navigate to the parent of the target
    for key in keys[:-1]:
        path.append(key)

        if isinstance(current, dict):
            if key not in current:
                raise KeyError(f"Key '{key}' not found at path {'.'.join(path)}")
            current = current[key]
        elif isinstance(current, list):
            try:
                index = int(key)
                if index < 0 or index >= len(current):
                    raise IndexError(f"Index {index} out of range for list at path {'.'.join(path[:-1])}")
                current = current[index]
            except ValueError as e:
                raise TypeError(f"Cannot use non-numeric key '{key}' for list at path {'.'.join(path[:-1])}") from e
        else:
            # Handle Python objects with attributes
            if not hasattr(current, key):
                raise KeyError(f"Attribute '{key}' not found at path {'.'.join(path)}")
            current = getattr(current, key)

    # Apply the value
    final_key = keys[-1]
    path.append(final_key)

    if isinstance(current, dict):
        if final_key not in current:
            raise KeyError(f"Key '{final_key}' not found at path {'.'.join(path)}")

        # Type check
        old_value = current[final_key]
        if old_value is not None and not isinstance(old_value, type(value)):
            # Allow int/float conversions
            if isinstance(old_value, (int, float)) and isinstance(value, (int, float)):
                pass
            # Allow None to be replaced by any type
            elif old_value is None:
                pass
            else:
                raise TypeError(
                    f"Type mismatch at path {'.'.join(path)}: "
                    f"expected {type(old_value).__name__}, got {type(value).__name__}"
                )

        current[final_key] = value
    elif isinstance(current, list):
        try:
            index = int(final_key)
            if index < 0 or index >= len(current):
                raise IndexError(f"Index {index} out of range for list at path {'.'.join(path[:-1])}")

            # Type check
            old_value = current[index]
            if old_value is not None and not isinstance(old_value, type(value)):
                # Allow int/float conversions
                if isinstance(old_value, (int, float)) and isinstance(value, (int, float)):
                    pass
                # Allow None to be replaced by any type
                elif old_value is None:
                    pass
                else:
                    raise TypeError(
                        f"Type mismatch at path {'.'.join(path)}: "
                        f"expected {type(old_value).__name__}, got {type(value).__name__}"
                    )

            current[index] = value
        except ValueError as e:
            raise TypeError(f"Cannot use non-numeric key '{final_key}' for list at path {'.'.join(path[:-1])}") from e
    else:
        # Handle Python objects with attributes
        if not hasattr(current, final_key):
            raise KeyError(f"Attribute '{final_key}' not found at path {'.'.join(path)}")

        # Type check
        old_value = getattr(current, final_key)
        if old_value is not None and not isinstance(old_value, type(value)):
            # Allow int/float conversions
            if isinstance(old_value, (int, float)) and isinstance(value, (int, float)):
                pass
            # Allow None to be replaced by any type
            elif old_value is None:
                pass
            else:
                raise TypeError(
                    f"Type mismatch at path {'.'.join(path)}: "
                    f"expected {type(old_value).__name__}, got {type(value).__name__}"
                )

        setattr(current, final_key, value)


def print_help(tree: Any, indent: int = 0, path: str = "") -> None:
    """Print the tree structure with types and doc comments."""
    prefix = "  " * indent

    if isinstance(tree, dict):
        for key, value in sorted(tree.items()):
            current_path = f"{path}.{key}" if path else key

            if isinstance(value, (dict, list)) or (
                hasattr(value, "__dict__") and not isinstance(value, (str, int, float, bool))
            ):
                print(f"{prefix}{key}:")
                print_help(value, indent + 1, current_path)
            else:
                type_name = type(value).__name__
                if value is None:
                    print(f"{prefix}--{current_path}: {type_name}")
                else:
                    print(f"{prefix}--{current_path}: {type_name} = {repr(value)}")
    elif isinstance(tree, list):
        for i, value in enumerate(tree):
            current_path = f"{path}.{i}"

            if isinstance(value, (dict, list)) or (
                hasattr(value, "__dict__") and not isinstance(value, (str, int, float, bool))
            ):
                print(f"{prefix}[{i}]:")
                print_help(value, indent + 1, current_path)
            else:
                type_name = type(value).__name__
                if value is None:
                    print(f"{prefix}--{current_path}: {type_name}")
                else:
                    print(f"{prefix}--{current_path}: {type_name} = {repr(value)}")
    else:
        # Handle Python objects with attributes
        if hasattr(tree, "__dict__") and not isinstance(tree, (str, int, float, bool)):
            # Get all non-private attributes
            attrs = {k: v for k, v in vars(tree).items() if not k.startswith("_")}
            for key, value in sorted(attrs.items()):
                current_path = f"{path}.{key}" if path else key

                if isinstance(value, (dict, list)) or (
                    hasattr(value, "__dict__") and not isinstance(value, (str, int, float, bool))
                ):
                    print(f"{prefix}{key}:")
                    print_help(value, indent + 1, current_path)
                else:
                    type_name = type(value).__name__
                    if value is None:
                        print(f"{prefix}--{current_path}: {type_name}")
                    else:
                        print(f"{prefix}--{current_path}: {type_name} = {repr(value)}")
        else:
            type_name = type(tree).__name__
            print(f"{prefix}{type_name} = {repr(tree)}")
