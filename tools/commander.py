from typing import Any, List, Tuple, Union


class CommanderError(Exception):
    """Exception raised by the commander function for all error conditions."""

    pass


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
            apply_value(tree, key, value)

            i += consumed
        else:
            # Skip non-flag arguments
            i += 1

    return tree


def parse_arguments(arguments: str) -> List[str]:
    """Parse command line string into tokens, respecting quotes and escapes."""
    if not isinstance(arguments, str):
        raise CommanderError(f"Arguments must be a string, got {type(arguments).__name__}")

    tokens = []
    current_token = ""
    in_quotes = None
    escape_next = False
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
        elif not in_quotes and char.isspace():
            # Split on whitespace when not in quotes
            if current_token:
                tokens.append(current_token)
                current_token = ""
        else:
            current_token += char

        i += 1

    if current_token:
        tokens.append(current_token)

    # Check for parsing errors
    if in_quotes:
        raise CommanderError(f"Unclosed quote: {in_quotes}")

    return tokens


def parse_key_value(args: List[str], index: int) -> Tuple[str, Any, int]:
    """Parse a key-value pair from arguments starting at index.
    Returns (key, value, number_of_args_consumed)"""
    if index >= len(args):
        raise CommanderError(f"Index {index} out of range for arguments list")

    arg = args[index]

    # Check if it's a valid flag
    if not arg.startswith("-"):
        raise CommanderError(f"Expected flag starting with - or --, got: {arg}")

    # Remove leading dashes
    if arg.startswith("--"):
        arg = arg[2:]
    elif arg.startswith("-"):
        arg = arg[1:]

    # Check if we have a key after removing dashes
    if not arg:
        raise CommanderError("Empty flag after removing dashes")

    # Check for = or : separator
    if "=" in arg:
        parts = arg.split("=", 1)
        if len(parts) != 2 or not parts[0]:
            raise CommanderError(f"Invalid key=value format: {arg}")
        key, value_str = parts
        value = parse_value(value_str)
        return key, value, 1
    elif ":" in arg:
        parts = arg.split(":", 1)
        if len(parts) != 2 or not parts[0]:
            raise CommanderError(f"Invalid key:value format: {arg}")
        key, value_str = parts
        value = parse_value(value_str)
        return key, value, 1
    else:
        # Key is the whole arg, check if there's a value after it
        key = arg
        if index + 1 < len(args):
            next_arg = args[index + 1]
            # Check if next arg is a value (not another flag)
            # Special handling for negative numbers which start with -
            is_negative_number = (
                next_arg.startswith("-")
                and len(next_arg) > 1
                and next_arg[1:]
                .replace(".", "")
                .replace("e", "")
                .replace("E", "")
                .replace("+", "")
                .replace("-", "")
                .isdigit()
            )

            if not next_arg.startswith("-") or is_negative_number:
                # There's a value after the key
                value = parse_value(next_arg)
                return key, value, 2

        # No value, treat as boolean true
        return key, True, 1


def parse_value(value_str: str) -> Any:
    """Parse a value string into the appropriate type."""
    if not isinstance(value_str, str):
        raise CommanderError(f"Value must be a string, got {type(value_str).__name__}")

    value_str = value_str.strip()

    if not value_str:
        return ""

    # Handle quoted strings first
    if (value_str.startswith('"') and value_str.endswith('"')) or (
        value_str.startswith("'") and value_str.endswith("'")
    ):
        if len(value_str) < 2:
            raise CommanderError(f"Invalid quoted string: {value_str}")
        # Remove quotes and handle escapes
        return parse_quoted_string(value_str[1:-1])

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
        raise CommanderError(f"Values starting with '{value_str[0]}' must be quoted: {value_str}")

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
    if not s:
        return False

    # Handle signs
    if s.startswith(("+", "-")):
        if len(s) == 1:
            return False
        s = s[1:]

    # Basic validation - must contain at least one digit
    if not any(c.isdigit() for c in s):
        return False

    # Check for valid float format
    dot_count = s.count(".")
    if dot_count > 1:
        return False

    # Check for valid scientific notation
    e_count = s.lower().count("e")
    if e_count > 1:
        return False

    if e_count == 1:
        parts = s.lower().split("e")
        if len(parts) != 2:
            return False

        # Validate base part (before 'e')
        base_part = parts[0]
        if not base_part:  # Empty base part like "e10"
            return False

        # Base part should be a valid number (without 'e')
        if not (
            base_part.replace(".", "").isdigit()
            or (base_part.startswith(("+", "-")) and base_part[1:].replace(".", "").isdigit())
        ):
            return False

        # Validate exponent part (after 'e')
        exp_part = parts[1]
        if not exp_part:  # Empty exponent like "1e"
            return False

        if exp_part.startswith(("+", "-")):
            if len(exp_part) == 1:  # Just a sign like "1e+"
                return False
            exp_part = exp_part[1:]

        if not exp_part.isdigit():
            return False
    else:
        # No scientific notation - must be all digits with optional dot
        clean = s.replace(".", "")
        if not clean.isdigit():
            return False

    return True


def parse_number(s: str) -> Union[int, float]:
    """Parse a number string."""
    if not is_number(s):
        raise CommanderError(f"Invalid number format: {s}")

    # Safe to convert since we validated it
    if "." in s or "e" in s.lower():
        return float(s)
    else:
        return int(s)


def apply_value(tree: Any, key_path: str, value: Any) -> None:
    """Apply a value to the tree at the given key path."""
    if not key_path:
        raise CommanderError("Empty key path")

    keys = key_path.split(".")
    current = tree
    path = []

    # Navigate to the parent of the target
    for key in keys[:-1]:
        path.append(key)
        current_path = ".".join(path)

        if isinstance(current, dict):
            if key not in current:
                raise CommanderError(f"Key '{key}' not found at path '{current_path}'")
            current = current[key]
        elif isinstance(current, list):
            if not key.isdigit():
                raise CommanderError(f"List index must be numeric, got '{key}' at path '{current_path}'")
            index = int(key)
            if index < 0:
                raise CommanderError(f"List index cannot be negative: {index} at path '{current_path}'")
            if index >= len(current):
                raise CommanderError(
                    f"List index {index} out of range (length {len(current)}) at path '{current_path}'"
                )
            current = current[index]
        else:
            # Handle Python objects with attributes
            if not hasattr(current, key):
                raise CommanderError(
                    f"Attribute '{key}' not found on {type(current).__name__} at path '{current_path}'"
                )
            current = getattr(current, key)

    # Apply the value
    final_key = keys[-1]
    path.append(final_key)
    final_path = ".".join(path)

    if isinstance(current, dict):
        if final_key not in current:
            raise CommanderError(f"Key '{final_key}' not found at path '{final_path}'")

        # Type check
        old_value = current[final_key]
        if old_value is not None and not isinstance(value, type(old_value)):
            # Allow int/float conversions
            if isinstance(old_value, (int, float)) and isinstance(value, (int, float)):
                pass
            else:
                raise CommanderError(
                    f"Type mismatch at path '{final_path}': "
                    f"expected {type(old_value).__name__}, got {type(value).__name__}"
                )

        current[final_key] = value
    elif isinstance(current, list):
        if not final_key.isdigit():
            raise CommanderError(f"List index must be numeric, got '{final_key}' at path '{final_path}'")

        index = int(final_key)
        if index < 0:
            raise CommanderError(f"List index cannot be negative: {index} at path '{final_path}'")
        if index >= len(current):
            raise CommanderError(f"List index {index} out of range (length {len(current)}) at path '{final_path}'")

        # Type check
        old_value = current[index]
        if old_value is not None and not isinstance(value, type(old_value)):
            # Allow int/float conversions
            if isinstance(old_value, (int, float)) and isinstance(value, (int, float)):
                pass
            else:
                raise CommanderError(
                    f"Type mismatch at path '{final_path}': "
                    f"expected {type(old_value).__name__}, got {type(value).__name__}"
                )

        current[index] = value
    else:
        # Handle Python objects with attributes
        if not hasattr(current, final_key):
            raise CommanderError(
                f"Attribute '{final_key}' not found on {type(current).__name__} at path '{final_path}'"
            )

        # Type check
        old_value = getattr(current, final_key)
        if old_value is not None and not isinstance(value, type(old_value)):
            # Allow int/float conversions
            if isinstance(old_value, (int, float)) and isinstance(value, (int, float)):
                pass
            else:
                raise CommanderError(
                    f"Type mismatch at path '{final_path}': "
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
