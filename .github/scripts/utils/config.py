import os
import sys


def parse_config(required_vars: list[str], optional_vars: dict[str, str] | None = None) -> dict[str, str]:
    """
    Parse and validate environment variables.

    Args:
        required_vars: List of `var_name` for required variables
        optional_vars: Dict of `{var_name: default value}` for optional variables

    Returns:
        Dict of {var_name: value} for all available env variables

    Raises:
        SystemExit: If any required variables are missing
    """
    if optional_vars is None:
        optional_vars = {}

    env_values = {}
    missing_vars = []

    # Validate required environment variables
    for var_name in required_vars:
        value = os.getenv(var_name)
        if not value:
            missing_vars.append(var_name)
        else:
            env_values[var_name] = value

    # Report all missing variables at once
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"  - {var}")
        sys.exit(1)

    # Get optional environment variables
    for var_name, default_value in optional_vars.items():
        env_values[var_name] = os.getenv(var_name, default_value)

    return env_values
