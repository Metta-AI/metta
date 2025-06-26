"""
Utility for checking if scripts are being run with uv.

This module provides functions to detect whether a Python script is being
executed via `uv run` and can enforce this requirement with helpful error messages.
"""

import os
import sys
from typing import Optional


def is_running_under_uv() -> bool:
    """
    Check if the current Python process is running under uv.

    Returns:
        True if running under uv, False otherwise.
    """
    # Check for uv-specific environment variables
    if "UV_PROJECT_ROOT" in os.environ or "UV_CACHE_DIR" in os.environ:
        return True

    # Check if uv is in the executable path
    if "uv" in sys.executable:
        return True

    # Try to check process tree if psutil is available
    try:
        import psutil

        current_process = psutil.Process()
        parent = current_process.parent()

        # Walk up the process tree looking for uv
        for _ in range(5):  # Check up to 5 levels up
            if parent is None:
                break
            if "uv" in parent.name().lower():
                return True
            try:
                parent = parent.parent()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break

    except ImportError:
        # psutil not available, fall back to other checks
        pass

    return False


def require_uv_execution(
    script_name: Optional[str] = None, exit_code: int = 1, custom_message: Optional[str] = None
) -> None:
    """
    Enforce that the script is being run with `uv run`.

    If not running under uv, prints an error message and exits.

    Args:
        script_name: Name of the script to show in error message.
                    If None, uses sys.argv[0].
        exit_code: Exit code to use when terminating (default: 1).
        custom_message: Custom error message. If None, uses default message.
    """
    if is_running_under_uv():
        return

    if script_name is None:
        script_name = sys.argv[0] if sys.argv else "this_script.py"

    if custom_message:
        print(custom_message, file=sys.stderr)
    else:
        print("❌ ERROR: This script should be run with 'uv run' not 'python'", file=sys.stderr)
        print("", file=sys.stderr)
        print("Instead of:", file=sys.stderr)
        print(f"  python {script_name}", file=sys.stderr)
        print("", file=sys.stderr)
        print("Run:", file=sys.stderr)
        print(f"  uv run {script_name}", file=sys.stderr)
        print("", file=sys.stderr)
        print("This ensures proper dependency management and virtual environment isolation.", file=sys.stderr)

    sys.exit(exit_code)


def warn_if_not_uv(script_name: Optional[str] = None, custom_message: Optional[str] = None) -> bool:
    """
    Print a warning if not running under uv, but don't exit.

    Args:
        script_name: Name of the script to show in warning message.
                    If None, uses sys.argv[0].
        custom_message: Custom warning message. If None, uses default message.

    Returns:
        True if running under uv, False if warning was printed.
    """
    if is_running_under_uv():
        return True

    if script_name is None:
        script_name = sys.argv[0] if sys.argv else "this_script.py"

    if custom_message:
        print(custom_message, file=sys.stderr)
    else:
        print("⚠️  WARNING: This script is intended to be run with 'uv run'", file=sys.stderr)
        print("", file=sys.stderr)
        print("For better dependency management, consider running:", file=sys.stderr)
        print(f"  uv run {script_name}", file=sys.stderr)
        print("", file=sys.stderr)

    return False


# Convenience function for the most common use case
def enforce_uv() -> None:
    """
    Convenience function that enforces uv execution with default settings.

    Equivalent to calling require_uv_execution() with no arguments.
    """
    require_uv_execution()


if __name__ == "__main__":
    # python common/src/metta/common/uv_check.py (should show errors/warnings)
    # uv run common/src/metta/common/uv_check.py (should show success)
    enforce_uv()
    print("✅ Great! You're running this demo with uv.")
