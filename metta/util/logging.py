import logging
import os
import shutil
import subprocess
import sys
from datetime import datetime

from rich.logging import RichHandler


def remap_io(logs_path: str):
    os.makedirs(logs_path, exist_ok=True)
    stdout_log_path = os.path.join(logs_path, "out.log")
    stderr_log_path = os.path.join(logs_path, "error.log")
    stdout = open(stdout_log_path, "a")
    stderr = open(stderr_log_path, "a")
    sys.stderr = stderr
    sys.stdout = stdout
    # Remove all handlers from root logger when remapping IO
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)


def restore_io():
    sys.stderr = sys.__stderr__
    sys.stdout = sys.__stdout__


# Create a custom formatter that supports milliseconds
class MillisecondFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        created = datetime.fromtimestamp(record.created)
        # Convert microseconds to milliseconds (keep only 3 digits)
        msec = created.microsecond // 1000
        if datefmt:
            # Replace %f with just 3 digits for milliseconds
            datefmt = datefmt.replace("%f", f"{msec:03d}")
        else:
            datefmt = "[%H:%M:%S.%03d]"
        return created.strftime(datefmt) % msec


# Create a custom handler that always shows the timestamp
class AlwaysShowTimeRichHandler(RichHandler):
    def emit(self, record):
        # Force a unique timestamp for each record
        record.created = record.created + (record.relativeCreated % 1000) / 1000000
        super().emit(record)


def get_log_level(provided_level=None):
    """
    Determine log level based on priority:
    1. Environment variable LOG_LEVEL
    2. Provided level parameter
    3. Default to INFO
    """
    # Check environment variable first
    env_level = os.environ.get("LOG_LEVEL")
    if env_level:
        return env_level.upper()

    # Check provided level next
    if provided_level:
        return provided_level.upper()

    # Default to INFO
    return "INFO"


def get_terminal_info():
    """Get comprehensive terminal information for debugging"""
    info = {}

    # Basic terminal info
    info["isatty_stdout"] = sys.stdout.isatty()
    info["isatty_stderr"] = sys.stderr.isatty()
    info["isatty_stdin"] = sys.stdin.isatty()

    # Terminal size from shutil
    try:
        size = shutil.get_terminal_size()
        info["shutil_columns"] = size.columns
        info["shutil_lines"] = size.lines
    except Exception as e:
        info["shutil_error"] = str(e)

    # Terminal size from os (if available)
    if hasattr(os, "get_terminal_size"):
        try:
            size = os.get_terminal_size()
            info["os_columns"] = size.columns
            info["os_lines"] = size.lines
        except Exception as e:
            info["os_error"] = str(e)

    # Try stty command
    try:
        result = subprocess.run(["stty", "size"], capture_output=True, text=True)
        if result.returncode == 0:
            lines, cols = result.stdout.strip().split()
            info["stty_columns"] = int(cols)
            info["stty_lines"] = int(lines)
        else:
            info["stty_error"] = result.stderr
    except Exception as e:
        info["stty_error"] = str(e)

    # Environment variables
    env_vars = {
        "COLUMNS": os.environ.get("COLUMNS"),
        "LINES": os.environ.get("LINES"),
        "TERM": os.environ.get("TERM"),
        "PYTHONUNBUFFERED": os.environ.get("PYTHONUNBUFFERED"),
        "AWS_BATCH_JOB_ID": os.environ.get("AWS_BATCH_JOB_ID"),
        "AWS_BATCH_JOB_ATTEMPT": os.environ.get("AWS_BATCH_JOB_ATTEMPT"),
        "AWS_BATCH_CE_NAME": os.environ.get("AWS_BATCH_CE_NAME"),
        "LOG_LEVEL": os.environ.get("LOG_LEVEL"),
    }
    info["env_vars"] = env_vars

    # Rich console settings (if available)
    try:
        from rich.console import Console

        console = Console()
        info["rich_width"] = console.width
        info["rich_height"] = console.height
        info["rich_is_terminal"] = console.is_terminal
        info["rich_is_jupyter"] = console.is_jupyter
        info["rich_is_dumb_terminal"] = console.is_dumb_terminal
        info["rich_legacy_windows"] = console.legacy_windows
        info["rich_color_system"] = str(console.color_system)
    except Exception as e:
        info["rich_error"] = str(e)

    # Python stdout/stderr encoding
    info["stdout_encoding"] = sys.stdout.encoding if hasattr(sys.stdout, "encoding") else None
    info["stderr_encoding"] = sys.stderr.encoding if hasattr(sys.stderr, "encoding") else None

    # Buffer settings
    info["stdout_line_buffering"] = getattr(sys.stdout, "line_buffering", None)
    info["stderr_line_buffering"] = getattr(sys.stderr, "line_buffering", None)

    return info


def log_terminal_debug_info(logger=None, prefix=""):
    """
    Log comprehensive terminal debug information.

    Args:
        logger: Logger instance to use. If None, uses the root logger.
        prefix: Optional prefix for the debug section (e.g., "BEFORE" or "AFTER")
    """
    if logger is None:
        logger = logging.getLogger()

    # Get terminal information
    terminal_info = get_terminal_info()

    section_title = f"=== {prefix + ' ' if prefix else ''}Terminal Debug Information ==="
    logger.info(section_title)

    logger.info("Terminal Information:")
    logger.info(f"  - stdout is TTY: {terminal_info.get('isatty_stdout')}")
    logger.info(f"  - stderr is TTY: {terminal_info.get('isatty_stderr')}")
    logger.info(f"  - stdin is TTY: {terminal_info.get('isatty_stdin')}")

    logger.info("Terminal Size:")
    if "shutil_columns" in terminal_info:
        logger.info(f"  - shutil size: {terminal_info['shutil_columns']}x{terminal_info['shutil_lines']}")
    if "os_columns" in terminal_info:
        logger.info(f"  - os size: {terminal_info['os_columns']}x{terminal_info['os_lines']}")
    if "stty_columns" in terminal_info:
        logger.info(f"  - stty size: {terminal_info['stty_columns']}x{terminal_info['stty_lines']}")

    logger.info("Environment Variables:")
    for var, value in terminal_info.get("env_vars", {}).items():
        logger.info(f"  - {var}: {value}")

    if "rich_width" in terminal_info:
        logger.info("Rich Console Settings:")
        logger.info(f"  - Width: {terminal_info['rich_width']}")
        logger.info(f"  - Height: {terminal_info['rich_height']}")
        logger.info(f"  - Is Terminal: {terminal_info['rich_is_terminal']}")
        logger.info(f"  - Color System: {terminal_info['rich_color_system']}")

    logger.info("Encoding and Buffering:")
    logger.info(f"  - stdout encoding: {terminal_info.get('stdout_encoding')}")
    logger.info(f"  - stderr encoding: {terminal_info.get('stderr_encoding')}")
    logger.info(f"  - stdout line buffering: {terminal_info.get('stdout_line_buffering')}")
    logger.info(f"  - stderr line buffering: {terminal_info.get('stderr_line_buffering')}")

    # Test log line width with a long line
    logger.info("Testing long log line (200 chars):")
    test_line = "=" * 200
    logger.info(test_line)

    # Test with a single line containing no newlines
    logger.info("Testing single line with text: " + "A" * 50 + " [MIDDLE] " + "B" * 50 + " [END]")

    logger.info(f"=== End {prefix + ' ' if prefix else ''}Terminal Debug ===")


def setup_mettagrid_logger(name: str, level=None) -> logging.Logger:
    # Get the appropriate log level based on priority
    log_level = get_log_level(level)

    # Remove all handlers from the root logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add back our custom Rich handler
    rich_handler = AlwaysShowTimeRichHandler(rich_tracebacks=True)
    formatter = MillisecondFormatter("%(message)s", datefmt="[%H:%M:%S.%f]")
    rich_handler.setFormatter(formatter)
    root_logger.addHandler(rich_handler)

    # Set the level
    root_logger.setLevel(getattr(logging, log_level))

    # Create logger for this module
    logger = logging.getLogger(name)

    # Log basic setup information
    logger.info(f"Logger initialized: name={name}, level={log_level}")

    # Log terminal state before forcing width
    log_terminal_debug_info(logger, prefix="BEFORE")

    # Force terminal width
    force_terminal_width(200)

    # Log terminal state after forcing width
    log_terminal_debug_info(logger, prefix="AFTER")

    return logger


# Additional utility function to force terminal width if needed
def force_terminal_width(width: int = 200):
    """
    Attempt to force terminal width through various methods.
    This is useful for AWS Batch environments.
    """
    logger = logging.getLogger(__name__)

    # Set environment variables
    os.environ["COLUMNS"] = str(width)
    os.environ["LINES"] = "50"

    # Try to use stty if available
    try:
        subprocess.run(["stty", "cols", str(width)], check=False)
        logger.info(f"Set terminal width to {width} using stty")
    except Exception:
        pass

    # Try to configure Rich console if available
    try:
        from rich.console import Console

        _console = Console(width=width, force_terminal=True)
        logger.info(f"Configured Rich console with width {width}")
    except Exception:
        pass
