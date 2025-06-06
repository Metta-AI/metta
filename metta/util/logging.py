import logging
import os
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


# Additional utility function to force terminal width if needed
def force_terminal_width(width: int):
    """
    Attempt to force terminal width through various methods.
    This is useful for AWS Batch environments.
    """
    logger = logging.getLogger(__name__)

    # Set environment variables
    os.environ["COLUMNS"] = str(width)

    # Try to use stty if available
    try:
        subprocess.run(["stty", "cols", str(width)], check=False)
        logger.info(f"Set terminal width to {width} using stty")
    except Exception:
        pass

    # # Try to configure Rich console if available
    # try:
    #     from rich.console import Console
    #     _console = Console(width=width, force_terminal=True)
    #     logger.info(f"Configured Rich console with width {width}")
    # except Exception:
    #     pass


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

    # Force terminal width
    # lines are padded with about 45 chars of metadata (timestamp, log level, file and line info)
    force_terminal_width(300)

    return logger
