import logging
import os
import shutil
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


def debug_logging_state(logger: logging.Logger) -> None:
    """Log detailed debugging information about the logging environment."""
    logger.info("LOGGING DEBUG INFORMATION")

    # Environment variables
    logger.info("Environment Variables:")
    for key, value in sorted(os.environ.items()):
        # Mask sensitive values
        if any(sensitive in key.upper() for sensitive in ["PASSWORD", "SECRET", "TOKEN", "KEY"]):
            logger.info(f"{key}=<MASKED>")
        else:
            logger.info(f"{key}={value}")

    # Terminal information
    logger.info("Terminal Information:")
    if hasattr(sys.stdin, "fileno"):
        logger.info(f"stdin isatty: {os.isatty(sys.stdin.fileno())}")
    if hasattr(sys.stdout, "fileno"):
        logger.info(f"stdout isatty: {os.isatty(sys.stdout.fileno())}")
    if hasattr(sys.stderr, "fileno"):
        logger.info(f"stderr isatty: {os.isatty(sys.stderr.fileno())}")

    # Terminal size
    try:
        size = shutil.get_terminal_size()
        logger.info(f"Terminal size: {size.columns}x{size.lines}")
    except Exception as e:
        logger.info(f"Terminal size: error - {e}")

    logger.info(f"COLUMNS env: {os.environ.get('COLUMNS', 'Not set')}")
    logger.info(f"LINES env: {os.environ.get('LINES', 'Not set')}")

    # AWS environment variables
    logger.info("AWS Environment Variables:")
    aws_vars = {k: v for k, v in os.environ.items() if k.startswith("AWS_")}
    if aws_vars:
        for key, value in sorted(aws_vars.items()):
            logger.info(f"{key}={value}")
    else:
        logger.info("No AWS environment variables found")

    # System information
    logger.info("System Information:")
    logger.info(f"Python: {sys.version.split()[0]}")
    logger.info(f"Platform: {sys.platform}")
    logger.info(f"Executable: {sys.executable}")

    # Logging configuration
    logger.info("Logging Configuration:")
    logger.info(f"Logger name: {logger.name}")
    logger.info(f"Logger level: {logging.getLevelName(logger.level)}")
    root_level = logging.getLogger().level
    logger.info(f"Root logger level: {logging.getLevelName(root_level)}")
    logger.info(f"Handler count: {len(logger.handlers)}")


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

    # Get the logger for the specified name
    logger = logging.getLogger(name)

    # Log debugging information
    debug_logging_state(logger)

    # Set env COLUMNS if we are on AWS
    if any(key.startswith("AWS_") for key in os.environ):
        if "COLUMNS" not in os.environ:
            os.environ["COLUMNS"] = "200"
            logger.info("Set COLUMNS=200 (AWS environment)")

    return logger
