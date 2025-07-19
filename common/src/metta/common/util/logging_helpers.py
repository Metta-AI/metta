import logging
import os
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
    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        created = datetime.fromtimestamp(record.created)
        # Convert microseconds to milliseconds (keep only 3 digits)
        msec = created.microsecond // 1000
        if datefmt:
            # If datefmt contains %f, replace it with milliseconds
            if "%f" in datefmt:
                # Replace %f with the millisecond value
                return created.strftime(datefmt.replace("%f", f"{msec:03d}"))
            else:
                return created.strftime(datefmt)
        else:
            # Default format with milliseconds
            return created.strftime(f"[%H:%M:%S.{msec:03d}]")


# Create a custom handler that always shows the timestamp
class AlwaysShowTimeRichHandler(RichHandler):
    def emit(self, record: logging.LogRecord) -> None:
        # Force a unique timestamp for each record
        record.created = record.created + (record.relativeCreated % 1000) / 1000000
        super().emit(record)


# Simple handler that formats logs without Rich when needed
class SimpleHandler(logging.StreamHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.formatter = MillisecondFormatter("%(message)s", datefmt="[%H:%M:%S.%f]")


def get_log_level(provided_level: str | None = None) -> str:
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


def setup_mettagrid_logger(name: str, level: str | None = None) -> logging.Logger:
    # Get the appropriate log level based on priority
    log_level = get_log_level(level)

    # Remove all handlers from the root logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Check if we're running with wandb or in a context where logs might be captured
    # This includes wandb runs, batch jobs, or when NO_HYPERLINKS is set
    use_simple_handler = (
        os.environ.get("WANDB_MODE") is not None  # wandb is configured
        or os.environ.get("WANDB_RUN_ID") is not None  # wandb run is active
        or os.environ.get("METTA_RUN_ID") is not None  # metta run that might use wandb
        or os.environ.get("AWS_BATCH_JOB_ID") is not None  # AWS batch job
        or os.environ.get("SKYPILOT_TASK_ID") is not None  # SkyPilot job
        or os.environ.get("NO_HYPERLINKS") is not None  # explicit disable
        or os.environ.get("NO_RICH_LOGS") is not None  # explicit disable rich
    )

    if use_simple_handler:
        # Use simple handler without Rich formatting
        handler = SimpleHandler(sys.stdout)
        # Apply custom formatting for timestamps
        formatter = MillisecondFormatter("%(message)s", datefmt="[%H:%M:%S.%f]")
        handler.setFormatter(formatter)

        # Prefix log level to message
        class LevelPrefixFormatter(MillisecondFormatter):
            def format(self, record):
                # Get the base formatted message
                msg = super().format(record)
                # Add level prefix with padding to align messages
                level_name = record.levelname
                timestamp = self.formatTime(record)

                # Add file location if available
                location = ""
                if hasattr(record, "pathname") and hasattr(record, "lineno"):
                    filename = os.path.basename(record.pathname)
                    location = f" [{filename}:{record.lineno}]"

                # Format: [timestamp] LEVEL message [file:line]
                return f"{timestamp} {level_name:<8} {msg}{location}"

        handler.setFormatter(LevelPrefixFormatter("%(message)s", datefmt="[%H:%M:%S.%f]"))
        root_logger.addHandler(handler)
    else:
        # Use Rich handler for interactive terminals
        rich_handler = AlwaysShowTimeRichHandler(
            rich_tracebacks=True,
            show_path=True,
            enable_link_path=True,  # Enable links in interactive mode
        )
        formatter = MillisecondFormatter("%(message)s", datefmt="[%H:%M:%S.%f]")
        rich_handler.setFormatter(formatter)
        root_logger.addHandler(rich_handler)

    # Set the level
    root_logger.setLevel(getattr(logging, log_level))

    # set env COLUMNS if we are in a batch job
    if os.environ.get("AWS_BATCH_JOB_ID") or os.environ.get("SKYPILOT_TASK_ID"):
        os.environ["COLUMNS"] = "200"

    return logging.getLogger(name)
