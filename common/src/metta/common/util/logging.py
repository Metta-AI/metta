import functools
import logging
import os
import sys
from datetime import datetime

import rich.traceback
from rich.logging import RichHandler

from metta.common.util.constants import RANK_ENV_VARS


def get_node_rank() -> str | None:
    for var in RANK_ENV_VARS:
        if rank := os.environ.get(var):
            return rank
    return None


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


@functools.cache
def _add_file_logging(run_dir: str) -> None:
    """Set up file logging in addition to stdout logging."""
    # Create logs directory
    logs_dir = os.path.join(run_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    log_file = "script.log" if (get_node_rank() in ("0", None)) else f"script_{get_node_rank()}.log"

    # Set up file handler for the root logger
    log_file = os.path.join(logs_dir, log_file)
    file_handler = logging.FileHandler(log_file, mode="a")

    # Use the same formatter as the existing console handler
    formatter = logging.Formatter(fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)  # Ensure file handler level is set

    # Add to root logger so all log messages go to file
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)

    # Force a flush to make sure the file is created properly
    file_handler.flush()


@functools.cache
def _init_console_logging() -> None:
    # Remove all handlers from the root logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Check if we're running with wandb or in a context where logs might be captured
    # This includes wandb runs, batch jobs, or when NO_HYPERLINKS is set
    use_simple_handler = any(
        os.environ.get(key) is not None
        for key in (
            "WANDB_MODE",
            "WANDB_RUN_ID",
            "METTA_RUN_ID",
            "AWS_BATCH_JOB_ID",
            "SKYPILOT_TASK_ID",
            "NO_HYPERLINKS",
            "NO_RICH_LOGS",
        )
    )

    rank = get_node_rank()
    local_rank = os.environ.get("LOCAL_RANK")
    rank_prefix = f"[{rank}-{local_rank}]" if rank and local_rank else f"[{rank}]" if rank else ""

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

                # Format: [timestamp] [rank] LEVEL message [file:line]
                return f"{timestamp} {rank_prefix}{level_name:<8} {msg}{location}"

        handler.setFormatter(LevelPrefixFormatter("%(message)s", datefmt="[%H:%M:%S.%f]"))
        root_logger.addHandler(handler)
    else:
        # Use Rich handler for interactive terminals
        rich_handler = AlwaysShowTimeRichHandler(
            rich_tracebacks=True,
            show_path=True,
            enable_link_path=True,
        )

        # Create a formatter that includes rank if needed
        class RankFormatter(MillisecondFormatter):
            def format(self, record):
                # Prepend rank to the message if available
                if rank_prefix:
                    original_msg = record.getMessage()
                    record.msg = f"{rank_prefix}{original_msg}"
                    result = super().format(record)
                    record.msg = original_msg
                    return result
                return super().format(record)

        formatter = RankFormatter("%(message)s", datefmt="[%H:%M:%S.%f]")
        rich_handler.setFormatter(formatter)
        root_logger.addHandler(rich_handler)

    # Set default level
    root_logger.setLevel(os.environ.get("LOG_LEVEL", "INFO").upper())

    # set env COLUMNS if we are in a batch job
    if os.environ.get("AWS_BATCH_JOB_ID") or os.environ.get("SKYPILOT_TASK_ID"):
        os.environ["COLUMNS"] = "200"

    rich.traceback.install(show_locals=False)
    logging.getLogger("httpx").setLevel(logging.WARNING)


# Safe to be called repeatedly, but if it is called with different run_dirs, it will add multiple file output handlers
def init_logging(*, name: str | None = None, run_dir: str | None = None) -> logging.Logger:
    _init_console_logging()
    if run_dir:
        _add_file_logging(run_dir)
    return logging.getLogger(name or "metta")


def log(message: str, level: int = logging.INFO, master_only: bool = False) -> None:
    logger = init_logging()
    if master_only and get_node_rank() not in ("0", None):
        return
    # stacklevel=2 so that the caller of `log`, not `log` itself, is identified in the log message
    logger.log(level, message, stacklevel=2)


# Initialize logging on module import with sensible defaults
init_logging()
