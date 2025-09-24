import functools
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import rich.traceback
from rich.logging import RichHandler

from metta.common.util.constants import RANK_ENV_VARS

logger = logging.getLogger(__name__)


def get_node_rank() -> str | None:
    for var in RANK_ENV_VARS:
        if rank := os.environ.get(var):
            return rank
    return None


class RankAwareLogger(logging.Logger):
    """Logger with built-in rank-aware methods."""

    def __init__(self, name, level=logging.NOTSET):
        super().__init__(name, level)
        self._is_master = None

    @property
    def is_master(self):
        if self._is_master is None:
            self._is_master = (get_node_rank() or "0") == "0"
        return self._is_master

    def debug_master(self, msg, *args, **kwargs):
        """Log debug message only on master (rank 0)."""
        if self.is_master:
            self.debug(msg, *args, **kwargs)

    def info_master(self, msg, *args, **kwargs):
        """Log info message only on master (rank 0)."""
        if self.is_master:
            self.info(msg, *args, **kwargs)

    def warning_master(self, msg, *args, **kwargs):
        """Log warning message only on master (rank 0)."""
        if self.is_master:
            self.warning(msg, *args, **kwargs)

    def error_master(self, msg, *args, **kwargs):
        """Log error message only on master (rank 0)."""
        if self.is_master:
            self.error(msg, *args, **kwargs)

    def critical_master(self, msg, *args, **kwargs):
        """Log critical message only on master (rank 0)."""
        if self.is_master:
            self.critical(msg, *args, **kwargs)


def getRankAwareLogger(name: str | None = None) -> RankAwareLogger:
    init_logging()
    return logging.getLogger(name)  # type: ignore[return-value]


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


class AlwaysShowTimeRichHandler(RichHandler):
    def emit(self, record: logging.LogRecord) -> None:
        # Force a unique timestamp for each record
        record.created = record.created + (record.relativeCreated % 1000) / 1000000
        super().emit(record)


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
    # Set the logger class to our rank-aware version
    # This must be done before any loggers are created
    logging.setLoggerClass(RankAwareLogger)

    # Remove all handlers from the root logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Check if we're running with wandb or in a context where logs might be captured
    # This includes wandb runs, batch jobs, or when NO_HYPERLINKS is set
    use_simple_handler = any(
        os.environ.get(key) is not None
        for key in (
            "WANDB_MODE",  # wandb is configured
            "WANDB_RUN_ID",  # wandb run is active
            "METTA_RUN_ID",  # metta run that might use wandb
            "AWS_BATCH_JOB_ID",  # AWS batch job
            "SKYPILOT_TASK_ID",  # SkyPilot job
            "NO_HYPERLINKS",  # explicit disable
            "NO_RICH_LOGS",  # explicit disable rich
        )
    )

    rank = get_node_rank()
    local_rank = os.environ.get("LOCAL_RANK")
    rank_prefix = f"[{rank}-{local_rank}] " if rank and local_rank else f"[{rank}] " if rank else ""

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
                return f"{timestamp} {rank_prefix} {level_name:<8} {msg}{location}"

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
                # Get the formatted message first, then prepend rank
                formatted_msg = super().format(record)
                if rank_prefix:
                    return f"{rank_prefix}{formatted_msg}"
                return formatted_msg

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
def init_logging(run_dir: Path | None = None) -> None:
    _init_console_logging()
    if run_dir:
        _add_file_logging(run_dir)

    # Do not log anything from here as it will interfere with scripts that return data on cli
    # e.g. calling constants.py will print a log statement and we won't be able to parse the expected value
