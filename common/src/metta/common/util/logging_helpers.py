import logging
import os
import sys
from datetime import datetime

import rich.traceback
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


def get_node_rank() -> str | None:
    """Get node/rank index from various distributed environments."""
    return (
        os.environ.get("SKYPILOT_NODE_RANK") or
        os.environ.get("RANK") or  # PyTorch DDP
        os.environ.get("OMPI_COMM_WORLD_RANK") or  # OpenMPI
        None
    )


def log_master(message: str, logger: logging.Logger | None = None, level: int = logging.INFO):
    """
    Log message only on master node (rank 0).

    Args:
        message: Message to log
        logger: Logger to use (default: root logger)
        level: Log level (default: INFO)
    """
    rank = get_node_rank() or "0"
    if rank == "0":
        if logger is None:
            logger = logging.getLogger()
        logger.log(level, message)


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


def init_file_logging(run_dir: str) -> None:
    """Set up file logging in addition to stdout logging."""
    # Create logs directory
    logs_dir = os.path.join(run_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    # Use node rank if available, otherwise RANK env var
    node_index = get_node_rank() or "0"
    if node_index == "0":
        log_file = "script.log"
    else:
        log_file = f"script_{node_index}.log"

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


def init_logging(level: str | None = None, run_dir: str | None = None, show_rank: bool = False) -> None:
    """
    Initialize logging with optional node/rank display.

    Args:
        level: Log level string
        run_dir: Directory for log files (optional)
        show_rank: Whether to show node/rank in log messages (default: False)
                  If True and rank is detected, prepends [rank] to messages
    """
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

    # Get node rank if needed
    rank = get_node_rank() if show_rank else None
    rank_prefix = f"[{rank}] " if rank else ""

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
            enable_link_path=True,  # Enable links in interactive mode
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

    # Set the level
    root_logger.setLevel(getattr(logging, log_level))

    # set env COLUMNS if we are in a batch job
    if os.environ.get("AWS_BATCH_JOB_ID") or os.environ.get("SKYPILOT_TASK_ID"):
        os.environ["COLUMNS"] = "200"

    rich.traceback.install(show_locals=False)

    # Add file logging (after console handlers are set up)
    if run_dir:
        init_file_logging(run_dir)
