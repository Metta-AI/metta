import functools
import json
import logging
import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import rich.traceback
from pydantic.warnings import UnsupportedFieldAttributeWarning
from rich.console import Console
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


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        for key in ("trace_id", "span_id", "service", "env", "version"):
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value

        return json.dumps(payload, ensure_ascii=True)


def _json_logging_mode() -> str:
    if os.environ.get("LOG_FORMAT", "").lower() == "json":
        return "only"
    if os.environ.get("LOG_JSON", "").lower() in ("1", "true", "yes"):
        return "dual"
    return "none"


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

    json_mode = _json_logging_mode()
    if json_mode == "only":
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        root_logger.addHandler(handler)
    elif use_simple_handler:
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

    if json_mode == "dual":
        json_handler = logging.StreamHandler(sys.stdout)
        json_handler.setFormatter(JsonFormatter())
        root_logger.addHandler(json_handler)

    if json_mode != "only":
        rich.traceback.install(show_locals=False)


# Safe to be called repeatedly, but if it is called with different run_dirs, it will add multiple file output handlers
def init_logging(run_dir: Path | None = None) -> None:
    _init_console_logging()
    if run_dir:
        _add_file_logging(run_dir)

    # Do not log anything from here as it will interfere with scripts that return data on cli
    # e.g. calling constants.py will print a log statement and we won't be able to parse the expected value


@functools.cache
def get_console() -> Console:
    # Good practice to use a global console instance by default
    return Console()


def should_use_rich_console() -> bool:
    """Determine if rich console output is appropriate based on terminal context."""
    if os.environ.get("DISABLE_RICH_LOGGING", "").lower() in ("1", "true", "yes"):
        return False

    if any(os.environ.get(var) for var in ["SLURM_JOB_ID", "PBS_JOBID", "WANDB_RUN_ID", "SKYPILOT_TASK_ID"]):
        return False

    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def suppress_noisy_logs() -> None:
    warnings.filterwarnings("ignore", category=UnsupportedFieldAttributeWarning, module="pydantic")
    # Suppress deprecation warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="pygame.pkgdata")

    # Silence PyTorch distributed elastic warning about redirects on MacOS/Windows
    logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
    warnings.filterwarnings(
        "ignore",
        message=r".*Redirects are currently not supported in Windows or MacOs.*",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # Silence ddtrace CI Visibility spam
    logging.getLogger("ddtrace").setLevel(logging.WARNING)


def init_mettagrid_system_environment() -> None:
    """Initialize environment variables for headless operation."""
    os.environ.setdefault("GLFW_PLATFORM", "osmesa")  # Use OSMesa as the GLFW backend
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    os.environ.setdefault("DISPLAY", "")
