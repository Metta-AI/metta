import atexit
import logging
import os
import sys
import tempfile
from datetime import datetime
from threading import Event, Thread

import boto3
from loguru import logger
from rich.logging import RichHandler


def remap_io(logs_path: str):
    os.makedirs(logs_path, exist_ok=True)
    stdout_log_path = os.path.join(logs_path, "out.log")
    stderr_log_path = os.path.join(logs_path, "error.log")
    stdout = open(stdout_log_path, "a")
    stderr = open(stderr_log_path, "a")
    sys.stderr = stderr
    sys.stdout = stdout
    logger.remove()


def restore_io():
    sys.stderr = sys.__stderr__
    sys.stdout = sys.__stdout__


def _upload_log_to_s3(log_path: str, s3_uri: str) -> None:
    """Upload ``log_path`` to ``s3_uri`` using boto3."""
    if not s3_uri.startswith("s3://"):
        return
    bucket, key = s3_uri[5:].split("/", 1)
    try:
        boto3.client("s3").upload_file(log_path, bucket, key, ExtraArgs={"ContentType": "text/plain"})
    except Exception as e:  # pragma: no cover - upload best effort
        logging.getLogger(__name__).error("Failed to upload logs to %s: %s", s3_uri, e)


def _start_s3_sync(log_path: str, s3_uri: str, *, interval: int = 30) -> tuple[Event, Thread]:
    """Start background uploader that periodically syncs the log to S3."""
    stop_event = Event()

    def _worker() -> None:  # pragma: no cover - thread
        _upload_log_to_s3(log_path, s3_uri)
        while not stop_event.wait(interval):
            _upload_log_to_s3(log_path, s3_uri)

    thread = Thread(target=_worker, daemon=True)
    thread.start()
    return stop_event, thread


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


def setup_mettagrid_logger(
    name: str,
    level: str | None = None,
    *,
    log_file: str | None = None,
    s3_uri: str | None = None,
) -> logging.Logger:
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

    if log_file or s3_uri:
        if not log_file:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".log")
            log_file = tmp.name
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        if s3_uri:
            stop_event, thread = _start_s3_sync(log_file, s3_uri)

            def _cleanup() -> None:
                stop_event.set()
                thread.join(timeout=5)
                _upload_log_to_s3(log_file, s3_uri)

            atexit.register(_cleanup)

    # Set the level
    root_logger.setLevel(getattr(logging, log_level))

    return logging.getLogger(name)
