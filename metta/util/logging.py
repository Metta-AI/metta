import logging
import os
import sys
from datetime import datetime

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
    logger.remove()  # Remove default handler
    logger.remove()  # Remove default handler
    # logger.add(
    #     sys.stdout, colorize=True,
    #     format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | "
    #            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    #            "<level>{message}</level>")


def restore_io():
    sys.stderr = sys.__stderr__


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

        return created.strftime(datefmt)


# Create a custom handler that always shows the timestamp
class AlwaysShowTimeRichHandler(RichHandler):
    def emit(self, record):
        # Force a unique timestamp for each record
        record.created = record.created + (record.relativeCreated % 1000) / 1000000
        super().emit(record)


# Configure rich colored logging
handler = AlwaysShowTimeRichHandler(rich_tracebacks=True)
formatter = MillisecondFormatter("%(message)s", datefmt="[%H:%M:%S.%f]")
handler.setFormatter(formatter)

logging.basicConfig(level="DEBUG", handlers=[handler])


# Create a function to reset logging to Rich after Hydra takes over
def rich_logger(name: str) -> logging.Logger:
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
    root_logger.setLevel(logging.DEBUG)

    return logging.getLogger(name)
