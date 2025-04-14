import os
import sys

from loguru import logger

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
