# metta/__init__.py
__path__ = __import__("pkgutil").extend_path(__path__, __name__)

from metta.common.util.log_config import init_logging

# Initialize logging (idempotent due to @functools.cache)
# This ensures the root logger has handlers configured, preventing
# "No handler" warnings for all loggers in the application
init_logging()
