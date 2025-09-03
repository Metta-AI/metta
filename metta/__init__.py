# metta/__init__.py
__path__ = __import__("pkgutil").extend_path(__path__, __name__)

from metta.common.util.log_config import init_logging

init_logging()
