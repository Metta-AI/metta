# metta/__init__.py
__path__ = __import__("pkgutil").extend_path(__path__, __name__)

import logging

from metta.common.util.log_config import init_logging

# Initialize logging (idempotent due to @functools.cache)
init_logging()

# Add NullHandler for this package
logging.getLogger(__name__).addHandler(logging.NullHandler())
