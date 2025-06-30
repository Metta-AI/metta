# metta/__init__.py
"""Metta: A modular reinforcement learning library.

This module provides programmatic access to Metta components through the api module.
"""

__path__ = __import__("pkgutil").extend_path(__path__, __name__)

# Import the API module for programmatic access
from . import api
