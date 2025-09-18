"""Pytest plugin to suppress Gym warnings during worker startup."""

import warnings

# Suppress Gym warnings immediately when this module is imported
warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*", category=UserWarning)


def pytest_configure(config):
    """Ensure warnings are suppressed during pytest configuration."""
    warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*", category=UserWarning)
