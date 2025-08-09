"""
A/B Testing Framework for Metta

This module provides a framework for defining and running A/B comparison experiments
using Python classes instead of YAML configuration files.
"""

from .config import ABExperiment, ABTestConfig, ABVariant
from .runner import ABTestRunner

__all__ = ["ABTestConfig", "ABVariant", "ABExperiment", "ABTestRunner"]
