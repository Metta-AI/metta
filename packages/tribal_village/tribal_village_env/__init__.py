"""
Tribal Village Environment - Minimal PufferLib wrapper

This package provides a thin wrapper for the Nim-based tribal village environment
to integrate with PufferLib. The core game logic is implemented in Nim.
"""

from .environment import TribalVillageEnv as TribalVillageEnv
from .environment import make_tribal_village_env as make_tribal_village_env

__all__ = [
    "TribalVillageEnv",
    "make_tribal_village_env",
]

__version__ = "0.1.0"
