from .multi_room import MultiRoom
from .random import Random
from .room import Room

# Try to import TerrainFromNumpy, but don't fail if dependencies are missing
try:
    from .terrain_from_numpy import TerrainFromNumpy

    __all__ = ["TerrainFromNumpy", "Random", "MultiRoom", "Room"]
except ImportError:
    # If TerrainFromNumpy can't be imported (e.g., missing boto3), just skip it
    __all__ = ["Random", "MultiRoom", "Room"]
