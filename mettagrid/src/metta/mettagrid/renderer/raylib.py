"""Raylib-based graphical renderer for MettaGridEnv."""

import math
from typing import Dict, Optional

try:
    from raylib import (
        FLAG_MSAA_4X_HINT,
        PI,
        WHITE,
        BeginDrawing,
        ClearBackground,
        DrawTexturePro,
        EndDrawing,
        InitWindow,
        LoadTexture,
        SetConfigFlags,
        SetTargetFPS,
    )

    RAYLIB_AVAILABLE = True
except ImportError:
    RAYLIB_AVAILABLE = False


class RaylibRenderer:
    """Raylib-based graphical renderer for MettaGridEnv."""

    def __init__(self, object_type_names: list[str], map_width: int, map_height: int):
        if not RAYLIB_AVAILABLE:
            raise ImportError("Raylib is required for RaylibRenderer but not installed")

        self._object_type_names = object_type_names
        self._map_width = map_width
        self._map_height = map_height
        self._initialized = False
        self.texture = None
        self.tiles = {}

    def _initialize(self):
        """Initialize raylib window and load textures."""
        SetConfigFlags(FLAG_MSAA_4X_HINT)
        InitWindow(16 * self._map_width, 16 * self._map_height, b"Mettagrid")
        self.texture = LoadTexture(b"resources/shared/puffers.png")
        SetTargetFPS(60)

        self.tiles = {}
        for id, name in enumerate(self._object_type_names):
            name = f"/puffertank/metta/mettascope/data/atlas/objects/{name}.png"
            self.tiles[id] = LoadTexture(name.encode("utf-8"))

        self._initialized = True

    def render(self, step: int, grid_objects: Dict[int, dict]) -> Optional[str]:
        """Render the environment using raylib."""
        if not self._initialized:
            self._initialize()

        BeginDrawing()
        background = [207, 169, 112, 255]
        ClearBackground(background)
        sz = 16

        for obj in grid_objects.values():
            type = obj["type"]
            tex = self.tiles[type]

            tint = WHITE
            if self._object_type_names[type] == "agent":
                id = obj["id"]
                tint = [
                    int(255 * ((id * PI) % 1.0)),
                    int(255 * ((id * math.e) % 1.0)),
                    int(255 * ((id * 2.0**0.5) % 1.0)),
                    255,
                ]

            y = obj["r"]
            x = obj["c"]
            size = sz * (256.0 / 200.0)
            DrawTexturePro(
                tex,
                [0, 0, tex.width, tex.height],
                [x * sz, y * sz, size, size],
                [0, 0],
                0,
                tint,
            )

        EndDrawing()
        # Raylib renders to window, so return None
        return None
