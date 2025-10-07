from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from mettagrid.map_builder.map_builder import GameMap, MapBuilder, MapBuilderConfig
from mettagrid.map_builder.utils import create_grid
from mettagrid.mapgen.types import MapGrid


class SparseObject(BaseModel):
    row: int
    column: int
    name: str

    @model_validator(mode="after")
    def validate_name(self) -> "SparseObject":
        if not self.name:
            raise ValueError("Object name must be non-empty")
        return self


class SparseMapBuilder(MapBuilder):
    class Config(MapBuilderConfig["SparseMapBuilder"]):
        width: int
        height: int
        objects: list[SparseObject] = Field(default_factory=list)

        @model_validator(mode="after")
        def validate_dimensions(self) -> "SparseMapBuilder.Config":
            if self.width <= 0 or self.height <= 0:
                raise ValueError("width and height must be positive")

            occupied: set[tuple[int, int]] = set()
            for obj in self.objects:
                if not 0 <= obj.row < self.height:
                    raise ValueError(f"Object {obj.name!r} has row {obj.row} outside grid height {self.height}")
                if not 0 <= obj.column < self.width:
                    raise ValueError(f"Object {obj.name!r} has column {obj.column} outside grid width {self.width}")
                coord = (obj.row, obj.column)
                if coord in occupied:
                    raise ValueError(f"Duplicate object placement at {coord}")
                occupied.add(coord)
            return self

        @classmethod
        def from_uri(cls, uri: str) -> "SparseMapBuilder.Config":
            return super().from_uri(uri)

    def __init__(self, config: Config):
        self.config = config
        self._grid = self._build_grid()

    def _build_grid(self) -> MapGrid:
        grid = create_grid(self.config.height, self.config.width, fill_value="empty")
        for obj in self.config.objects:
            grid[obj.row, obj.column] = obj.name
        return grid

    def build(self) -> GameMap:
        return GameMap(self._grid)
