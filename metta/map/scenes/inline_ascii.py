import numpy as np

from metta.map.scene import Scene
from metta.util.config import Config

SYMBOLS = {
    "A": "agent.agent",
    "Ap": "agent.prey",
    "AP": "agent.predator",
    "a": "altar",
    "c": "converter",
    "g": "generator",
    "m": "mine",
    "W": "wall",
    " ": "empty",
    "b": "block",
    "L": "lasery",
    "Q": "agent.team_1",
    "E": "agent.team_2",
    "R": "agent.team_3",
    "T": "agent.team_4",
    "🧱": "wall",
    "⚙": "generator",
    "⛩": "altar",
    "🏭": "factory",
    "🔬": "lab",
    "🏰": "temple",
}


class InlineAsciiParams(Config):
    data: str
    row: int = 0
    column: int = 0


class InlineAscii(Scene):
    Params = InlineAsciiParams

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        params = self.params

        lines = params.data.strip().splitlines()
        self.ascii_grid = np.array([list(line) for line in lines], dtype="U6")
        self.ascii_grid = np.vectorize(SYMBOLS.get)(self.ascii_grid)

    def render(self):
        params = self.params
        if self.width < self.ascii_grid.shape[1] + params.column or self.height < self.ascii_grid.shape[0] + params.row:
            raise ValueError(
                f"Grid size {self.ascii_grid.shape} is too large for scene size {self.width}x{self.height} at "
                f"{params.column},{params.row}"
            )

        grid_height, grid_width = self.ascii_grid.shape
        self.grid[
            params.row : params.row + grid_height,
            params.column : params.column + grid_width,
        ] = self.ascii_grid
