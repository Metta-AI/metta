import numpy as np

from metta.map.scene import Scene

SYMBOLS = {
    "A": "agent.agent",
    "Ap": "agent.prey",
    "AP": "agent.predator",
    "a": "altar",
    "c": "converter",
    "n": "generator",
    "m": "mine",
    "W": "wall",
    " ": "empty",
    "s": "block",
    "L": "lasery",
    "1": "agent.team_1",
    "2": "agent.team_2",
    "3": "agent.team_3",
    "4": "agent.team_4",
    "r": "mine.red",
    "b": "mine.blue",
    "g": "mine.green",
    "R": "generator.red",
    "B": "generator.blue",
    "G": "generator.green",
}


class InlineAscii(Scene):
    def __init__(self, data: str, row: int = 0, column: int = 0):
        super().__init__()
        lines = data.strip().splitlines()
        self._level = np.array([list(line) for line in lines], dtype="U6")
        self._level = np.vectorize(SYMBOLS.get)(self._level)
        self._row = row
        self._column = column

    def _render(self, node):
        if node.width < self._level.shape[1] + self._column or node.height < self._level.shape[0] + self._row:
            raise ValueError(
                f"Level size {self._level.shape} is too large for node size {node.width}x{node.height} at "
                f"{self._column},{self._row}"
            )

        level_height, level_width = self._level.shape
        node.grid[
            self._row: self._row + level_height,
            self._column: self._column + level_width,
        ] = self._level
