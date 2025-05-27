import numpy as np

from mettagrid.room.room import Room

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
    "🧱": "wall",
    "⚙": "generator",
    "⛩": "altar",
    "🏭": "factory",
    "🔬": "lab",
    "🏰": "temple",
}


class Ascii(Room):
    def __init__(self, uri: str, border_width: int = 0, border_object: str = "wall"):
        super().__init__(border_width=border_width, border_object=border_object)
        with open(uri, "r", encoding="utf-8") as f:
            ascii_map = f.read()
        lines = ascii_map.strip().splitlines()
        self._level = np.array([list(line) for line in lines], dtype="U6")
        self._level = np.vectorize(SYMBOLS.get)(self._level)
        self.set_size_labels(self._level.shape[1], self._level.shape[0])

    def _build(self):
        return self._level
