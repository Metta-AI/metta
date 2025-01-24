import numpy as np

from mettagrid.config.room_builder import RoomBuilder

class AsciiRoomBuilder(RoomBuilder):
    def __init__(self, uri: str, border_width: int = 0, border_object: str = "wall"):
        super().__init__(border_width=border_width, border_object=border_object)
        with open(uri, "r", encoding="utf-8") as f:
            ascii_map = f.read()
        lines = ascii_map.strip().splitlines()
        self._level = np.array([list(line) for line in lines], dtype="U6")

    def _build(self):
        return self._level
