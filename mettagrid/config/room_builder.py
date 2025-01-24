
import numpy as np


SYMBOLS = {
    "agent": "A",
    "altar": "a",
    "converter": "c",
    "generator": "g",
    "wall": "W",
    "empty": " ",
}

class RoomBuilder():
    def __init__(self, border_width: int = 0, border_object: str = "wall"):
        self._border_width = border_width
        self._border_object = border_object

    def build(self):
        room = self._build()
        return self._add_border(room)

    def _add_border(self, room):
        b = self._border_width
        h, w = room.shape
        final_level = np.full((h + b * 2, w + b * 2), SYMBOLS[self._border_object], dtype="U6")
        final_level[b:b + h, b:b + w] = room
        return final_level

    def _build(self):
        raise NotImplementedError("Subclass must implement _build method")
