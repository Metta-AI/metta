from types import SimpleNamespace
import numpy as np

class GameObject:
    def __init__(self, symbol: str):
        self.symbol = symbol

OBJECTS = SimpleNamespace(
    Agent = GameObject("A"),
    Altar = GameObject("a"),
    Converter = GameObject("c"),
    Generator = GameObject("g"),
    Wall = GameObject("W"),
    Empty = GameObject(" "),

    named = lambda name: OBJECTS.__dict__[name.title()]
)

class Room():
    def __init__(self, border_width: int = 0, border_object: GameObject = OBJECTS.Wall):
        self._border_width = border_width
        self._border_object = border_object

    def build(self):
        room = self._build()
        return self._add_border(room)

    def _add_border(self, room):
        b = self._border_width
        h, w = room.shape
        final_level = np.full((h + b * 2, w + b * 2), self._border_object.symbol, dtype="U6")
        final_level[b:b + h, b:b + w] = room
        return final_level

    def _build(self):
        raise NotImplementedError("Subclass must implement _build method")
