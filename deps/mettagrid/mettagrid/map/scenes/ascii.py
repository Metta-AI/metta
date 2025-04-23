from mettagrid.map.scenes.inline_ascii import InlineAscii

# Note that these symbols don't match the encoding used by C env (see constants.hpp for those)
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
}


class Ascii(InlineAscii):
    def __init__(self, uri: str):
        with open(uri, "r", encoding="utf-8") as f:
            data = f.read()
        super().__init__(data)
