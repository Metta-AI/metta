from mettagrid.map.scenes.inline_ascii import InlineAscii


class Ascii(InlineAscii):
    def __init__(self, uri: str):
        with open(uri, "r", encoding="utf-8") as f:
            data = f.read()
        super().__init__(data)
