from metta.common.util.config import Config
from metta.map.scene import Scene
from metta.map.scenes.inline_ascii import InlineAscii
from metta.map.types import ChildrenAction


class AsciiParams(Config):
    uri: str


class Ascii(Scene[AsciiParams]):
    def post_init(self):
        with open(self.params.uri, "r", encoding="utf-8") as f:
            self.ascii_data = f.read()

    def get_children(self):
        return [
            ChildrenAction(
                scene=lambda grid: InlineAscii(grid=grid, params={"data": self.ascii_data}),
                where="full",
            ),
            *self.children,
        ]

    def render(self):
        pass
