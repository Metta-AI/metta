from mettagrid.config.config import Config
from mettagrid.mapgen.scene import ChildrenAction, Scene
from mettagrid.mapgen.scenes.inline_ascii import InlineAscii
from mettagrid.mapgen.utils.ascii_grid import char_grid_to_lines


class AsciiParams(Config):
    uri: str


class Ascii(Scene[AsciiParams]):
    def post_init(self):
        with open(self.params.uri, "r", encoding="utf-8") as f:
            self.ascii_data = f.read()

    def get_children(self):
        # Delegate rendering to the inline ascii scene.
        return [
            ChildrenAction(
                scene=InlineAscii.factory(InlineAscii.Params(data=self.ascii_data)),
                where="full",
            ),
            *self.children_actions,
        ]

    def render(self):
        pass

    @classmethod
    def intrinsic_size(cls, params: AsciiParams) -> tuple[int, int]:
        """
        We have to load the file twice, because this is a class method and we can't reuse the file descriptor.
        (See the documentation of `Scene.intrinsic_size` for more details.)

        But this is probably not a big deal, because the file is usually small.
        """
        params = cls.validate_params(params)
        with open(params.uri, "r", encoding="utf-8") as f:
            data = f.read()
            _, width, height = char_grid_to_lines(data)
            return height, width
