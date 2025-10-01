from mettagrid.mapgen.scene import ChildrenAction, Scene, SceneConfig
from mettagrid.mapgen.scenes.inline_ascii import InlineAscii
from mettagrid.mapgen.utils.ascii_grid import char_grid_to_lines


class AsciiConfig(SceneConfig):
    uri: str


class Ascii(Scene[AsciiConfig]):
    def post_init(self):
        with open(self.config.uri, "r", encoding="utf-8") as f:
            self.ascii_data = f.read()

    def get_children(self):
        # Delegate rendering to the inline ascii scene.
        return [
            ChildrenAction(
                scene=InlineAscii.Config(data=self.ascii_data),
                where="full",
            ),
            *self.config.children,
        ]

    def render(self):
        pass

    @classmethod
    def intrinsic_size(cls, config: AsciiConfig) -> tuple[int, int]:
        """
        We have to load the file twice, because this is a class method and we can't reuse the file descriptor.
        (See the documentation of `Scene.intrinsic_size` for more details.)

        But this is probably not a big deal, because the file is usually small.
        """
        config = cls.Config.model_validate(config)
        with open(config.uri, "r", encoding="utf-8") as f:
            data = f.read()
            _, width, height = char_grid_to_lines(data)
            return height, width
