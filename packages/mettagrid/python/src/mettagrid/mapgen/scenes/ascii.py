from mettagrid.map_builder.ascii import AsciiMapBuilder
from mettagrid.mapgen.scene import ChildrenAction, Scene, SceneConfig
from mettagrid.mapgen.scenes.inline_ascii import InlineAscii


class AsciiConfig(SceneConfig):
    uri: str


class Ascii(Scene[AsciiConfig]):
    def post_init(self):
        ascii_config = AsciiMapBuilder.Config.from_uri(self.config.uri)
        self.ascii_data = "\n".join("".join(line) for line in ascii_config.map_data)
        self.char_to_name_map = ascii_config.char_to_name_map

    def get_children(self):
        return [
            ChildrenAction(
                scene=InlineAscii.Config(data=self.ascii_data, char_to_name=self.char_to_name_map),
                where="full",
            ),
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
        ascii_config = AsciiMapBuilder.Config.from_uri(config.uri)
        return ascii_config.height, ascii_config.width
