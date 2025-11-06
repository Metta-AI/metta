import mettagrid.map_builder.ascii
import mettagrid.mapgen.scene
import mettagrid.mapgen.scenes.inline_ascii


class AsciiConfig(mettagrid.mapgen.scene.SceneConfig):
    uri: str


class Ascii(mettagrid.mapgen.scene.Scene[AsciiConfig]):
    def post_init(self):
        ascii_config = mettagrid.map_builder.ascii.AsciiMapBuilder.Config.from_uri(self.config.uri)
        self.ascii_data = "\n".join("".join(line) for line in ascii_config.map_data)
        self.char_to_name_map = ascii_config.char_to_name_map

    def get_children(self):
        return [
            mettagrid.mapgen.scene.ChildrenAction(
                scene=mettagrid.mapgen.scenes.inline_ascii.InlineAscii.Config(
                    data=self.ascii_data, char_to_name=self.char_to_name_map
                ),
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
        ascii_config = mettagrid.map_builder.ascii.AsciiMapBuilder.Config.from_uri(config.uri)
        return ascii_config.height, ascii_config.width
