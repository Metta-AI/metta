import yaml

from mettagrid.mapgen.scene import ChildrenAction, Scene, SceneConfig, validate_any_scene_config


class YamlSceneConfig(SceneConfig):
    file: str


class YamlScene(Scene[YamlSceneConfig]):
    def get_children(self) -> list[ChildrenAction]:
        with open(self.config.file, "r") as fh:
            cfg = yaml.safe_load(fh)
            scene = validate_any_scene_config(cfg)
        return [
            ChildrenAction(scene=scene, where="full"),
        ]

    def render(self):
        pass
