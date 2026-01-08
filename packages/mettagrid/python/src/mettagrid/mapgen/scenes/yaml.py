import yaml

from mettagrid.mapgen.scene import ChildrenAction, Scene, SceneConfig


class YamlSceneConfig(SceneConfig):
    file: str


class YamlScene(Scene[YamlSceneConfig]):
    def get_children(self) -> list[ChildrenAction]:
        with open(self.config.file, "r") as fh:
            cfg = yaml.safe_load(fh)
            scene = SceneConfig.model_validate(cfg)
        return [
            ChildrenAction(scene=scene, where="full"),
        ]

    def render(self):
        pass
