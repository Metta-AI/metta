import yaml

from mettagrid.config.config import Config
from mettagrid.mapgen.scene import ChildrenAction, Scene, SceneConfig


class YamlSceneParams(Config):
    file: str


class YamlScene(Scene[YamlSceneParams]):
    def get_children(self) -> list[ChildrenAction]:
        with open(self.params.file, "r") as fh:
            cfg = yaml.safe_load(fh)
            scene = SceneConfig.model_validate(cfg)
        return [
            ChildrenAction(scene=scene, where="full"),
            *self.children_actions,
        ]

    def render(self):
        pass
