import yaml

from metta.mettagrid.config import Config
from metta.mettagrid.mapgen.scene import ChildrenAction, Scene, SceneConfig


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
