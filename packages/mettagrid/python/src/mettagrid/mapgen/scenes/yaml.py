import yaml

import mettagrid.mapgen.scene


class YamlSceneConfig(mettagrid.mapgen.scene.SceneConfig):
    file: str


class YamlScene(mettagrid.mapgen.scene.Scene[YamlSceneConfig]):
    def get_children(self) -> list[mettagrid.mapgen.scene.ChildrenAction]:
        with open(self.config.file, "r") as fh:
            cfg = yaml.safe_load(fh)
            scene = mettagrid.mapgen.scene.SceneConfig.model_validate(cfg)
        return [
            mettagrid.mapgen.scene.ChildrenAction(scene=scene, where="full"),
        ]

    def render(self):
        pass
