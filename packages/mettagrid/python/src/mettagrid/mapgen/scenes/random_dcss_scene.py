from __future__ import annotations

import os
from glob import glob

from pydantic import model_validator

from mettagrid.mapgen.scene import ChildrenAction, Scene, SceneConfig
from mettagrid.mapgen.scenes.random_yaml_scene import RandomYamlScene, RandomYamlSceneCandidate


class RandomDcssSceneConfig(SceneConfig):
    wfc: bool
    dcss: bool

    @model_validator(mode="after")
    def validate_required_fields(self) -> RandomDcssSceneConfig:
        if not self.wfc and not self.dcss:
            raise ValueError("Either wfc or dcss must be true")
        return self


class RandomDcssScene(Scene[RandomDcssSceneConfig]):
    def get_children(self) -> list[ChildrenAction]:
        candidates: list[RandomYamlSceneCandidate] = []

        root_dir = os.path.dirname(__file__) + "/dcss"
        if self.config.wfc:
            for yaml_file in glob(f"{root_dir}/wfc/*.yaml"):
                candidates.append(RandomYamlSceneCandidate(scene_file=yaml_file))

        if self.config.dcss:
            for yaml_file in glob(f"{root_dir}/dcss/*.yaml"):
                candidates.append(RandomYamlSceneCandidate(scene_file=yaml_file))

        if not candidates:
            raise ValueError(f"No candidates found in dcss directory {root_dir}")

        scene = RandomYamlScene.Config(candidates=candidates)

        return [
            ChildrenAction(scene=scene, where="full"),
        ]

    def render(self):
        pass
