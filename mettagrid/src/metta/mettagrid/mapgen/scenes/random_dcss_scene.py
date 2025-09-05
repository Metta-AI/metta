from __future__ import annotations

import os
from glob import glob

from pydantic import model_validator

from metta.mettagrid.config import Config
from metta.mettagrid.mapgen.scene import ChildrenAction, Scene
from metta.mettagrid.mapgen.scenes.random_yaml_scene import RandomYamlScene, RandomYamlSceneCandidate


class RandomDcssSceneParams(Config):
    wfc: bool
    dcss: bool

    @model_validator(mode="after")
    def validate_required_fields(self) -> RandomDcssSceneParams:
        if not self.wfc and not self.dcss:
            raise ValueError("Either wfc or dcss must be true")
        return self


class RandomDcssScene(Scene[RandomDcssSceneParams]):
    def get_children(self) -> list[ChildrenAction]:
        candidates: list[RandomYamlSceneCandidate] = []

        root_dir = os.path.dirname(__file__) + "/dcss"
        if self.params.wfc:
            for yaml_file in glob(f"{root_dir}/wfc/*.yaml"):
                candidates.append(RandomYamlSceneCandidate(scene_file=yaml_file))

        if self.params.dcss:
            for yaml_file in glob(f"{root_dir}/dcss/*.yaml"):
                candidates.append(RandomYamlSceneCandidate(scene_file=yaml_file))

        if not candidates:
            raise ValueError(f"No candidates found in dcss directory {root_dir}")

        scene = RandomYamlScene.factory(RandomYamlScene.Params(candidates=candidates))

        return [
            ChildrenAction(scene=scene, where="full"),
            *self.children_actions,
        ]

    def render(self):
        pass
