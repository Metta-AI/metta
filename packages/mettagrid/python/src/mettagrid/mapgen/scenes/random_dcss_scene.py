import glob
import os
import typing

import pydantic

import mettagrid.mapgen.scene
import mettagrid.mapgen.scenes.random_yaml_scene


class RandomDcssSceneConfig(mettagrid.mapgen.scene.SceneConfig):
    wfc: bool
    dcss: bool

    @pydantic.model_validator(mode="after")
    def validate_required_fields(self) -> typing.Self:
        if not self.wfc and not self.dcss:
            raise ValueError("Either wfc or dcss must be true")
        return self


class RandomDcssScene(mettagrid.mapgen.scene.Scene[RandomDcssSceneConfig]):
    def get_children(self) -> list[mettagrid.mapgen.scene.ChildrenAction]:
        candidates: list[mettagrid.mapgen.scenes.random_yaml_scene.RandomYamlSceneCandidate] = []

        root_dir = os.path.dirname(__file__) + "/dcss"
        if self.config.wfc:
            for yaml_file in glob.glob(f"{root_dir}/wfc/*.yaml"):
                candidates.append(
                    mettagrid.mapgen.scenes.random_yaml_scene.RandomYamlSceneCandidate(scene_file=yaml_file)
                )

        if self.config.dcss:
            for yaml_file in glob.glob(f"{root_dir}/dcss/*.yaml"):
                candidates.append(
                    mettagrid.mapgen.scenes.random_yaml_scene.RandomYamlSceneCandidate(scene_file=yaml_file)
                )

        if not candidates:
            raise ValueError(f"No candidates found in dcss directory {root_dir}")

        scene = mettagrid.mapgen.scenes.random_yaml_scene.RandomYamlScene.Config(candidates=candidates)

        return [
            mettagrid.mapgen.scene.ChildrenAction(scene=scene, where="full"),
        ]

    def render(self):
        pass
