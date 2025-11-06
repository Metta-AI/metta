import numpy as np

import mettagrid.mapgen.scene
import mettagrid.mapgen.scenes.yaml


class RandomYamlSceneCandidate(mettagrid.mapgen.scene.Config):
    scene_file: str
    weight: float = 1


class RandomYamlSceneConfig(mettagrid.mapgen.scene.SceneConfig):
    candidates: list[RandomYamlSceneCandidate]


class RandomYamlScene(mettagrid.mapgen.scene.Scene[RandomYamlSceneConfig]):
    def get_children(self) -> list[mettagrid.mapgen.scene.ChildrenAction]:
        candidates = self.config.candidates
        weights = np.array([c.weight for c in candidates], dtype=np.float32)
        weights /= weights.sum()

        idx = self.rng.choice(len(candidates), p=weights)
        scene = mettagrid.mapgen.scenes.yaml.YamlScene.Config(file=candidates[idx].scene_file)

        return [
            mettagrid.mapgen.scene.ChildrenAction(scene=scene, where="full"),
        ]

    def render(self):
        pass
