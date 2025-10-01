import numpy as np

from mettagrid.mapgen.scene import ChildrenAction, Config, Scene, SceneConfig
from mettagrid.mapgen.scenes.yaml import YamlScene


class RandomYamlSceneCandidate(Config):
    scene_file: str
    weight: float = 1


class RandomYamlSceneConfig(SceneConfig):
    candidates: list[RandomYamlSceneCandidate]


class RandomYamlScene(Scene[RandomYamlSceneConfig]):
    def get_children(self) -> list[ChildrenAction]:
        candidates = self.config.candidates
        weights = np.array([c.weight for c in candidates], dtype=np.float32)
        weights /= weights.sum()

        idx = self.rng.choice(len(candidates), p=weights)
        scene = YamlScene.Config(file=candidates[idx].scene_file)

        return [
            ChildrenAction(scene=scene, where="full"),
        ]

    def render(self):
        pass
