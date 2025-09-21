import numpy as np

from mettagrid.config.config import Config
from mettagrid.mapgen.scene import ChildrenAction, Scene
from mettagrid.mapgen.scenes.yaml import YamlScene


class RandomYamlSceneCandidate(Config):
    scene_file: str
    weight: float = 1


class RandomYamlSceneParams(Config):
    candidates: list[RandomYamlSceneCandidate]


class RandomYamlScene(Scene[RandomYamlSceneParams]):
    def get_children(self) -> list[ChildrenAction]:
        candidates = self.params.candidates
        weights = np.array([c.weight for c in candidates], dtype=np.float32)
        weights /= weights.sum()

        idx = self.rng.choice(len(candidates), p=weights)
        scene = YamlScene.factory(YamlScene.Params(file=candidates[idx].scene_file))

        return [
            ChildrenAction(scene=scene, where="full"),
            *self.children_actions,
        ]

    def render(self):
        pass
