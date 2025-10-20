import numpy as np

from mettagrid.base_config import Config
from mettagrid.mapgen.scene import ChildrenAction, Scene, SceneConfig


class RandomSceneCandidate(Config):
    scene: SceneConfig
    weight: float = 1


class RandomSceneConfig(SceneConfig):
    candidates: list[RandomSceneCandidate]


class RandomScene(Scene[RandomSceneConfig]):
    def get_children(self) -> list[ChildrenAction]:
        candidates = self.config.candidates
        weights = np.array([c.weight for c in candidates], dtype=np.float32)
        weights /= weights.sum()

        idx = self.rng.choice(len(candidates), p=weights)
        scene = candidates[idx].scene

        return [
            ChildrenAction(scene=scene, where="full"),
        ]

    def render(self):
        pass
