import numpy as np

from metta.map.node import Node
from metta.map.scene import SceneCfg, TypedChild
from metta.util.config import Config


class RandomSceneCandidate(Config):
    scene: SceneCfg
    weight: float = 1


class RandomSceneParams(Config):
    candidates: list[RandomSceneCandidate]


class RandomScene(Node[RandomSceneParams]):
    params_type = RandomSceneParams

    def get_children(self) -> list[TypedChild]:
        candidates = self.params.candidates
        weights = np.array([c.weight for c in candidates], dtype=np.float32)
        weights /= weights.sum()

        idx = self.rng.choice(len(candidates), p=weights)
        scene = candidates[idx].scene

        return [
            {"scene": scene, "where": "full"},
            *self.children,
        ]

    def render(self):
        pass
