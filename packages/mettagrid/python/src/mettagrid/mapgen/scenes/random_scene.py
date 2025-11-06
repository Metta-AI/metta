import numpy as np

import mettagrid.base_config
import mettagrid.mapgen.scene


class RandomSceneCandidate(mettagrid.base_config.Config):
    scene: mettagrid.mapgen.scene.AnySceneConfig
    weight: float = 1


class RandomSceneConfig(mettagrid.mapgen.scene.SceneConfig):
    candidates: list[RandomSceneCandidate]


class RandomScene(mettagrid.mapgen.scene.Scene[RandomSceneConfig]):
    def get_children(self) -> list[mettagrid.mapgen.scene.ChildrenAction]:
        candidates = self.config.candidates
        weights = np.array([c.weight for c in candidates], dtype=np.float32)
        weights /= weights.sum()

        idx = self.rng.choice(len(candidates), p=weights)
        scene = candidates[idx].scene

        return [
            mettagrid.mapgen.scene.ChildrenAction(scene=scene, where="full"),
        ]

    def render(self):
        pass
