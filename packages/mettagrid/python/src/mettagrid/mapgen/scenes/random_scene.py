import numpy as np
from pydantic import field_validator

from mettagrid.base_config import Config
from mettagrid.mapgen.scene import ChildrenAction, Scene, SceneConfig, validate_any_scene_config


class RandomSceneCandidate(Config):
    scene: SceneConfig
    weight: float = 1

    @field_validator("scene", mode="wrap")
    @classmethod
    def _validate_scene(cls, v, handler):
        # Accept already-validated SceneConfig or a dict with a 'type' pointer
        if isinstance(v, SceneConfig):
            return v
        return validate_any_scene_config(v)


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
