from typing import Optional, TypedDict

import numpy as np

from mettagrid.map.scene import Scene, TypedChild
from mettagrid.map.utils.random import MaybeSeed

RandomSceneCandidate = TypedDict("RandomSceneCandidate", {"scene": Scene, "weight": float})


class RandomScene(Scene):
    def __init__(
        self,
        candidates: list[RandomSceneCandidate],
        seed: MaybeSeed = None,
        children: Optional[list[TypedChild]] = None,
    ):
        super().__init__(children or [])
        self._candidates = candidates
        self._rng = np.random.default_rng(seed)

    def get_children(self, node) -> list[TypedChild]:
        weights = np.array([c.get("weight", 1) for c in self._candidates], dtype=np.float32)
        weights /= weights.sum()

        idx = self._rng.choice(len(self._candidates), p=weights)
        scene = self._candidates[idx]["scene"]

        return [{"scene": scene, "where": "full"}]

    def _render(self, _):
        pass
