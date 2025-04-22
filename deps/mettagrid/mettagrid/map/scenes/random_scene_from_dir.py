import os
from typing import Optional, TypedDict, cast

import numpy as np

from mettagrid.map.scene import Scene, TypedChild
from mettagrid.map.scenes.random_scene import RandomScene
from mettagrid.map.utils.random import MaybeSeed

RandomSceneCandidate = TypedDict("RandomSceneCandidate", {"scene": Scene, "weight": float})


class RandomSceneFromDir(Scene):
    def __init__(
        self,
        dir: str,  # relative to the scene root (currently configs/ directory because we use Hydra)
        seed: MaybeSeed = None,
        children: Optional[list[TypedChild]] = None,
    ):
        super().__init__(children or [])
        self._dir = dir

        files = os.listdir("configs/" + self._dir)
        files = [f.replace(".yaml", "") for f in files if f.endswith(".yaml")]
        if not files:
            raise ValueError(f"No files found in {self._dir}")
        self._files = files

        self._rng = np.random.default_rng(seed)

    def get_children(self, node) -> list[TypedChild]:
        candidates = [cast(RandomSceneCandidate, {"scene": self._dir + "/" + f, "weight": 1.0}) for f in self._files]
        return [
            {
                "scene": RandomScene(candidates),
                "where": "full",
            }
        ]

    def _render(self, _):
        pass
