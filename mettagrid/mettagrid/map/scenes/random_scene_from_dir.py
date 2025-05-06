from pathlib import Path
from typing import Optional, TypedDict, cast

import numpy as np

from mettagrid.config.utils import scenes_root
from mettagrid.map.scene import Scene, TypedChild
from mettagrid.map.scenes.random_scene import RandomScene
from mettagrid.map.utils.random import MaybeSeed

RandomSceneCandidate = TypedDict("RandomSceneCandidate", {"scene": Scene, "weight": float})


class RandomSceneFromDir(Scene):
    def __init__(
        self,
        dir: str,
        seed: MaybeSeed = None,
        children: Optional[list[TypedChild]] = None,
    ):
        super().__init__(children or [])
        self._dir = Path(dir).resolve()

        if not self._dir.exists():
            raise ValueError(f"Directory {self._dir} does not exist")

        self._scenes = []
        for file in self._dir.iterdir():
            self._scenes.append("/" + str(file.relative_to(scenes_root)))

        if not self._scenes:
            raise ValueError(f"No files found in {self._dir}")

        self._rng = np.random.default_rng(seed)

    def get_children(self, node) -> list[TypedChild]:
        candidates = [cast(RandomSceneCandidate, {"scene": scene, "weight": 1.0}) for scene in self._scenes]
        return [
            {
                "scene": RandomScene(candidates),
                "where": "full",
            }
        ]

    def _render(self, _):
        pass
