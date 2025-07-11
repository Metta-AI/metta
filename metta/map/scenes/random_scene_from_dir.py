from pathlib import Path
from typing import cast

from metta.common.util.config import Config
from metta.map.config import scenes_root
from metta.map.scene import Scene
from metta.map.scenes.random_scene import RandomScene, RandomSceneCandidate
from metta.map.types import ChildrenAction


class RandomSceneFromDirParams(Config):
    dir: str


class RandomSceneFromDir(Scene[RandomSceneFromDirParams]):
    def post_init(self):
        self._dir = Path(self.params.dir).resolve()

        if not self._dir.exists():
            raise ValueError(f"Directory {self._dir} does not exist")

        self._scenes = []
        for file in self._dir.iterdir():
            self._scenes.append("/" + str(file.relative_to(scenes_root)))

        if not self._scenes:
            raise ValueError(f"No files found in {self._dir}")

    def get_children(self) -> list[ChildrenAction]:
        candidates = [cast(RandomSceneCandidate, {"scene": scene, "weight": 1.0}) for scene in self._scenes]
        return [
            ChildrenAction(
                scene=RandomScene.factory({"candidates": candidates}),
                where="full",
            ),
            *self.children,
        ]

    def render(self):
        pass
