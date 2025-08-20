import copy
from typing import Callable

from pydantic import ConfigDict, Field

from metta.common.util.config import Config
from metta.map.scene import Scene
from metta.map.types import MapGrid


class TransplantSceneParams(Config):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Scene that will be transplanted.
    scene: Scene = Field(exclude=True)

    # Callback to get the full outer grid.
    # Why? Because in MapGen class it's convenient to pass scene factories around, and we don't have the final grid yet
    # when we construct the factory for this scene.
    # Sorry about the complexity; it might be possible to implement `transplant_to_grid` without having the full outer
    # grid object, but it's *much* easier when we have it.
    get_grid: Callable[[], MapGrid] = Field(exclude=True)


class TransplantScene(Scene[TransplantSceneParams]):
    """
    This is a helper scene that allows us to salvage the scene tree from a scene that was rendered on an external grid.

    See the logic about prebuilt instances in MapGen to understand why this is needed.
    """

    def render(self):
        if self.width != self.params.scene.width or self.height != self.params.scene.height:
            raise ValueError(
                "TransplantScene can only be used with scenes that have the same width and height as the parent grid"
            )

        scene_copy = copy.deepcopy(self.params.scene)
        scene_copy.transplant_to_grid(
            self.params.get_grid(), self.area.x - self.params.scene.area.x, self.area.y - self.params.scene.area.y
        )
        self.children.append(scene_copy)

    def get_labels(self) -> list[str]:
        return self.params.scene.get_labels()
