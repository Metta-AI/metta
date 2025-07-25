import copy

from pydantic import ConfigDict, Field

from metta.common.util.config import Config
from metta.map.scene import Scene
from metta.map.types import MapGrid


class TransplantSceneParams(Config):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    scene: Scene = Field(exclude=True)  # scene that will be transplanted
    grid: MapGrid = Field(exclude=True)  # full outer grid


class TransplantScene(Scene[TransplantSceneParams]):
    def render(self):
        if self.width != self.params.scene.width or self.height != self.params.scene.height:
            raise ValueError(
                "TransplantScene can only be used with scenes that have the same width and height as the parent grid"
            )

        scene_copy = copy.deepcopy(self.params.scene)
        scene_copy.transplant_to_grid(
            self.params.grid, self.area.x - self.params.scene.area.x, self.area.y - self.params.scene.area.y
        )
        self.children.append(scene_copy)

    def get_labels(self) -> list[str]:
        return self.params.scene.get_labels()
