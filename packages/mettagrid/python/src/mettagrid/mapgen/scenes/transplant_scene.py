import copy

from pydantic import ConfigDict, Field

from mettagrid.mapgen.scene import Scene, SceneConfig


class TransplantSceneConfig(SceneConfig):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Scene that will be transplanted.
    scene: Scene = Field(exclude=True)


class TransplantScene(Scene[TransplantSceneConfig]):
    """
    This is a helper scene that allows us to salvage the scene tree from a scene that was rendered on an external grid.

    See the logic about prebuilt instances in MapGen to understand why this is needed.
    """

    def render(self):
        if self.width != self.config.scene.area.width or self.height != self.config.scene.area.height:
            raise ValueError(
                "TransplantScene can only be used with scenes that have the same width and height as the parent grid"
            )

        scene_copy = copy.deepcopy(self.config.scene)
        scene_copy.transplant_to_grid(
            self.area.outer_grid, self.area.x - self.config.scene.area.x, self.area.y - self.config.scene.area.y
        )
        self.children.append(scene_copy)
