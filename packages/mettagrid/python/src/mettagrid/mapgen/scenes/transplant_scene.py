import copy

import pydantic

import mettagrid.mapgen.scene


class TransplantSceneConfig(mettagrid.mapgen.scene.SceneConfig):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    # Scene that will be transplanted.
    scene: mettagrid.mapgen.scene.Scene = pydantic.Field(exclude=True)


class TransplantScene(mettagrid.mapgen.scene.Scene[TransplantSceneConfig]):
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
