from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import Field, ValidatorFunctionWrapHandler, field_validator, model_validator

from mettagrid.map_builder import AnyMapBuilderConfig, GameMap, MapBuilder, MapBuilderConfig, MapGrid
from mettagrid.map_builder.map_builder import validate_any_map_builder
from mettagrid.map_builder.utils import create_grid
from mettagrid.mapgen.area import Area, AreaWhere
from mettagrid.mapgen.scene import ChildrenAction, Scene, SceneConfig, load_symbol, validate_any_scene_config
from mettagrid.mapgen.scenes.copy_grid import CopyGrid
from mettagrid.mapgen.scenes.room_grid import RoomGrid
from mettagrid.mapgen.scenes.transplant_scene import TransplantScene


class MapGen(MapBuilder):
    class Config(MapBuilderConfig["MapGen"]):
        ########## Global parameters ##########

        border_width: int = Field(
            default=5,
            ge=0,
            description="Default value guarantees that agents don't see beyond the outer walls. This value usually "
            "shouldn't be changed.",
        )

        seed: int | None = Field(
            default=None,
            ge=0,
            description="Random seed. If not set, a random seed will be generated. Seeds for root"
            " scene and all its children will be derived from this seed, unless they set their own seeds.",
        )

        ########## Single instance parameters ##########

        # Configuration of the instance scene.
        # Can be either a scene config or another MapBuilder config.
        # If it's a scene config, you need to set `width` and `height` explicitly.
        # If `instances` or `num_agents` are set, this configuration will be used multiple times.
        instance: SceneConfig | AnyMapBuilderConfig | None = Field(default=None)

        # Legacy fields, to be removed soon.
        instance_map: AnyMapBuilderConfig | None = Field(default=None, deprecated="Use `instance` instead")
        root: SceneConfig | None = Field(default=None, deprecated="Use `instance` instead")

        @model_validator(mode="before")
        @classmethod
        def validate_legacy_instance_fields(cls, data):
            # Temporary validation for legacy fields, to avoid merge collisions with other PRs.
            if isinstance(data, cls) or not isinstance(data, dict):
                return data

            if data.get("instance") is not None:
                if data.get("instance_map") is not None or data.get("root") is not None:
                    raise ValueError("instance, instance_map, and root cannot be set at the same time")
                return data

            if data.get("instance_map") is not None:
                data["instance"] = data["instance_map"]
                del data["instance_map"]

            if data.get("root") is not None:
                if data.get("instance") is not None:
                    raise ValueError("instance_map and root cannot be set at the same time")
                data["instance"] = data["root"]
                del data["root"]

            return data

        @field_validator("instance", mode="wrap")
        @classmethod
        def _validate_instance(cls, v: Any, handler: ValidatorFunctionWrapHandler) -> SceneConfig | AnyMapBuilderConfig:
            if isinstance(v, SceneConfig):
                return v
            elif isinstance(v, MapBuilderConfig):
                return v
            elif isinstance(v, dict):
                # We need to decide whether it's a scene config or a MapBuilder config.
                # Either of them can be polymorphic, so Pydantic won't decide this for us.
                t = v.get("type")
                if t is None:
                    raise ValueError("'type' is required")
                target = load_symbol(t) if isinstance(t, str) else t
                if isinstance(target, type) and issubclass(target, Scene):
                    return validate_any_scene_config(v)
                elif isinstance(target, type) and issubclass(target, MapBuilder):
                    return validate_any_map_builder(v)
                else:
                    raise ValueError(f"Invalid instance type: {target!r}")
            else:
                raise ValueError(f"Invalid instance configuration: {v!r}")

        width: int | None = Field(
            default=None,
            ge=0,
            description="""Inner grid width. Doesn't take outer border into account. If `instance` is a MapBuilder
            config, this field must be None; otherwise, it must be set. If `instances` is set, this is the size used for
            each instance.""",
        )
        height: int | None = Field(
            default=None,
            ge=0,
            description="""Inner grid width. Doesn't take outer border into account. If `instance` is a MapBuilder
            config, this field must be None; otherwise, it must be set. If `instances` is set, this is the size used for
            each instance.""",
        )

        ########## Multiple instances parameters ##########

        # MapGen can place multiple instances of the given instance configuration on the grid.
        #
        # This is useful for additional parallelization. By default, the map will be generated as a single instance
        # scene, with the given width and height.
        #
        # There are two ways to get multiple instances:
        # 1. Set `instances` explicitly to the number of instances that you need.
        # 2. Set `num_agents` and allow MapGen to compute the number of instances based on it.
        #
        # In either case, if the number of instances is larger than 1, MapGen will organize them in a grid separated by
        # borders, and make the overall grid as square as possible.

        # Number of instances to generate. If set, the map will be generated as a grid of instances, separated by the
        # given `instance_border_width`.
        instances: int | None = Field(default=None, ge=1)

        # Number of agents to generate. If set, MapGen will automatically compute the number of instances based on how
        # many agents there are in the instance scene. (It will assume that the instance always places the same number
        # of agents.)
        num_agents: int | None = Field(default=None, ge=0)

        # Inner border width between instances. This value usually shouldn't be changed.
        instance_border_width: int = Field(default=5, ge=0)

        @model_validator(mode="after")
        def validate_required_fields(self) -> MapGen.Config:
            if not self.instance:
                raise ValueError("instance is required")

            if isinstance(self.instance, MapBuilderConfig):
                if self.width is not None or self.height is not None:
                    raise ValueError("width and height must be None if instance is a MapBuilder config")

            # The opposite situation, when `instance` is a scene config, but width and height are set,
            # could be valid, if the scene has an intrinsic size.

            return self

    def __init__(self, config: Config):
        self.config = config

        self.rng = np.random.default_rng(self.config.seed)
        self.grid = None

    def guarded_grid(self) -> MapGrid:
        assert self.grid is not None
        return self.grid

    def prebuild_instances(self):
        """In some cases, we need to render individual instances in separate grids before we render the final grid.

        This is the case for:
        1) If `instance` is a MapBuilderConfig (not a scene, so it defines its own size).
        2) Using `num_agents`, where we don't know the number of instances in advance.

        In both of these cases, we render to the temporary grid first, and then copy ("transplant") the result into the
        final grid.

        This allows us to find the number of instances and the width/height of the final grid.

        Note that we prefer _not_ to prebuild scenes in advance: this complicates the final scene tree and the
        implementation logic. (It's also a little bit slower, but this part is negligible.)

        After this method is done, we'll have the following fields set:
        - `self.instances` (either copied from the config, or derived from the number of agents)
        - `self.width` (either copied from the config, or derived from the instance map)
        - `self.height` (either copied from the config, or derived from the instance map)
        - `self.instance_scene_factories` (a list of scene factories, one for each instance)
        """
        self.instance_scene_factories: list[SceneConfig] = []

        # Can be None, but we'll set these fields to their actual values after the loop.
        self.width = self.config.width
        self.height = self.config.height
        self.instances = self.config.instances

        def continue_to_prerender():
            if not self.width or not self.height:
                # We haven't detected the instance size yet.
                return True
            if self.config.num_agents and not len(self.instance_scene_factories):
                # We need to derive the number of instances from the number of agents by rendering at least one
                # instance.
                return True
            if (
                isinstance(self.config.instance, MapBuilderConfig)
                and self.instances
                and self.instances > len(self.instance_scene_factories)
            ):
                # We need to prebuild all instances.
                return True
            return False

        while continue_to_prerender():
            # Auto-detect the number of instances.
            # We'll render the first instance in a separate grid to count the number of agents.
            # Then we'll transplant it into the final multi-instance grid.

            if isinstance(self.config.instance, SceneConfig):
                instance_scene_config = self.config.instance
                if not self.width or not self.height:
                    intrinsic_size = instance_scene_config.scene_cls.intrinsic_size(self.config.instance)
                    if not intrinsic_size:
                        raise ValueError(
                            "width and height must be provided if the instance scene has no intrinsic size"
                        )
                    if instance_scene_config.transform.transpose:
                        intrinsic_size = intrinsic_size[::-1]
                    self.height, self.width = intrinsic_size

                instance_grid = create_grid(self.height, self.width)
                instance_area = Area.root_area_from_grid(instance_grid)
                instance_scene = instance_scene_config.create_root(instance_area, self.rng)
                instance_scene.render_with_children()
                self.instance_scene_factories.append(TransplantScene.Config(scene=instance_scene))
            else:
                assert isinstance(self.config.instance, MapBuilderConfig)
                # Instance is a map, not a scene, so it defines its own size.
                # We need to prerender it to find the full size of our grid.
                instance_map_builder = self.config.instance.create()
                if not isinstance(instance_map_builder, MapBuilder):
                    raise ValueError("instance must be a MapBuilder")

                instance_map = instance_map_builder.build()
                instance_grid = instance_map.grid

                self.instance_scene_factories.append(
                    # TODO - if the instance class is MapGen, we want to transplant its scene tree too.
                    CopyGrid.Config(
                        grid=instance_grid,
                    )
                )
                self.width = max(self.width or 0, instance_grid.shape[1])
                self.height = max(self.height or 0, instance_grid.shape[0])

            if self.config.num_agents and len(self.instance_scene_factories) == 1:
                # First prebuilt instance, let's derive the number of instances from the number of agents.
                instance_num_agents = int(np.count_nonzero(np.char.startswith(instance_grid, "agent")))
                if self.config.num_agents % instance_num_agents != 0:
                    raise ValueError(
                        f"Number of agents {self.config.num_agents} is not divisible by number of agents"
                        f" in a single instance {instance_num_agents}"
                    )
                instances = self.config.num_agents // instance_num_agents

                # Usually, when num_agents is set, you don't need to set `instances` explicitly.
                if self.instances and self.instances != instances:
                    raise ValueError(
                        f"Derived number of instances {instances} does not match the explicitly requested"
                        f" number of instances {self.instances}"
                    )
                self.instances = instances

        if self.instances is None:
            self.instances = 1

    def prepare_grid(self):
        """Prepare the full grid with outer walls and inner area for instances."""
        assert self.instances is not None

        self.instance_rows = int(np.ceil(np.sqrt(self.instances)))
        self.instance_cols = int(np.ceil(self.instances / self.instance_rows))

        assert self.width is not None and self.height is not None

        self.inner_width = (
            self.width * self.instance_cols + (self.instance_cols - 1) * self.config.instance_border_width
        )
        self.inner_height = (
            self.height * self.instance_rows + (self.instance_rows - 1) * self.config.instance_border_width
        )

        bw = self.config.border_width

        self.grid = create_grid(self.inner_height + 2 * bw, self.inner_width + 2 * bw)

        # draw outer walls
        # note that the inner walls when instances > 1 will be drawn by the RoomGrid scene
        self.grid[:bw, :] = "wall"
        self.grid[-bw:, :] = "wall"
        self.grid[:, :bw] = "wall"
        self.grid[:, -bw:] = "wall"

        self.inner_area = Area(
            outer_grid=self.grid,
            x=bw,
            y=bw,
            width=self.inner_width,
            height=self.inner_height,
        )

    def get_root_scene_cfg(self) -> SceneConfig:
        """Create the full root scene configuration, handling single or multiple instances."""
        assert self.instances is not None

        if self.instances == 1:
            if self.instance_scene_factories:
                # We need just one instance and it's already prebuilt.
                assert len(self.instance_scene_factories) == 1, (
                    "Internal logic error: MapGen wants 1 instance but prebuilt more"
                )
                return self.instance_scene_factories[0]
            else:
                assert isinstance(self.config.instance, SceneConfig), (
                    "Internal logic error: instance is not a scene but we don't have prebuilt instances either"
                )
                return self.config.instance

        # We've got more than one instance, so we'll need a RoomGrid.

        children_actions: list[ChildrenAction] = []
        for instance_scene_factory in self.instance_scene_factories:
            children_actions.append(
                ChildrenAction(
                    scene=instance_scene_factory,
                    where=AreaWhere(tags=["room"]),
                    limit=1,
                    order_by="first",
                    lock="lock",
                )
            )

        remaining_instances = self.instances - len(self.instance_scene_factories)

        if remaining_instances > 0:
            assert isinstance(self.config.instance, SceneConfig), (
                "Internal logic error: MapGen failed to prebuild enough instances"
            )

            children_actions.append(
                ChildrenAction(
                    scene=self.config.instance,
                    where=AreaWhere(tags=["room"]),
                    limit=remaining_instances,
                    order_by="first",
                    lock="lock",
                )
            )

        return RoomGrid.Config(
            rows=self.instance_rows,
            columns=self.instance_cols,
            border_width=self.config.instance_border_width,
            children=children_actions,
        )

    def build(self):
        if self.grid is not None:
            return GameMap(self.grid)

        self.prebuild_instances()
        self.prepare_grid()

        root_scene_cfg = self.get_root_scene_cfg()

        self.root_scene = root_scene_cfg.create_root(self.inner_area, self.rng)
        self.root_scene.render_with_children()

        return GameMap(self.guarded_grid())

    def get_scene_tree(self) -> dict:
        return self.root_scene.get_scene_tree()
