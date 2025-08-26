from typing import TYPE_CHECKING, Optional

import numpy as np
from pydantic import Field, model_validator

from metta.mettagrid.map_builder.ascii import AsciiMapBuilder
from metta.mettagrid.map_builder.map_builder import AnyMapBuilderConfig

if TYPE_CHECKING:
    pass

from metta.map.scene import ChildrenAction, SceneConfigOrFile, make_scene, resolve_scene_config
from metta.map.scenes.copy_grid import CopyGrid
from metta.map.scenes.room_grid import RoomGrid
from metta.map.scenes.transplant_scene import TransplantScene
from metta.mettagrid.map_builder import GameMap, MapBuilder, MapBuilderConfig
from metta.mettagrid.map_builder.utils import create_grid

from .types import Area, AreaWhere


# Root map generator, based on scenes.
class MapGen(MapBuilder):
    class Config(MapBuilderConfig["MapGen"]):
        ########## Global parameters ##########

        # Default border_width value guarantees that agents don't see beyond the outer walls.
        # This value usually shouldn't be changed.
        border_width: int = Field(default=5, ge=0)

        # Random seed. If not set, a random seed will be generated.
        # Seeds for root scene and all its children will be derived from this seed, unless they set their own seeds.
        seed: int | None = Field(default=None, ge=0)

        ########## Single instance parameters ##########

        # Root scene configuration.
        # In YAML configs, this is usually the dict with `type` and `params` keys, and possible children.
        root: Optional[SceneConfigOrFile] = None

        # Inner grid size. Doesn't take outer border into account.
        # If `instances` is set, this is the size used for each instance.
        # The map must have either `width`, `height` and `root` set, or `instance_map`.
        width: Optional[int] = Field(default=None, ge=0)
        height: Optional[int] = Field(default=None, ge=0)

        # Alternative to `root`: Root map configuration.
        # Either this or `root` must be set.
        # The difference is that `root` doesn't have an intrinsic size, so you
        # need to set `width` and `height` explicitly.
        # `instance_map` must point to a `MapBuilder` configuration, with the class name specified in `type`, and params
        # specified in `params` dict.
        instance_map: Optional[AnyMapBuilderConfig] = Field(default=None)

        ########## Multiple instances parameters ##########

        # MapGen can place multiple instances of the root scene on the
        # grid. This is useful for additional parallelization.
        # By default, the map will be generated as a single root scene instance, with the given width and height.
        #
        # There are two ways to get multiple root scene instances:
        # 1. Set `instances` explicitly to the number of instances that you need.
        # 2. Set `num_agents` and allow MapGen to compute the number of instances based on it.
        #
        # In either case, if the number of instances is larger than 1, MapGen will organize them in a grid separated by
        # borders, and make the overall grid as square as possible.

        # Number of root scene instances to generate. If set, the map will be generated
        # as a grid of instances, separated by
        # the given `instance_border_width`.
        instances: Optional[int] = Field(default=None, ge=1)

        # Number of agents to generate. If set, MapGen will automatically compute the
        # number of instances based on how many
        # agents there are in the root scene.
        num_agents: Optional[int] = Field(default=None, ge=0)

        # Inner border width between instances. This value usually shouldn't be changed.
        instance_border_width: int = Field(default=5, ge=0)

        @model_validator(mode="after")
        def validate_required_fields(self) -> "MapGen.Config":
            """Validate that either (root, width, height) are all set, or instance_map is set."""
            has_basic_config = self.root is not None
            has_instance_map = self.instance_map is not None and self.width is None and self.height is None

            # XOR-ing the two booleans to check if exactly one of them is True
            if has_basic_config == has_instance_map:
                raise ValueError("Either (root, width, height) must all be set, or instance_map must be set")

            return self

        @classmethod
        def with_ascii_uri(cls, ascii_map_uri: str, **kwargs) -> "MapGen.Config":
            """Create a MapGenConfig with an ASCII map file as the instance_map."""
            kwargs["instance_map"] = AsciiMapBuilder.Config.from_uri(ascii_map_uri)
            return cls(**kwargs)

    def __init__(self, config: Config):
        self.config = config

        self.root = self.config.root

        self.rng = np.random.default_rng(self.config.seed)

    def prebuild_instances(self):
        """Prebuild instances when using instance_map or deriving instances from num_agents.

        Sets self.instances, self.width, self.height, and self.instance_scene_factories."""
        self.instance_scene_factories: list[SceneConfigOrFile] = []

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
            if self.config.instance_map and self.instances and self.instances > len(self.instance_scene_factories):
                # We need to prebuild all instances.
                return True
            return False

        while continue_to_prerender():
            # Auto-detect the number of instances.
            # We'll render the first instance in a separate grid to count the number of agents.
            # Then we'll transplant it into the final multi-instance grid.

            if self.root:
                if not self.width or not self.height:
                    root_config = resolve_scene_config(self.root)
                    intrinsic_size = root_config.type.intrinsic_size(root_config.params)
                    if not intrinsic_size:
                        raise ValueError("width and height must be provided if the root scene has no intrinsic size")
                    self.height, self.width = intrinsic_size

                instance_grid = create_grid(self.height, self.width)
                instance_area = Area(x=0, y=0, width=self.width, height=self.height, grid=instance_grid, tags=[])
                instance_scene = make_scene(self.root, instance_area, rng=self.rng)
                instance_scene.render_with_children()
                self.instance_scene_factories.append(
                    TransplantScene.factory(
                        params=TransplantScene.Params(
                            scene=instance_scene,
                            get_grid=lambda: self.grid,
                        )
                    )
                )
            else:
                assert self.config.instance_map is not None
                # Instance is a map, not a scene, so it defines its own size.
                # We need to prerender it to find the full size of our grid.
                instance_map = self.config.instance_map.create()
                if not isinstance(instance_map, MapBuilder):
                    raise ValueError("instance_map must be a MapBuilder")

                instance_level = instance_map.build()
                instance_grid = instance_level.grid

                self.instance_scene_factories.append(
                    # TODO - if the instance_map class is MapGen, we want to transplant its scene tree too.
                    CopyGrid.factory(
                        params=CopyGrid.Params(grid=instance_grid),
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

        inner_grid = self.grid[
            bw : bw + self.inner_height,
            bw : bw + self.inner_width,
        ]

        self.inner_area = Area(x=bw, y=bw, width=self.inner_width, height=self.inner_height, grid=inner_grid, tags=[])

    def get_root_scene_cfg(self) -> SceneConfigOrFile:
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
                assert self.root, "Internal logic error: no root config but no prebuilt instances either"
                return self.root

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
            assert self.root, "Internal logic error: MapGen failed to prebuild enough instances"

            children_actions.append(
                ChildrenAction(
                    scene=self.root,
                    where=AreaWhere(tags=["room"]),
                    limit=remaining_instances,
                    order_by="first",
                    lock="lock",
                )
            )

        return RoomGrid.factory(
            RoomGrid.Params(
                rows=self.instance_rows,
                columns=self.instance_cols,
                border_width=self.config.instance_border_width,
            ),
            children_actions=children_actions,
        )

    def build(self):
        self.prebuild_instances()
        self.prepare_grid()

        root_scene_cfg = self.get_root_scene_cfg()

        self.root_scene = make_scene(root_scene_cfg, self.inner_area, rng=self.rng)
        self.root_scene.render_with_children()

        return GameMap(self.grid)

    def get_scene_tree(self) -> dict:
        return self.root_scene.get_scene_tree()
