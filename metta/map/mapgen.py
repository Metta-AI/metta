from typing import Any, cast

import numpy as np
from omegaconf import DictConfig, OmegaConf
from pydantic import model_validator

from metta.common.util.config import Config
from metta.map.scene import load_class, make_scene, scene_cfg_to_dict
from metta.map.scenes.copy_grid import CopyGrid
from metta.map.scenes.room_grid import RoomGrid, RoomGridParams
from metta.map.scenes.transplant_scene import TransplantScene
from metta.map.types import MapGrid
from metta.mettagrid.level_builder import Level, LevelBuilder

from .types import Area, AreaWhere, ChildrenAction, SceneCfg


class MapGenParams(Config):
    ########## Global parameters ##########

    # Default border_width value guarantees that agents don't see beyond the outer walls.
    # This value usually shouldn't be changed.
    border_width: int = 5

    # Random seed. If not set, a random seed will be generated.
    # Seeds for root scene and all its children will be derived from this seed, unless they set their own seeds.
    seed: int | None = None

    ########## Single instance parameters ##########

    # Root scene configuration.
    # In YAML configs, this is usually the dict with `type` and `params` keys, and possible children.
    root: SceneCfg | None = None

    # Inner grid size. Doesn't take outer border into account.
    # If `instances` is set, this is the size used for each instance.
    # The map must have either `width`, `height` and `root` set, or `instance_map`.
    width: int | None = None
    height: int | None = None

    # Alternative to `root`: Root map configuration.
    # Either this or `root` must be set.
    # The difference is that `root` doesn't have an intrinsic size, so you need to set `width` and `height` explicitly.
    # `instance_map` must point to a `LevelBuilder` configuration, with the class name specified in `type`, and params
    # specified in `params` dict.
    instance_map: dict[str, Any] | None = None

    ########## Multiple instances parameters ##########

    # MapGen can place multiple instances of the root scene on the grid. This is useful for additional parallelization.
    # By default, the map will be generated as a single root scene instance, with the given width and height.
    #
    # There are two ways to get multiple root scene instances:
    # 1. Set `instances` explicitly to the number of instances that you need.
    # 2. Set `num_agents` and allow MapGen to compute the number of instances based on it.
    #
    # In either case, if the number of instances is larger than 1, MapGen will organize them in a grid separated by
    # borders, and make the overall grid as square as possible.

    # Number of root scene instances to generate. If set, the map will be generated as a grid of instances, separated by
    # the given `instance_border_width`.
    instances: int | None = None

    # Number of agents to generate. If set, MapGen will automatically compute the number of instances based on how many
    # agents there are in the root scene.
    num_agents: int | None = None

    # Inner border width between instances. This value usually shouldn't be changed.
    instance_border_width: int = 5

    @model_validator(mode="after")
    def validate_required_fields(self) -> "MapGenParams":
        """Validate that either (root, width, height) are all set, or instance_map is set."""
        has_basic_config = self.root is not None
        has_instance_map = self.instance_map is not None and self.width is None and self.height is None

        # XOR-ing the two booleans to check if exactly one of them is True
        if has_basic_config == has_instance_map:
            raise ValueError("Either (root, width, height) must all be set, or instance_map must be set")

        return self


# Root map generator, based on scenes.
class MapGen(LevelBuilder):
    def __init__(self, **kwargs):
        params = MapGenParams(**kwargs)
        self.params = params

        self.root = params.root
        if isinstance(self.root, DictConfig):
            self.root = cast(dict, OmegaConf.to_container(self.root))

        self.rng = np.random.default_rng(self.params.seed)

    def prebuild_instances(self):
        """
        In some cases, we need to render individual instances in separate grids before we render the final grid.

        This is the case for:
        1) Using `instance_map` (which is a map, not a scene, so it defines its own size).
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
        self.instance_scene_factories: list[SceneCfg] = []

        # Can be None, but we'll set these fields to their actual values after the loop.
        self.width = self.params.width
        self.height = self.params.height
        self.instances = self.params.instances

        def continue_to_prerender():
            if not self.width or not self.height:
                # We haven't detected the instance size yet.
                return True
            if self.params.num_agents and not len(self.instance_scene_factories):
                # We need to derive the number of instances from the number of agents by rendering at least one
                # instance.
                return True
            if self.params.instance_map and self.instances and self.instances > len(self.instance_scene_factories):
                # We need to prebuild all instances.
                return True
            return False

        while continue_to_prerender():
            # Auto-detect the number of instances.
            # We'll render the first instance in a separate grid to count the number of agents.
            # Then we'll transplant it into the final multi-instance grid.

            if self.root:
                if not self.width or not self.height:
                    dict_cfg = scene_cfg_to_dict(self.root)
                    root_cls = load_class(dict_cfg["type"])
                    intrinsic_size = root_cls.intrinsic_size(dict_cfg.get("params", {}))
                    if not intrinsic_size:
                        raise ValueError("width and height must be provided if the root scene has no intrinsic size")
                    self.height, self.width = intrinsic_size

                instance_grid = np.full((self.height, self.width), "empty", dtype="<U50")
                instance_area = Area(x=0, y=0, width=self.width, height=self.height, grid=instance_grid, tags=[])
                instance_scene = make_scene(self.root, instance_area, rng=self.rng)
                instance_scene.render_with_children()
                self.instance_scene_factories.append(
                    TransplantScene.factory(
                        {
                            "scene": instance_scene,
                            "get_grid": lambda: self.grid,
                        }
                    )
                )
            else:
                assert self.params.instance_map is not None
                # Instance is a map, not a scene, so it defines its own size.
                # We need to prerender it to find the full size of our grid.
                instance_map_cls = load_class(self.params.instance_map["type"], check_is_scene=False)
                instance_map = instance_map_cls(**self.params.instance_map["params"])
                if not isinstance(instance_map, LevelBuilder):
                    raise ValueError("instance_map must be a LevelBuilder")

                instance_level = instance_map.build()
                instance_grid = instance_level.grid

                self.instance_scene_factories.append(
                    # TODO - if the instance_map class is MapGen, we want to transplant its scene tree too.
                    CopyGrid.factory(
                        {
                            "grid": instance_grid,
                            "labels": instance_level.labels,
                        }
                    )
                )
                self.width = max(self.width or 0, instance_grid.shape[1])
                self.height = max(self.height or 0, instance_grid.shape[0])

            if self.params.num_agents and len(self.instance_scene_factories) == 1:
                # First prebuilt instance, let's derive the number of instances from the number of agents.
                instance_num_agents = int(np.count_nonzero(np.char.startswith(instance_grid, "agent")))
                if self.params.num_agents % instance_num_agents != 0:
                    raise ValueError(
                        f"Number of agents {self.params.num_agents} is not divisible by number of agents"
                        f" in a single instance {instance_num_agents}"
                    )
                instances = self.params.num_agents // instance_num_agents

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
        """
        Prepare the full grid and its inner area.
        """
        assert self.instances is not None

        self.instance_rows = int(np.ceil(np.sqrt(self.instances)))
        self.instance_cols = int(np.ceil(self.instances / self.instance_rows))

        assert self.width is not None and self.height is not None

        self.inner_width = (
            self.width * self.instance_cols + (self.instance_cols - 1) * self.params.instance_border_width
        )
        self.inner_height = (
            self.height * self.instance_rows + (self.instance_rows - 1) * self.params.instance_border_width
        )

        bw = self.params.border_width

        self.grid: MapGrid = np.full(
            (self.inner_height + 2 * bw, self.inner_width + 2 * bw),
            "empty",
            dtype="<U50",
        )

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

    def get_root_scene_cfg(self) -> SceneCfg:
        """
        Create the full root scene, which might contain multiple instances.
        """
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
            RoomGridParams(
                rows=self.instance_rows,
                columns=self.instance_cols,
                border_width=self.params.instance_border_width,
            ),
            children_actions=children_actions,
        )

    def build(self):
        self.prebuild_instances()
        self.prepare_grid()

        root_scene_cfg = self.get_root_scene_cfg()

        self.root_scene = make_scene(root_scene_cfg, self.inner_area, rng=self.rng)
        self.root_scene.render_with_children()

        labels = self.root_scene.get_labels()
        area = self.inner_width * self.inner_height
        if area < 4000:
            labels.append("small")
        elif area < 6000:
            labels.append("medium")
        else:
            labels.append("large")

        return Level(self.grid, labels=labels)

    def get_scene_tree(self) -> dict:
        return self.root_scene.get_scene_tree()
