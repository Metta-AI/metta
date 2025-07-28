from typing import cast

import numpy as np
from omegaconf import DictConfig, OmegaConf

from metta.common.util.config import Config
from metta.map.scene import Scene, load_class, make_scene, scene_cfg_to_dict
from metta.map.scenes.room_grid import RoomGrid, RoomGridParams
from metta.map.scenes.transplant_scene import TransplantScene
from metta.map.types import MapGrid
from metta.mettagrid.level_builder import Level, LevelBuilder

from .types import Area, AreaWhere, ChildrenAction, SceneCfg


class MapGenParams(Config):
    # Root scene configuration.
    # In YAML configs, this is usually the dict with `type` and `params` keys, and possible children.
    # This is the only required parameter.
    root: SceneCfg

    # Inner grid size. Doesn't take outer border into account.
    # If `instances` is set, this is the size used for each instance.
    # If width and height are not set, the root scene must provide an intrinsic size.
    width: int | None = None
    height: int | None = None

    # Default border_width value guarantees that agents don't see beyond the outer walls.
    # This value usually shouldn't be changed.
    border_width: int = 5

    # Random seed. If not set, a random seed will be generated.
    # Seeds for root scene and all its children will be derived from this seed, unless they set their own seeds.
    seed: int | None = None

    ##### Multiple instances parameters #####

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


# Root map generator, based on scenes.
class MapGen(LevelBuilder):
    def __init__(self, **kwargs):
        params = MapGenParams(**kwargs)

        self.root = params.root
        if isinstance(self.root, DictConfig):
            self.root: SceneCfg = cast(dict, OmegaConf.to_container(self.root))

        self.width = params.width
        self.height = params.height
        self.border_width = params.border_width
        self.instances = params.instances
        self.num_agents = params.num_agents
        self.instance_border_width = params.instance_border_width
        self.seed = params.seed
        self.rng = np.random.default_rng(self.seed)

    def __getstate__(self):
        """Prepare the object for pickling."""
        state = self.__dict__.copy()
        # Store the random state instead of the generator itself
        state["_rng_state"] = self.rng.__getstate__()
        del state["rng"]
        # Convert any remaining DictConfig to regular dict
        if isinstance(state.get("root"), DictConfig):
            state["root"] = cast(dict, OmegaConf.to_container(state["root"]))
        # Remove root_scene if it exists (it's created during build())
        state.pop("root_scene", None)
        # Remove any other build-time attributes
        state.pop("grid", None)
        state.pop("inner_width", None)
        state.pop("inner_height", None)
        return state

    def __setstate__(self, state):
        """Restore the object from pickled state."""
        # Restore the random generator from its state
        rng_state = state.pop("_rng_state")
        self.__dict__.update(state)
        self.rng = np.random.default_rng()
        self.rng.__setstate__(rng_state)

    def build(self):
        if not self.width or not self.height:
            dict_cfg = scene_cfg_to_dict(self.root)
            root_cls = load_class(dict_cfg["type"])
            intrinsic_size = root_cls.intrinsic_size(dict_cfg.get("params", {}))
            if not intrinsic_size:
                raise ValueError("width and height must be provided if the root scene has no intrinsic size")
            self.height, self.width = intrinsic_size

        single_instance_scene: Scene | None = None
        if self.num_agents:
            # Auto-detect the number of instances.
            # We'll render the first instance in a separate grid to count the number of agents.
            # Then we'll transplant it into the final multi-instance grid.
            single_instance_grid = np.full((self.height, self.width), "empty", dtype="<U50")
            single_instance_area = Area(
                x=0, y=0, width=self.width, height=self.height, grid=single_instance_grid, tags=[]
            )
            single_instance_scene = make_scene(self.root, single_instance_area, rng=self.rng)
            single_instance_scene.render_with_children()
            single_instance_num_agents = int(np.count_nonzero(np.char.startswith(single_instance_grid, "agent")))
            if self.num_agents % single_instance_num_agents != 0:
                raise ValueError(
                    f"Number of agents {self.num_agents} is not divisible by number of agents in a single instance"
                    f" {single_instance_num_agents}"
                )
            instances = self.num_agents // single_instance_num_agents

            # Usually, when num_agents is set, you don't need to set `instances` explicitly.
            if self.instances and self.instances != instances:
                raise ValueError(
                    f"Derived number of instances {instances} does not match the explicitly requested"
                    f" number of instances {self.instances}"
                )
            self.instances = instances

        if self.instances is None:
            # neither `instances` nor `num_agents` were set, so we'll generate a single instance
            self.instances = 1

        ######### Prepare the full grid and its inner area #########
        instance_rows = int(np.ceil(np.sqrt(self.instances)))
        instance_cols = int(np.ceil(self.instances / instance_rows))

        self.inner_width = self.width * instance_cols + (instance_cols - 1) * self.instance_border_width
        self.inner_height = self.height * instance_rows + (instance_rows - 1) * self.instance_border_width

        bw = self.border_width

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

        inner_area = Area(x=bw, y=bw, width=self.inner_width, height=self.inner_height, grid=inner_grid, tags=[])

        ######### Prepare the root scene #########
        root_scene_cfg = self.root

        if self.instances > 1:
            children_actions: list[ChildrenAction] = []
            if single_instance_scene:
                # first instance is already rendered, so we want to transplant it into our larger grid
                children_actions.append(
                    ChildrenAction(
                        scene=TransplantScene.factory(
                            {
                                "scene": single_instance_scene,
                                "grid": self.grid,
                            }
                        ),
                        where=AreaWhere(tags=["room"]),
                        limit=1,
                        order_by="first",
                        lock="lock",
                    )
                )

            children_actions.append(
                ChildrenAction(
                    scene=self.root,
                    where=AreaWhere(tags=["room"]),
                    limit=self.instances - (1 if single_instance_scene else 0),
                    order_by="first",
                    lock="lock",
                )
            )

            root_scene_cfg = RoomGrid.factory(
                RoomGridParams(
                    rows=instance_rows,
                    columns=instance_cols,
                    border_width=self.instance_border_width,
                ),
                children_actions=children_actions,
            )

        ######### Render the root scene #########
        self.root_scene = make_scene(root_scene_cfg, inner_area, rng=self.rng)

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
