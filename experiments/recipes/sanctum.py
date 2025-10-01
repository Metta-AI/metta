from typing import Any, List

from cogames.cogs_vs_clips import stations as st
from cogames.cogs_vs_clips.scenarios import machina_sanctum, machina_symmetry_sanctum
from metta.sim.simulation_config import SimulationConfig
from mettagrid.config.mettagrid_config import MettaGridConfig
from mettagrid.mapgen.scene import ChildrenAction
from mettagrid.mapgen.scenes.radial_objects import RadialObjects, RadialObjectsParams
from mettagrid.mapgen.types import AreaWhere


def make_mettagrid() -> MettaGridConfig:
    """MettaGridConfig

    Gridworks Config Maker: returns the Sanctum-in-the-Quadrants map with central base.
    """
    return machina_sanctum(num_cogs=4)


def make_evals() -> List[SimulationConfig]:
    """Return a list of SimulationConfig variants for Sanctum mapgen experiments.

    Variants tweak QuadrantLayout weights, QuadrantResources counts, symmetry, and
    optional quadrant-type relabeling to explore different construction methods.
    """

    def clone(env: MettaGridConfig) -> MettaGridConfig:
        return env.model_copy(deep=True)

    def get_child_params(env: MettaGridConfig, type_name: str):
        mb: Any = env.game.map_builder
        root = getattr(mb, "root", None)
        children = getattr(root, "children", None)
        if root is not None and children:
            for action in children:
                scene_type_name = getattr(
                    getattr(action.scene, "type", None), "__name__", ""
                )
                if scene_type_name == type_name:
                    return getattr(action.scene, "params", None)
        return None

    sims: list[SimulationConfig] = []

    def add_extractor_objects(env: MettaGridConfig) -> None:
        objects = env.game.objects
        # Register fast/slow variants if missing
        objects.setdefault("carbon_extractor_fast", st.carbon_extractor_fast())
        objects.setdefault("carbon_extractor_slow", st.carbon_extractor_slow())
        objects.setdefault("oxygen_extractor_fast", st.oxygen_extractor_fast())
        objects.setdefault("oxygen_extractor_slow", st.oxygen_extractor_slow())
        objects.setdefault("germanium_extractor_fast", st.germanium_extractor_fast())
        objects.setdefault("germanium_extractor_slow", st.germanium_extractor_slow())
        objects.setdefault("silicon_extractor_fast", st.silicon_extractor_fast())
        objects.setdefault("silicon_extractor_slow", st.silicon_extractor_slow())

    def add_quadrant_extractors(env: MettaGridConfig, count_per_type: int = 2) -> None:
        # Ensure objects exist
        add_extractor_objects(env)
        mb = env.game.map_builder
        root = getattr(mb, "root", None)
        if root is None:
            return
        # Place a few of each extractor per quadrant, biased away from the center
        extractor_counts = {
            "carbon_extractor_fast": count_per_type,
            "carbon_extractor_slow": count_per_type,
            "oxygen_extractor_fast": count_per_type,
            "oxygen_extractor_slow": count_per_type,
            "germanium_extractor_fast": count_per_type,
            "germanium_extractor_slow": count_per_type,
            "silicon_extractor_fast": count_per_type,
            "silicon_extractor_slow": count_per_type,
        }
        child = ChildrenAction(
            scene=RadialObjects.factory(
                RadialObjectsParams(
                    objects=extractor_counts,
                    k=2.0,
                    min_radius=6,
                    clearance=1,
                    carve=True,
                )
            ),
            where=AreaWhere(tags=["quadrant"]),
            lock="extractors",
            limit=1,
            order_by="first",
        )
        children = getattr(root, "children", None) or []
        children.append(child)
        root.children = children

    # 1) Baseline Sanctum with BSP-heavy quadrants
    env1 = clone(machina_sanctum(num_cogs=4))
    ql1 = get_child_params(env1, "QuadrantLayout")
    if ql1 is not None:
        ql1.weight_bsp10 = 1.5
        ql1.weight_bsp8 = 1.5
        ql1.weight_maze = 0.0
        ql1.weight_terrain_balanced = 0.0
        ql1.weight_terrain_maze = 0.0
    sims.append(SimulationConfig(suite="cogs_sanctum", name="sanctum_bsp", env=env1))

    # 2) Sanctum with Maze-heavy quadrants
    env2 = clone(machina_sanctum(num_cogs=4))
    ql2 = get_child_params(env2, "QuadrantLayout")
    if ql2 is not None:
        ql2.weight_bsp10 = 0.2
        ql2.weight_bsp8 = 0.2
        ql2.weight_maze = 2.0
        ql2.weight_terrain_balanced = 0.0
        ql2.weight_terrain_maze = 0.0
    sims.append(SimulationConfig(suite="cogs_sanctum", name="sanctum_maze", env=env2))

    # 3) Sanctum with terrain-maze style quadrants
    env3 = clone(machina_sanctum(num_cogs=4))
    ql3 = get_child_params(env3, "QuadrantLayout")
    if ql3 is not None:
        ql3.weight_bsp10 = 0.0
        ql3.weight_bsp8 = 0.0
        ql3.weight_maze = 0.2
        ql3.weight_terrain_balanced = 0.0
        ql3.weight_terrain_maze = 2.0
    sims.append(
        SimulationConfig(suite="cogs_sanctum", name="sanctum_terrain_maze", env=env3)
    )

    # 4) Sanctum balanced with more converters per quadrant
    env4 = clone(machina_sanctum(num_cogs=4))
    qr4 = get_child_params(env4, "QuadrantResources")
    if qr4 is not None:
        qr4.count_per_quadrant = 8
    sims.append(
        SimulationConfig(suite="cogs_sanctum", name="sanctum_balanced_8", env=env4)
    )

    # 5) Symmetry Sanctum (default)
    env5 = clone(machina_symmetry_sanctum(num_cogs=4))
    sims.append(
        SimulationConfig(suite="cogs_sanctum", name="sym_sanctum_default", env=env5)
    )

    # 6) Symmetry Sanctum with explicit quadrant type relabeling (NW/NE/SW/SE)
    env6 = clone(machina_symmetry_sanctum(num_cogs=4))
    rc6 = get_child_params(env6, "RelabelConverters")
    if rc6 is not None:
        # Use recently added quadrant_types to assign converter types per quadrant
        rc6.quadrant_types = {
            "nw": "generator_red",
            "ne": "generator_blue",
            "sw": "generator_green",
            "se": "lab",
        }
    sims.append(
        SimulationConfig(suite="cogs_sanctum", name="sym_sanctum_quadtypes", env=env6)
    )

    # 7) Symmetry Sanctum, maze-heavy terrain and higher converter density
    env7 = clone(machina_symmetry_sanctum(num_cogs=4))
    ql7 = get_child_params(env7, "QuadrantLayout")
    if ql7 is not None:
        ql7.weight_bsp10 = 0.2
        ql7.weight_bsp8 = 0.2
        ql7.weight_maze = 2.0
        ql7.weight_terrain_balanced = 0.0
        ql7.weight_terrain_maze = 0.0
    qr7 = get_child_params(env7, "QuadrantResources")
    if qr7 is not None:
        qr7.count_per_quadrant = 10
    sims.append(
        SimulationConfig(suite="cogs_sanctum", name="sym_sanctum_maze_dense", env=env7)
    )

    # 8) Symmetry Sanctum with DistanceBalance carving disabled (terrain-only balance)
    env8 = clone(machina_symmetry_sanctum(num_cogs=4))
    db8 = get_child_params(env8, "DistanceBalance")
    if db8 is not None:
        db8.balance = False
        if hasattr(db8, "carves_per_type"):
            db8.carves_per_type = 0
    sims.append(
        SimulationConfig(suite="cogs_sanctum", name="sym_sanctum_no_balance", env=env8)
    )

    # 9) Sanctum with quadrant extractors added (fast/slow variants)
    env9 = clone(machina_sanctum(num_cogs=4))
    add_quadrant_extractors(env9, count_per_type=2)
    sims.append(
        SimulationConfig(suite="cogs_sanctum", name="sanctum_extractors", env=env9)
    )

    # 10) Sanctum with assembler at center instead of altar
    env10 = clone(machina_sanctum(num_cogs=4))
    bh10 = get_child_params(env10, "BaseHub")
    if bh10 is not None:
        setattr(bh10, "altar_object", "assembler")
    sims.append(
        SimulationConfig(
            suite="cogs_sanctum", name="sanctum_assembler_center", env=env10
        )
    )

    # 11) Symmetry Sanctum with quadrant extractors and assembler center
    env11 = clone(machina_symmetry_sanctum(num_cogs=4))
    add_quadrant_extractors(env11, count_per_type=2)
    bh11 = get_child_params(env11, "BaseHub")
    if bh11 is not None:
        setattr(bh11, "altar_object", "assembler")
    sims.append(
        SimulationConfig(
            suite="cogs_sanctum", name="sym_sanctum_extractors_assembler", env=env11
        )
    )

    # 9) Linear-ish (power k=1) resource distribution in quadrants with traversal distance metric
    env9 = clone(machina_sanctum(num_cogs=4))
    qr9 = get_child_params(env9, "QuadrantResources")
    if qr9 is not None:
        qr9.mode = "power"
        qr9.k = 1.0
        qr9.distance_metric = "traversal"
    sims.append(
        SimulationConfig(
            suite="cogs_sanctum", name="sanctum_linear_traversal_resources", env=env9
        )
    )

    # 10) Exponential resource distribution with traversal distance metric
    env10 = clone(machina_sanctum(num_cogs=4))
    qr10 = get_child_params(env10, "QuadrantResources")
    if qr10 is not None:
        qr10.mode = "exp"
        qr10.alpha = 8.0
        qr10.distance_metric = "traversal"
    sims.append(
        SimulationConfig(
            suite="cogs_sanctum", name="sanctum_exp_traversal_resources", env=env10
        )
    )

    # 11) Gaussian ring-biased resource distribution in quadrants
    env11 = clone(machina_sanctum(num_cogs=4))
    qr11 = get_child_params(env11, "QuadrantResources")
    if qr11 is not None:
        qr11.mode = "gaussian"
        qr11.mu = 0.85
        qr11.sigma = 0.07
        qr11.distance_metric = "traversal"
    sims.append(
        SimulationConfig(
            suite="cogs_sanctum", name="sanctum_gaussian_traversal_resources", env=env11
        )
    )

    # 12) Sanctum with Manhattan distance weighting for resources and linear distribution
    env12 = clone(machina_sanctum(num_cogs=4))
    qr12 = get_child_params(env12, "QuadrantResources")
    if qr12 is not None:
        qr12.distance_metric = "manhattan"
        qr12.mode = "power"
        qr12.k = 1.0
    sims.append(
        SimulationConfig(
            suite="cogs_sanctum", name="sanctum_linear_manhattan_resources", env=env12
        )
    )

    # 13) Sanctum with path-aware traversal distance weighting for resources
    env13 = clone(machina_sanctum(num_cogs=4))
    qr13 = get_child_params(env13, "QuadrantResources")
    if qr13 is not None:
        qr13.distance_metric = "traversal"
    sims.append(
        SimulationConfig(
            suite="cogs_sanctum", name="sanctum_traversal_resources", env=env13
        )
    )

    # 14) Symmetry Sanctum with Manhattan distance weighting
    env14 = clone(machina_symmetry_sanctum(num_cogs=4))
    qr14 = get_child_params(env14, "QuadrantResources")
    if qr14 is not None:
        qr14.distance_metric = "manhattan"
    sims.append(
        SimulationConfig(
            suite="cogs_sanctum", name="sym_sanctum_manhattan_resources", env=env14
        )
    )

    # 15) Symmetry Sanctum with path-aware traversal distance weighting
    env15 = clone(machina_symmetry_sanctum(num_cogs=4))
    qr15 = get_child_params(env15, "QuadrantResources")
    if qr15 is not None:
        qr15.distance_metric = "traversal"
    sims.append(
        SimulationConfig(
            suite="cogs_sanctum", name="sym_sanctum_traversal_resources", env=env15
        )
    )

    return sims
