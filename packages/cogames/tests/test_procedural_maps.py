import math

import pytest

from cogames.cogs_vs_clips.procedural import make_machina_procedural_map_builder
from mettagrid.mapgen.mapgen import MapGen
from mettagrid.mapgen.scenes.base_hub import DEFAULT_CORNER_CHESTS, DEFAULT_EXTRACTORS, BaseHub


def _collect_types(tree: dict) -> list[str]:
    types: list[str] = []
    node_type = tree.get("config", {}).get("type")
    if node_type:
        types.append(str(node_type))
    for child in tree.get("children", []):
        types.extend(_collect_types(child))
    return types


def _find_nodes(tree: dict, type_suffix: str) -> list[dict]:
    matches: list[dict] = []
    cfg = tree.get("config", {})
    t = cfg.get("type", "")
    if isinstance(t, str) and t.endswith(type_suffix):
        matches.append(tree)
    for child in tree.get("children", []):
        matches.extend(_find_nodes(child, type_suffix))
    return matches


def _extract_base_hub_config(tree: dict) -> dict:
    nodes = _find_nodes(tree, "BaseHub")
    assert nodes, "BaseHub should be present"
    return nodes[0]["config"]


@pytest.mark.parametrize(
    "width,height,density_scale,max_biome_frac,max_dungeon_frac",
    [
        (40, 40, 0.2, 0.30, 0.20),
        (60, 60, 0.4, 0.25, 0.25),
        (100, 100, 0.2, 0.20, 0.12),
    ],
)
def test_procedural_builder_builds_and_has_expected_layers(
    width: int, height: int, density_scale: float, max_biome_frac: float, max_dungeon_frac: float
):
    for seed in [0, 1, 2]:
        cfg = make_machina_procedural_map_builder(
            num_cogs=3,
            width=width,
            height=height,
            seed=seed,
            base_biome="caves",
            extractor_coverage=0.01,
            extractors={"chest": 1.0, "charger": 0.5},
            biome_weights={"caves": 0.5, "forest": 0.5, "city": 0.5, "desert": 0.5},
            dungeon_weights={"bsp": 0.6, "maze": 0.2, "radial": 0.2},
            density_scale=density_scale,
            max_biome_zone_fraction=max_biome_frac,
            max_dungeon_zone_fraction=max_dungeon_frac,
        )
        builder = cfg.create()
        game_map = builder.build()
        assert game_map.grid.size > 0

        # Scene tree structure checks
        tree = builder.get_scene_tree()
        types = _collect_types(tree)
        # Expect BSPLayout nodes for biome/dungeon when weights provided
        assert any(t.endswith("BSPLayout") for t in types)
        # BoundedLayout must wrap biome/dungeon zones
        assert any(t.endswith("BoundedLayout") for t in types)
        # UniformExtractorScene should be present
        assert any(t.endswith("UniformExtractorScene") for t in types)
        # BaseHub should be present
        assert any(t.endswith("BaseHub") for t in types)


@pytest.mark.parametrize(
    "width,height,max_biome_frac,max_dungeon_frac",
    [
        (100, 100, 0.30, 0.20),
        (80, 60, 0.25, 0.25),
    ],
)
def test_zone_counts_respect_max_zone_fraction(width: int, height: int, max_biome_frac: float, max_dungeon_frac: float):
    cfg = make_machina_procedural_map_builder(
        num_cogs=2,
        width=width,
        height=height,
        base_biome="caves",
        biome_weights={"caves": 1.0, "forest": 1.0},
        dungeon_weights={"bsp": 1.0, "maze": 1.0},
        max_biome_zone_fraction=max_biome_frac,
        max_dungeon_zone_fraction=max_dungeon_frac,
    )

    builder = cfg.create()
    builder.build()
    tree = builder.get_scene_tree()

    biome_layouts = _find_nodes(tree, "BSPLayout")
    assert biome_layouts, "Expected BSPLayout for zones"

    # find the one that wraps biome or dungeon children via BoundedLayout tag fields
    # area_count is directly on BSPLayout config
    def _min_count_for_fraction(frac: float) -> int:
        if frac <= 0:
            return 1
        return int(math.ceil(1.0 / min(0.9, max(0.02, float(frac)))))

    min_biomes = _min_count_for_fraction(max_biome_frac)
    min_dungeons = _min_count_for_fraction(max_dungeon_frac)

    # Extract all area_count values from BSPLayout nodes
    area_counts = [n["config"].get("area_count", 0) for n in biome_layouts]
    assert any(c >= min_biomes for c in area_counts)
    assert any(c >= min_dungeons for c in area_counts)


def test_uniform_extractors_configuration_pass_through():
    extractors = {"chest": 1.0, "charger": 0.3, "carbon_extractor": 0.7}
    cfg = make_machina_procedural_map_builder(
        num_cogs=2,
        seed=123,
        extractors=extractors,
        extractor_coverage=0.0125,
    )
    builder = cfg.create()
    builder.build()
    tree = builder.get_scene_tree()
    nodes = _find_nodes(tree, "UniformExtractorScene")
    assert nodes, "UniformExtractorScene should be present"
    ucfg = nodes[0]["config"]
    assert ucfg.get("extractor_names") == list(extractors.keys())
    assert ucfg.get("extractor_weights") == extractors
    assert pytest.approx(ucfg.get("target_coverage", 0.0), rel=1e-6) == 0.0125


CORNER_COORDS = [(2, 2), (18, 2), (2, 18), (18, 18)]
CROSS_COORDS = [(10, 6), (14, 10), (10, 14), (6, 10)]


def _build_base_hub_only(*, corner_bundle: str | None, cross_bundle: str | None, cross_distance: int = 4):
    cfg = MapGen.Config(
        width=21,
        height=21,
        instance=BaseHub.Config(
            spawn_count=0,
            include_inner_wall=False,
            corner_bundle=corner_bundle or "chests",
            cross_bundle=cross_bundle or "none",
            cross_distance=cross_distance,
        ),
    )
    builder = cfg.create()
    game_map = builder.build()
    return game_map.grid


@pytest.mark.parametrize(
    "corner_bundle,cross_bundle,expected_corner,expected_cross",
    [
        ("chests", "none", DEFAULT_CORNER_CHESTS, ("empty",) * 4),
        ("extractors", "none", DEFAULT_EXTRACTORS, ("empty",) * 4),
        ("chests", "extractors", DEFAULT_CORNER_CHESTS, DEFAULT_EXTRACTORS),
    ],
)
def test_base_hub_grid_matches_bundles(
    corner_bundle: str,
    cross_bundle: str,
    expected_corner: tuple[str, str, str, str],
    expected_cross: tuple[str, str, str, str],
):
    grid = _build_base_hub_only(corner_bundle=corner_bundle, cross_bundle=cross_bundle)

    for (x, y), name in zip(CORNER_COORDS, expected_corner, strict=False):
        assert grid[y, x] == name, (
            f"Expected corner object {name} at {(x, y)}, found {grid[y, x]!r} for bundle {corner_bundle}"
        )

    for (x, y), name in zip(CROSS_COORDS, expected_cross, strict=False):
        value = grid[y, x]
        if name == "empty":
            assert value == "empty", f"Expected empty cross tile at {(x, y)}, found {value!r}"
        else:
            assert value == name, f"Expected cross object {name} at {(x, y)}, found {value!r}"


def test_procedural_builder_deterministic_with_seed():
    cfg1 = make_machina_procedural_map_builder(num_cogs=2, width=50, height=50, seed=42)
    cfg2 = make_machina_procedural_map_builder(num_cogs=2, width=50, height=50, seed=42)

    b1 = cfg1.create()
    b2 = cfg2.create()
    m1 = b1.build()
    m2 = b2.build()

    # Deterministic grid given the same seed and dimensions
    assert (m1.grid == m2.grid).all()
