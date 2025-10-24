## Cogs vs Clips: Procedural Map Generation and Missions

This guide shows how to build procedural maps, define Sites, create Missions and Variants, and run them via the CLI.

### Key files

- `packages/cogames/src/cogames/cogs_vs_clips/procedural.py` (procedural map builders and overrides)
- `packages/cogames/src/cogames/cogs_vs_clips/mission.py` (base `Site`, `Mission`, `MissionVariant` types and
  instantiation flow)
- `packages/cogames/src/cogames/cogs_vs_clips/missions.py` (catalog of Sites, Missions, and Variants)
- `packages/cogames/src/cogames/cli/mission.py` (CLI glue; `--mission`, `--variant`, `--cogs`)

### Concepts

- **Site**: A location with a `map_builder` and min/max cogs. A mission selects a Site.
- **Mission**: A template that becomes an instantiated mission with a concrete map and number of cogs.
- **Variant**: A small modifier that updates a mission (e.g., resource efficiencies, biome mix, hub contents).
- **Procedural builder**: Functions that return a `MapBuilderConfig` (e.g., `make_machina_procedural_map_builder`).

### Procedural builders

Use `make_machina_procedural_map_builder` for an asteroid arena with hub and resource pockets.

Parameters you can set:

- **Size**: `width`, `height`
- **Seed**: `seed`
- **Biomes**: `base_biome` in {`caves`,`forest`,`desert`,`city`}; `biome_weights`, `biome_count`, `density_scale`
- **Dungeons**: `dungeon_weights`, `dungeon_count`
- **Hub**: `hub_corner_bundle` in {`chests`,`extractors`,`none`}, `hub_cross_bundle` in {`chests`,`extractors`,`none`},
  `hub_cross_distance`
- **Resources**: `extractor_names`, `extractor_weights`, `extractor_coverage`

Example (as a Site map builder):

```python
from cogames.cogs_vs_clips.procedural import make_machina_procedural_map_builder

MACHINA_PROCEDURAL_200 = Site(
    name="machina_procedural_200",
    description="Large procedural map",
    map_builder=make_machina_procedural_map_builder(
        num_cogs=4, width=200, height=200, base_biome="caves",
        hub_corner_bundle="chests", hub_cross_bundle="extractors", hub_cross_distance=7,
        extractor_names=["chest","charger","carbon_extractor","oxygen_extractor"],
        extractor_weights={"chest": 0.2, "charger": 0.6, "carbon_extractor": 0.3, "oxygen_extractor": 0.3},
        extractor_coverage=0.01,
    ),
    min_cogs=1,
    max_cogs=20,
)
```

### Sites: fixed vs procedural

You can use either:

- A fixed map file via `get_map("name.map")`
- A procedural builder (recommended for flexibility)

See `HELLO_WORLD` and `MACHINA_1` in `missions.py` for procedural 100x100 and 200x200 examples.

### Variants (MissionVariant)

Variants are small modifiers applied during `Mission.instantiate`. They typically update `mission.procedural_overrides`
or station parameters.

Example variant that forces a city biome with dense coverage and no dungeons:

```python
class CityBiomeVariant(MissionVariant):
    name: str = "city"
    description: str = "Ancient city grid"

    def apply(self, mission: Mission) -> Mission:
        mission.procedural_overrides.update({
            "base_biome": "city",
            "biome_weights": {"city": 1.0, "caves": 0.0, "desert": 0.0, "forest": 0.0},
            "biome_count": 1,
            "max_biome_zone_fraction": 0.95,
            "dungeon_weights": {"bsp": 0.0, "maze": 0.0, "radial": 0.0},
            "max_dungeon_zone_fraction": 0.0,
        })
        return mission
```

Common hub variants set:

- `hub_corner_bundle`: `"chests" | "extractors" | "none"`
- `hub_cross_bundle`: `"chests" | "extractors" | "none"`
- `hub_cross_distance`: distance from center for cross placements

### How overrides are applied

`Mission.instantiate` (see `mission.py`) will:

1. Copy and configure the mission
2. Set `map` and `num_cogs`
3. Apply the selected Variant
4. Apply `procedural_overrides` to the builder when possible

For procedural missions that must rebuild the map builder, see `ProceduralMissionBase` in `missions.py` which
regenerates the builder after variants using accumulated overrides.

### Missions

Create a `Mission` subclass, pick a `site`, optionally set `procedural_overrides` in `configure()`, and customize in
`make_env()` after the map is built.

Example: chest-only explore procedural mission (from `missions.py`):

```python
class MachinaProceduralExploreMission(ProceduralMissionBase):
    name: str = "explore"
    description: str = "There are HEARTs scattered around the map. Collect them all."

    def configure(self):
        self.heart_capacity = 99
        self.procedural_overrides = {
            "extractor_names": ["chest"],
            "extractor_weights": {"chest": 1.0},
            "extractor_coverage": 0.004,
            "hub_corner_bundle": "chests",
            "hub_cross_bundle": "none",
            "hub_cross_distance": 7,
        }

    def make_env(self) -> MettaGridConfig:
        env = super().make_env()
        # Example: ensure chest template starts with a heart
        chest_cfg = env.game.objects.get("chest")
        if isinstance(chest_cfg, ChestConfig):
            chest_cfg.initial_inventory = 1
        return env
```

### Agent counts (cogs)

Agent count is set when calling `Mission.instantiate(map_builder, num_cogs, variant)`. The CLI passes `--cogs` into this
path.

If a mission needs a default (e.g., VibeCheck requires 4 agents) but should still respect `--cogs`, override
`instantiate` minimally:

```python
def instantiate(self, map_builder: MapBuilderConfig, num_cogs: int, variant: MissionVariant | None = None) -> "Mission":
    desired = 4 if (self.site and num_cogs == self.site.min_cogs) else num_cogs
    return super().instantiate(map_builder, desired, variant)
```

### Using the CLI

List missions and variants:

```bash
cogames missions
```

Run a mission:

```bash
# Site.Mission with variants and cogs
cogames play --mission machina_procedural.open_world --variant city --variant both_base --cogs 8

# Training facility explore
cogames play --mission hello_world.explore --cogs 4
```

Tips:

- Combine multiple `--variant` flags; they apply in order.
- If you omit `--cogs`, the CLI uses the Site’s `min_cogs`. Per-mission overrides (like the VibeCheck default to 4) can
  be implemented as above while still allowing `--cogs` to override.

### Common patterns

- Tight hub vs default hub: controlled by `BaseHub` config via `make_hub_only_map_builder` or via procedural hub
  bundles.
- Resource density: tune `extractor_coverage` and `extractor_weights`.
- Biome vs dungeon structure: set `biome_weights/dungeon_weights` and counts; use `density_scale` for autoscaling.

### Where to start

1. Duplicate an existing Site in `missions.py` and point it to a procedural builder with your dimensions.
2. Add a new `MissionVariant` to flip biomes/hub layout as needed.
3. Add a new `Mission` (optionally using `ProceduralMissionBase`) to set mission-specific overrides.
4. Run with the CLI using your `site.mission` and any `--variant`/`--cogs` you want.

That’s it—this flow lets you iterate quickly on arenas, content density, and mission rules without touching engine
internals.
