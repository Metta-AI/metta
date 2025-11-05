## Cogs vs Clips: Procedural Map Generation, Variants, and Missions

This guide explains how the procedural map system is wired together and how to extend it safely.

---

### Core Modules

- `cogs_vs_clips/procedural.py`
  - MachinaArena scene (`MachinaArenaConfig`) and `make_hub_only_map_builder`
  - Runtime override helpers (`apply_hub_overrides_to_builder`, `apply_procedural_overrides_to_builder`)
- `mettagrid/mapgen/scenes/building_distributions.py`
  - `UniformExtractorScene` and `DistributionConfig` for building placement
- `cogs_vs_clips/mission.py`
  - Base `Site`, `Mission`, `MissionVariant` types and the instantiate → make_env flow
- `cogs_vs_clips/missions.py`
  - Catalog of sites, missions, and variants; examples of procedural overrides and hub tuning
- `cogs_vs_clips/cli/mission.py`
  - CLI glue (`cogames play`, `cogames missions`, `cogames train`) and variant composition

Everything ultimately produces a `MapBuilderConfig` that feeds into a `MettaGridConfig`. Missions coordinate map
building, agent setup, and post-processing such as assembler rewrites.

---

### Procedural Composition

#### `MachinaArena` (Scene)

Asteroid arena built as a Scene graph: base-biome shell, optional biome/dungeon overlays, resource placement,
connectivity, and a central hub.

Config fields (all keyword-only):

| Category                | Parameters                                                                                      |
| ----------------------- | ----------------------------------------------------------------------------------------------- |
| Size (MapGen)           | `width`, `height`                                                                               |
| Randomness (MapGen)     | `seed`                                                                                          |
| Base biome              | `base_biome` (`"caves"`, `"forest"`, `"desert"`, `"city"`), `base_biome_config`                 |
| Biome overlays          | `biome_weights`, `biome_count`, `density_scale`, `max_biome_zone_fraction`                      |
| Dungeon overlays        | `dungeon_weights`, `dungeon_count`, `max_dungeon_zone_fraction`                                 |
| Buildings               | `building_names`, `building_weights`, `building_coverage`                                       |
| Placement distributions | `distribution` (global `DistributionConfig`), `building_distributions` (per-building overrides) |
| Hub layout              | `hub_corner_bundle`, `hub_cross_bundle`, `hub_cross_distance`                                   |

Important details:

- Building parameters are expressed in “buildings” (stations). Legacy extractor fields are not accepted.
- The top-level `MapGen.Config` (not the scene) carries `seed`. Scenes inherit RNGs spawned from the root, so the same
  seed reproduces terrain and placement exactly.
- Hub defaults place chests in the corners, but variants commonly override bundles and cross spacing.

#### Example: site-level builder

```python
from cogames.cogs_vs_clips.procedural import MachinaArenaConfig
from mettagrid.mapgen.mapgen import MapGen

MACHINA_PROCEDURAL_200 = Site(
    name="machina_procedural_200",
    description="Large procedural arena",
    map_builder=MapGen.Config(
        width=200,
        height=200,
        seed=12345,
        instance=MachinaArenaConfig(
            spawn_count=4,
            base_biome="caves",
            hub_corner_bundle="chests",
            hub_cross_bundle="extractors",
            hub_cross_distance=7,
            building_names=[
                "chest",
                "charger",
                "carbon_extractor",
                "oxygen_extractor",
                "germanium_extractor",
                "silicon_extractor",
            ],
            building_weights={
                "chest": 0.2,
                "charger": 0.6,
                "carbon_extractor": 0.3,
                "oxygen_extractor": 0.3,
                "germanium_extractor": 0.3,
                "silicon_extractor": 0.3,
            },
            building_coverage=0.01,
            distribution={"type": "bimodal", "cluster_std": 0.15},
            building_distributions={
                "chest": {"type": "exponential", "decay_rate": 5.0, "origin_x": 0.0, "origin_y": 0.0},
                "charger": {"type": "poisson"},
            },
        ),
    ),
    min_cogs=1,
    max_cogs=20,
)
```

#### `make_hub_only_map_builder`

Produces a hub-only `MapGen.Config` (typically 21×21) used for the training facility or as a base for variants.
Parameters include `seed`, `corner_bundle`, `cross_bundle`, `cross_distance`, and a transform set that rotates/flips the
hub.

---

### Placement Distributions (`building_distributions.py`)

`UniformExtractorScene` (configured via `UniformExtractorParams`) handles actual placement of buildings. It works in two
modes:

1. **Coverage-driven**: If `target_coverage` is set, it samples enough center points to hit the requested coverage using
   the supplied distributions.
2. **Grid-driven**: Without coverage, it places objects on a jittered grid defined by `rows`, `cols`, and `padding`.

Every random sample uses `self.rng`, which comes from the parent scene. Because `SceneConfig.seed` defaults to the
parent’s RNG when unset, the top-level `MapGen.Config(seed=...)` ensures deterministic results.

`DistributionConfig` supports:

- `type`: `"uniform"`, `"normal"`, `"exponential"`, `"poisson"`, `"bimodal"`
- Additional parameters per distribution (means, standard deviations, decay rates, cluster centers, etc.)

Per-building overrides live in `building_distributions` and accept the same schema. Omitted buildings fall back to the
global `distribution`.

---

### Missions and Variants

#### Sites (`missions.py`)

Sites describe reusable environments. They point to either a static map (`get_map("name.map")`) or a procedural builder.
Examples:

- `TRAINING_FACILITY`: hub-only builder, 21×21
- `HELLO_WORLD`: 100×100 procedural arena
- `MACHINA_1`: 200×200 procedural arena
- `MACHINA_PROCEDURAL`: shared base builder reused by procedural missions

Each site defines `min_cogs`/`max_cogs`. CLI calls (`--cogs`) override the default during mission instantiation.

#### Mission lifecycle (`mission.py`)

1. `Mission.configure()` (optional) tweaks defaults before instantiation.
2. `Mission.instantiate(map_builder, num_cogs, variant)`
   - Clones the mission, runs `configure()`, applies the selected variant, and sets `map` + `num_cogs`.
   - Applies procedural overrides to the builder (`apply_procedural_overrides_to_builder`). Overrides can include seeds,
     building distributions, hub bundles, etc.
3. `Mission.make_env()` finalizes the `MettaGridConfig` and allows post-processing (e.g., assembler recipe adjustments).

`ProceduralMissionBase` rebuilds the map after variant application, ensuring mission-specific `procedural_overrides`
drive a fresh call to `make_machina_procedural_map_builder`.

#### Procedural overrides cheat sheet

`Mission.procedural_overrides` is a plain `dict[str, Any]`. After variants run, we build a new `MapGen.Config` with
`instance=MachinaArenaConfig(**overrides)`. Width/height/seed are applied at the `MapGen.Config` level. Typical keys
include:

```python
self.procedural_overrides = {
    # Map size + randomness
    "width": 120,
    "height": 120,
    # normally this is set by the CLI --seed flag
    "seed": 24601,

    # Biome / dungeon structure
    "base_biome": "forest",
    "biome_weights": {"forest": 1.0, "caves": 0.25},
    "biome_count": 6,
    "dungeon_weights": {"maze": 0.75, "radial": 0.5},
    "density_scale": 1.2,

    # Resource placement (building-based API)
    # Defines the set of buildings that can be placed on the map
    "building_names": [
        "chest",
        "charger",
        "carbon_extractor",
        "oxygen_extractor",
        "germanium_extractor",
        "silicon_extractor",
    ],
    # What proportion of buildings are of a type, falls back to default if not set
    # If building_names is not set, this is used to determine the buildings
    "building_weights": {
        "chest": 1.0,
        "charger": 0.7,
        "carbon_extractor": 0.4,
        "oxygen_extractor": 0.4,
        "germanium_extractor": 0.4,
        "silicon_extractor": 0.4,
    },
    # How much of the map is covered by buildings
    "building_coverage": 0.012,
    # How buildings are distributed on the map
    "distribution": {"type": "exponential", "decay_rate": 4.5, "origin_x": 0.0, "origin_y": 1.0},
    # How buildings are distributed on the map per building type, falls back to global distribution if not set
    "building_distributions": {
        "chest": {"type": "normal", "mean_x": 0.5, "mean_y": 0.65, "std_x": 0.12, "std_y": 0.12},
        "charger": {"type": "poisson"},
    },

    # Hub controls
    "hub_corner_bundle": "chests",
    "hub_cross_bundle": "extractors",
    "hub_cross_distance": 7,
}
```

You can add or remove keys as needed—anything not provided falls back to `MachinaArenaConfig` defaults. Because
overrides accept `seed`, variants or missions can pin a layout for reproducibility.

---

#### Variants

Variants inherit from `MissionVariant` and override `apply(self, mission)`. Common patterns:

- Update resource/tuning parameters:

  ```python
  class MinedOutVariant(MissionVariant):
      name = "mined_out"
      description = "Some resources are depleted."

      def apply(self, mission: Mission) -> Mission:
          mission.carbon_extractor.efficiency -= 50
          mission.oxygen_extractor.efficiency -= 50
          mission.germanium_extractor.efficiency -= 50
          mission.silicon_extractor.efficiency -= 50
          return mission
  ```

- Switch biomes or hub contents via `procedural_overrides`:

  ```python
  class CityBiomeVariant(MissionVariant):
      name = "city"
      description = "Ancient city ruins provide structured pathways."

      def apply(self, mission: Mission) -> Mission:
          mission.procedural_overrides.update(
              {
                  "base_biome": "city",
                  "biome_weights": {"city": 1.0, "caves": 0.0, "desert": 0.0, "forest": 0.0},
                  "density_scale": 1.0,
                  "biome_count": 1,
                  "max_biome_zone_fraction": 0.95,
                  "dungeon_weights": {"bsp": 0.0, "maze": 0.0, "radial": 0.0},
                  "max_dungeon_zone_fraction": 0.0,
              }
          )
          return mission
  ```

- Adjust hub bundles:

  ```python
  class BothBaseVariant(MissionVariant):
      name = "both_base"
      description = "Chests on corners, extractors on cross arms."

      def apply(self, mission: Mission) -> Mission:
          mission.procedural_overrides.update(
              {
                  "hub_corner_bundle": "chests",
                  "hub_cross_bundle": "extractors",
                  "hub_cross_distance": 7,
              }
          )
          return mission
  ```

CLI variants are composed in order, so `cogames play -m machina_procedural.open_world -v city -v both_base` applies
`city`, then `both_base`.

---

### Seeds and Reproducibility

- Passing `seed` into a procedural builder guarantees deterministic terrain and building placement.
- Variants or missions can inject `procedural_overrides["seed"]` so the same CLI command reproduces runs.
- Hub-only builders preserve the original seed unless an override provides a new one (`apply_hub_overrides_to_builder`).
- CLI `--seed` flags (e.g., `cogames train --seed`) currently only seed the RL training loop; they do **not** inject a
  procedural seed. Add a mission/variant override if you need deterministic maps from the CLI today.

Example override from a mission:

```python
self.procedural_overrides = {
    "seed": 9876,
    "building_distributions": {
        "chest": {"type": "normal", "mean_x": 0.5, "mean_y": 0.5, "std_x": 0.15, "std_y": 0.15},
    },
}
```

---

### Building New Missions

1. **Define or reuse a Site** with the desired map builder.
2. **Create a Mission subclass**:
   - Set `site` to the site you want
   - Override `configure()` to set mission defaults (e.g., procedural overrides, reward tuning)
   - Optionally override `make_env()` for post-processing (e.g., seed chests, adjust assembler recipes)
3. **Add the mission class to `MISSIONS`** so the CLI picks it up.
4. (Optional) **Create variants** for common modifiers and append them to `VARIANTS`.

`ProceduralMissionBase` rebuilds a procedural map by creating a new `MapGen.Config` with `MachinaArenaConfig` from
`procedural_overrides`.

---

### CLI Reference

- List missions/variants:

  ```bash
  cogames missions
  ```

- Play with variants and overrides:

  ```bash
  cogames play --mission machina_procedural.open_world \
               --variant city \
               --variant both_base \
               --cogs 8 \
               --policy random
  ```

- Train on one or more missions (default policy `lstm`):

  ```bash
  # Single mission
  uv run cogames train -m machina_1.open_world --steps 200000 --seed 12345

  # Multiple missions
  uv run cogames train \
      -m machina_1.open_world \
      -m machina_procedural.explore \
      -m training_facility.harvest \
      --steps 200000 \
      --seed 12345
  ```

- Reproduce a procedural seed:
  ```bash
  cogames play -m machina_procedural.open_world --variant city --seed 24601
  ```
  (Assuming the mission/variant combination sets or respects `procedural_overrides["seed"]`.)

---

### Recommended Workflow

1. Start with an existing Site + Mission pair (e.g., `HELLO_WORLD`, `ExploreMission`).
2. Copy the mission, adjust `procedural_overrides` or hub settings, and add it to `MISSIONS`.
3. Define variants for reusable tweaks.
4. Use CLI commands (`missions`, `play`, `train`) to iterate quickly.
5. When training, consider reducing `--parallel-envs`/`--num-workers` on macOS to avoid long startup times while
   generating large maps.

This structure keeps procedural content declarative, supports deterministic reproduction, and allows incremental
extension without touching engine internals.
