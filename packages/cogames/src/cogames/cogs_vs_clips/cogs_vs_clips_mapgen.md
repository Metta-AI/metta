## Cogs vs Clips: Procedural Map Generation, Variants, and Missions

This guide explains how the procedural map system is wired together and how to extend it safely.

---

### Core Modules

- `cogs_vs_clips/procedural.py`
  - `MachinaArena` scene
  - Variant helpers (`ProceduralOverridesVariant`, `apply_procedural_overrides`, `apply_base_hub_overrides`)
- `mettagrid/mapgen/scenes/building_distributions.py`
  - `UniformExtractorScene` and `DistributionConfig` for building placement
- `cogs_vs_clips/mission.py`
  - Base `Site`, `Mission`, `MissionVariant` types
- `cogs_vs_clips/sites.py`
  - Catalog of sites
- `cogs_vs_clips/variants.py`
  - Catalog of variants
- `cogs_vs_clips/missions.py`
  - Catalog of missions, mostly composed from variants
- `cogs_vs_clips/cli/mission.py`
  - CLI glue (`cogames play`, `cogames missions`, `cogames train`) and variant composition

Everything ultimately produces a `MapBuilderConfig` that feeds into a `MettaGridConfig`. Missions and variants
coordinate map building, agent setup, and post-processing such as assembler rewrites.

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
| Base biome              | `base_biome` (`"caves"`, `"forest"`, `"desert"`, `"city"`, `"plains"`), `base_biome_config`     |
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
from cogames.cogs_vs_clips.procedural import MachinaArena
from mettagrid.mapgen.mapgen import MapGen

MACHINA_PROCEDURAL_200 = Site(
    name="machina_procedural_200",
    description="Large procedural arena",
    map_builder=MapGen.Config(
        width=200,
        height=200,
        seed=12345,
        instance=MachinaArena.Config(
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

#### Sites (`sites.py`)

Sites describe reusable environments. They point to either a static map (`get_map("name.map")`) or a procedural builder.
Examples:

- `TRAINING_FACILITY`: hub-only builder, 21×21
- `HELLO_WORLD`: 100×100 procedural arena
- `MACHINA_1`: 200×200 procedural arena

Each site defines `min_cogs`/`max_cogs`. CLI calls (`--cogs`) override the default during mission instantiation.

#### Mission lifecycle (`mission.py`)

1. `mission.with_variants(variants_list)` (optional) clones the mission and attach variants to it
   - `variant.modify_mission()` is applied immediately
   - `variant.modify_env()` is applied when `make_env` is called
   -
2. `mission.make_env()` finalizes the `MettaGridConfig` and applies variants to the environment

#### Variants

Variants inherit from `MissionVariant` and override either `modify_mission(self, mission)`,
`modify_env(self, mission, env)`, or both.

Common patterns:

- Update resource/tuning parameters:

  ```python
  class MinedOutVariant(MissionVariant):
      name: str = "mined_out"
      description: str = "Some resources are depleted."

      def modify_mission(self, mission: Mission):
          mission.carbon_extractor.efficiency -= 50
          mission.oxygen_extractor.efficiency -= 50
          mission.germanium_extractor.efficiency -= 50
          mission.silicon_extractor.efficiency -= 50
  ```

- Switch biomes or hub contents:

  ```python
  class CityVariant(ProceduralOverridesVariant):
      name: str = "city"
      description: str = "Ancient city ruins provide structured pathways."
      overrides: ProceduralOverrides = ProceduralOverrides(
          biome_weights={"city": 1.0, "caves": 0.0, "desert": 0.0, "forest": 0.0},
          base_biome="city",
          # Fill almost the entire map with the city layer
          density_scale=1.0,
          biome_count=1,
          max_biome_zone_fraction=0.95,
          # Tighten the city grid itself
      )
  ```

- Adjust hub bundles:

  ```python
  class BothBaseVariant(MissionVariant):
      name: str = "both_base"
      description: str = "Chests on corners, extractors on cross arms."

      def modify_mission(self, mission: Mission) -> Mission:
          mission.procedural_overrides.update(
              {
                  "hub_corner_bundle": "chests",
                  "hub_cross_bundle": "extractors",
                  "hub_cross_distance": 7,
              }
          )
  ```

- Adjust any env properties:

  ```python
  class MyVariant(MissionVariant):
      name: str = "my"
      def modify_env(self, mission: Mission, env: MettaGridConfig) -> None:
          env.game.charger.efficiency -= 50
  ```

CLI variants are composed in order, so `cogames play -m machina_procedural.open_world -v city -v both_base` applies
`city`, then `both_base`.

---

### Seeds and Reproducibility

- `MapGen.Config.seed` (`env.game.map_builder.seed`) controls **map layout**: set it for deterministic terrain/building
  placement for a mission/site.
- `cogames evaluate` and `cogames play` use `--seed` for the simulator/policy RNG and `--map-seed` (or `--seed` if
  omitted) for `MapGenConfig.seed`, so runs can be made fully reproducible from the CLI.
- `cogames train` treats `--map-seed` as an **opt-in** override: when set, all procedural training missions use that
  fixed `MapGenConfig.seed`; when left `None`, the vectorized env factory derives per-env map seeds from the runner’s
  RNG so a fixed `--seed` yields a reproducible but diverse map sequence.

Example programmatic override using the shared `MapSeedVariant` helper:

```python
from cogames.cogs_vs_clips.procedural import MapSeedVariant

base_mission = HelloWorldOpenWorldMission
seeded_mission = base_mission.with_variants([MapSeedVariant(seed=1234)])
env_cfg = seeded_mission.make_env()
# env_cfg.game.map_builder is a MapGen.Config with seed=1234; calling builder.build()
# will now deterministically reproduce the same grid.
```

---

### Building New Missions

1. **Define or reuse a Site** with the desired map builder.
2. **Define the desired behavior in a variant**
3. **Create a Mission object**:

```python
mission = Mission(
  name="my_mission",
  description="My mission",
  site=site,
  variants=[Variant1(), Variant2()],
)
```

4. **Add the mission object to `MISSIONS`** so the CLI picks it up.

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
2. Copy the mission, adjust its properties, and add it to `MISSIONS`.
3. Define variants for reusable tweaks.
4. Use CLI commands (`missions`, `play`, `train`) to iterate quickly.
5. When training, consider reducing `--parallel-envs`/`--num-workers` on macOS to avoid long startup times while
   generating large maps.

---

### Curated Integrated Evals (Scorable Baselines)

We provide a small integrated evaluation set (see `cogs_vs_clips/evals/integrated_eval.py`) tuned to yield non-zero
scores for baseline agents while leaving headroom for improvement. These are composed from procedural `HELLO_WORLD` maps
with variants that balance approachability and challenge.

Key design choices:

- Pack the base hub lightly (`EmptyBaseVariant`) where appropriate to encourage early exploration without
  over-constraining.
- Add guidance (`CompassVariant`) on distance-heavy tasks to reduce pure exploration failure modes.
- Raise agent caps modestly (`PackRatVariant`) to avoid early inventory stalls but keep routing relevant.
- Shape reward on vibe missions (`HeartChorusVariant`) so partial progress is scored.
- Keep vibe mechanics intact unless the mission explicitly focuses on vibe manipulation.

Included missions and variants:

- oxygen_bottleneck: `EmptyBaseVariant(missing=["oxygen_extractor"])`, `ResourceBottleneckVariant(["oxygen"])`,
  `SingleResourceUniformVariant("oxygen_extractor")`, `PackRatVariant`
- energy_starved: `EmptyBaseVariant`, `DarkSideVariant`, `PackRatVariant`
- distant_resources: `EmptyBaseVariant`, `CompassVariant`, `DistantResourcesVariant`
- quadrant_buildings: `EmptyBaseVariant`, `QuadrantBuildingsVariant`, `CompassVariant`
- single_use_swarm: `EmptyBaseVariant`, `SingleUseSwarmVariant`, `CompassVariant`, `PackRatVariant`
- vibe_check: `HeartChorusVariant`, `VibeCheckMin2Variant`

Usage example:

```bash
uv run python packages/cogames/scripts/run_evaluation.py \
  --agent cogames.policy.nim_agents.agents.ThinkyAgentsMultiPolicy \
  --mission-set integrated_evals \
  --cogs 4 \
  --repeats 2
```

Recommendation: When designing new scorable baselines, combine one “shaping” variant (e.g., `CompassVariant`,
`HeartChorusVariant`, `PackRatVariant`) with one “constraint” variant (e.g., `DarkSideVariant`,
`ResourceBottleneckVariant`, `SingleUseSwarmVariant`) to keep tasks legible yet challenging.
