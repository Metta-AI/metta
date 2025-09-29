## Sanctum Quadrants Map Generator

This document describes the “Sanctum-in-the-Quadrants” map generator and its key components, parameters, and usage
across CLI and Gridworks.

### Overview

Pipeline (high-level):

1. Split full inner map into four quadrants (no cross leftovers)
2. Populate terrain per quadrant from a library (BSP/Maze/etc.)
3. Optional symmetry pass (horizontal/vertical/both)
4. Place placeholder converters symmetrically; mirror if desired
5. Ensure connectivity (dig tunnels) across inner map
6. Stamp the central Sanctum/Base (overrides terrain)
7. Relabel converter types to achieve target mix and totals
8. Balance converter distances from the altar (optional moves/carves)

### Key Scenes and Parameters

- Quadrants (splits map and defines `base` and quadrant tags)
  - `QuadrantsParams`
    - `base_size` (int): size of the central sanctum area stamped later

- QuadrantLayout (per-quadrant terrain picker)
  - `QuadrantLayoutParams`
    - `weight_bsp10`, `weight_bsp8`, `weight_maze`, `weight_terrain_maze` (floats): weighted mix of scenes per quadrant

- EnforceSymmetry (in-place mirroring)
  - `EnforceSymmetryParams`
    - `horizontal` (bool): mirror top→bottom
    - `vertical` (bool): mirror left→right

- QuadrantResources (per-quadrant converter placement)
  - `QuadrantResourcesParams`
    - `resource_types` (list[str]): allowed converter types
    - `forced_type` (str|None): pin a specific type in this instance
    - `count_per_quadrant` (int): how many to place per quadrant
    - Distribution controls (forwarded to `RadialObjects`):
      - `mode` (str): one of `power`, `exp`, `log`, `gaussian`
      - `k`, `alpha`, `beta`, `mu`, `sigma`: parameters for the chosen mode
      - `distance_metric` (str): one of `euclidean`, `manhattan`, `traversal`
        - `euclidean`: radial distance √((x−cx)^2+(y−cy)^2)
        - `manhattan`: L1 distance |x−cx|+|y−cy|
        - `traversal`: shortest path in the current terrain (BFS over "empty" cells) from the center
      - `min_radius` (int|None): exclude inner ring (interpreted under the chosen metric)
      - `clearance` (int): empty buffer around each placement

- RadialObjects (radial placement with multiple distributions)
  - `RadialObjectsParams`
    - `objects` (dict[str,int]): {type → count}
    - `mode` (str): one of `power`, `exp`, `log`, `gaussian`
      - `power`: weight ∝ (r/rmax)^`k`
      - `exp`: weight ∝ exp(`alpha`\*(r/rmax−1))
      - `log`: weight ∝ log(1 + `beta`\*r)
      - `gaussian`: weight ∝ exp(−0.5\*((r/rmax − `mu`)/`sigma`)^2)
    - `k` (float): power exponent
    - `alpha` (float): exponential growth factor
    - `beta` (float): logarithmic growth factor
    - `mu`, `sigma` (floats): gaussian center and spread (as fractions of rmax)
    - `distance_metric` (str): `euclidean` (default), `manhattan`, or `traversal` (BFS path distance through empties)
    - `min_radius` (int|None): exclude ring near center (interpreted under the chosen metric)
    - `clearance` (int): empty buffer around each placement
    - `carve` (bool): carve clearance area to guarantee space

- MakeConnected (global connectivity)
  - `MakeConnectedParams`: none (connects all components via shortest tunnels)

- BaseHub (central Sanctum stamp)
  - `BaseHubParams`
    - `altar_object` (str): center object name
    - `corner_generator` (str): corner station object
    - `spawn_symbol` (str): spawn tokens
    - `include_inner_wall` (bool): ring with 4 gates (mid-side)

- RelabelConverters (type balancing while preserving positions)
  - `RelabelConvertersParams`
    - `target_counts` (dict[str,int]): final totals per converter type
    - `source_types` (list[str]): which objects to consider for relabeling
    - `symmetry` ("none"|"horizontal"|"vertical"|"both")
    - `quadrant_types` (dict{"nw","ne","sw","se"}→str|None): when `symmetry="both"`, explicitly assign converter types
      per quadrant for candidate positions

- DistanceBalance (mean distance equalization)
  - `DistanceBalanceParams`
    - `converter_types` (list[str])
    - `tolerance` (float): allowed mean gap vs global mean
    - `relocate` (bool): move far-out instances toward target mean
    - `moves_per_type` (int), `relocation_clearance` (int), `relocation_min_radius` (int)
    - `balance` (bool): optional corridor carving
    - `carves_per_type` (int), `carve_width` (int)

### Deterministic Seeding

Top-level: `MapGen.Config(seed=SEED)` sets the global seed.

Children derive deterministic seeds in `Scene.render_with_children()` as:

`child_seed = parent_seed + 1000003*action_idx + 7919*area_idx + seed_offset`

- You can provide `seed_offset` on `ChildrenAction` to make 4 quadrants different but reproducible.

### Gridworks

Config Makers (discoverable in Gridworks):

- `experiments.recipes.sanctum.make_mettagrid()` → base Sanctum map
- `experiments.recipes.symmetry_sanctum.make_mettagrid()` → Sanctum with both-axis symmetry
- `experiments.recipes.sanctum.make_evals()` → list[SimulationConfig] of Sanctum variants (terrain mixes, resource
  distributions, symmetry, quadrant relabeling)

Use the Map Editor → Get Map to visualize and export.

### CLI

List games:

```bash
uv run python packages/cogames/src/cogames/main.py games
```

Play Sanctum:

```bash
uv run python packages/cogames/src/cogames/main.py play machina_sanctum --steps 200 --interactive 0
```

Deterministic map (seeded):

```bash
uv run python packages/cogames/src/cogames/main.py play machina_sanctum --steps 200 --interactive 0 --seed 123
```

### Common Recipes

- Symmetric terrain, non-symmetric converters:
  1. QuadrantLayout → EnforceSymmetry(horizontal/vertical) → QuadrantResources → MakeConnected → BaseHub

- Symmetric positions, rebalanced types:
  1. QuadrantLayout → EnforceSymmetry (terrain) → placeholder converters → EnforceSymmetry (resources) → MakeConnected →
     BaseHub → RelabelConverters (target mix)

- Distance balancing:
  - Enable `DistanceBalance` with `relocate=True` and set `tolerance`; increase `moves_per_type` for stronger
    equalization.

### Extending per-quadrant distributions

- Use `QuadrantResources` per quadrant with `RadialObjectsParams(mode=...)`:
  - Top-right geranium (exponential): `mode="exp"`, `alpha>0`
  - Bottom-left oxygen (linear-ish): `mode="power"`, `k≈1`
  - Outer-ring bias: `mode="gaussian"`, `mu≈0.9`, `sigma≈0.05`
  - Gentle increase: `mode="log"`, `beta>0`
  - Choose distance metric: `distance_metric="euclidean" | "manhattan" | "traversal"`

### Testing

- `tests/test_symmetry_sanctum.py` validates:
  - Minimum converter counts outside the base
  - Symmetry of converter placements (both axes) in the symmetry scenario
