# CoGames Evaluation Missions

This directory contains two types of evaluation missions for testing CoGames agents:

1. **Diagnostic Missions** - Fixed-map missions testing specific skills in controlled environments
2. **Integrated Eval Missions** - Procedural missions combining multiple challenges for curriculum training

---

## Diagnostic Missions

**Location:** `diagnostic_evals.py` **Access:** `cogames play --mission evals.diagnostic_*` **Map Type:** Fixed ASCII
maps (deterministic layouts)

Diagnostic missions test specific skills in isolation with controlled, repeatable environments.

### Available Diagnostic Missions

#### Navigation & Delivery

- `diagnostic_chest_navigation1/2/3` - Navigate to chest and deposit hearts (varying difficulty)
- `diagnostic_chest_near` - Chest nearby, test deposit mechanics
- `diagnostic_chest_search` - Find chest through exploration
- `diagnostic_chest_deposit_near/search` - Combined navigation and deposit tests

#### Resource Extraction

- `diagnostic_extract_missing_carbon` - Extract carbon when it's missing from inventory
- `diagnostic_extract_missing_oxygen` - Extract oxygen when it's missing
- `diagnostic_extract_missing_germanium` - Extract germanium when it's missing
- `diagnostic_extract_missing_silicon` - Extract silicon when it's missing

#### Assembly

- `diagnostic_assembler_near` - Assemble hearts at nearby assembler
- `diagnostic_assembler_search` - Find assembler and craft hearts

#### Energy Management

- `diagnostic_charge_up` - Test charging mechanics and energy management

#### Unclipping

- `diagnostic_unclip_craft` - Craft unclip items to restore clipped extractors
- `diagnostic_unclip_preseed` - Unclip with pre-seeded inventory

#### Complex Scenarios

- `diagnostic_radial` - Radial resource layout with chorus assembly
- `diagnostic_agile` - Test agility and quick decision-making
- `diagnostic_memory` - Test memory and state tracking

#### Hard Variants

Most diagnostic missions have `_hard` variants with increased difficulty (e.g., `diagnostic_chest_navigation1_hard`,
`diagnostic_radial_hard`).

### Playing Diagnostic Missions

```bash
# Basic diagnostic
uv run cogames play --mission evals.diagnostic_chest_navigation1 --cogs 1

# Multi-agent
uv run cogames play --mission evals.diagnostic_extract_missing_oxygen --cogs 2

# Hard variant
uv run cogames play --mission evals.diagnostic_radial_hard --cogs 1

# With policy
uv run cogames play --mission evals.diagnostic_unclip_craft -p scripted_baseline --cogs 1
```

---

## Integrated Eval Missions

**Location:** `integrated_evals.py` **Access:** `cogames play --mission hello_world.*` **Map Type:** Procedural
generation (MachinaArena)

Integrated eval missions use procedural generation and combine multiple mission variants to create diverse training
scenarios. They use 50×50 to 150×150 procedural maps with randomized building placements.

### Available Integrated Missions

#### oxygen_bottleneck

**Challenge:** Oxygen is the limiting resource; agents must prioritize oxygen extraction.

**Variants Applied:**

- EmptyBase (missing oxygen_extractor initially)
- ResourceBottleneck (oxygen)
- SingleResourceUniform (oxygen_extractor)
- PackRat (increased inventory capacity)

```bash
uv run cogames play --mission hello_world.oxygen_bottleneck --cogs 2
```

#### energy_starved

**Challenge:** Low energy regen and weak chargers require careful energy management.

**Variants Applied:**

- EmptyBase
- DarkSide (reduced energy regen)
- PackRat

```bash
uv run cogames play --mission hello_world.energy_starved --cogs 2
```

#### distant_resources

**Challenge:** Resources scattered far from base; requires efficient routing.

**Variants Applied:**

- EmptyBase
- DistantResources

```bash
uv run cogames play --mission hello_world.distant_resources --cogs 4
```

#### quadrant_buildings

**Challenge:** Buildings placed in four quadrants; requires region partitioning.

**Variants Applied:**

- EmptyBase
- QuadrantBuildings

```bash
uv run cogames play --mission hello_world.quadrant_buildings --cogs 4
```

#### single_use_swarm

**Challenge:** All extractors are single-use; agents must fan out and coordinate.

**Variants Applied:**

- EmptyBase
- SingleUseSwarm
- PackRat

```bash
uv run cogames play --mission hello_world.single_use_swarm --cogs 4
```

#### vibe_check

**Challenge:** Vibe-based coordination and chorus assembly.

**Variants Applied:**

- EmptyBase
- HeartChorus
- VibeCheckMin2

```bash
uv run cogames play --mission hello_world.vibe_check --cogs 4
```

#### easy_hearts

**Challenge:** Simplified heart crafting with generous parameters.

**Variants Applied:**

- LonelyHeart
- HeartChorus
- PackRat

```bash
uv run cogames play --mission hello_world.easy_hearts --cogs 2
```

### Playing Integrated Missions with Additional Variants

You can apply additional variants on top of the mission's built-in variants:

```bash
# Add compass variant
uv run cogames play --mission hello_world.oxygen_bottleneck --cogs 2 --variant compass

# Add multiple variants
uv run cogames play --mission hello_world.energy_starved --cogs 2 --variant compass --variant small_50

# With policy
uv run cogames play --mission hello_world.single_use_swarm --cogs 4 -p scripted_baseline
```

---

## Programmatic Evaluation

### Using run_evaluation.py

For systematic evaluation across multiple missions and configurations:

```bash
# Evaluate on integrated eval suite
uv run python packages/cogames/scripts/run_evaluation.py \
  --agent cogames.policy.nim_agents.agents.ThinkyAgentsMultiPolicy \
  --mission-set integrated_evals \
  --cogs 4 \
  --repeats 2

# Evaluate specific agent
uv run python packages/cogames/scripts/run_evaluation.py \
  --agent simple \
  --steps 1000 \
  --output eval_simple.json
```

### Using in Curriculum Training

Both diagnostic and integrated missions can be used in curriculum training via `mission_variant_curriculum.py`:

```python
from recipes.experiment.cvc import mission_variant_curriculum

# Train on diagnostic missions
mission_variant_curriculum.train(
    base_missions=["diagnostic_missions"],
    num_cogs=4,
    variants="all"
)

# Train on specific integrated missions
mission_variant_curriculum.train(
    base_missions=["oxygen_bottleneck", "energy_starved"],
    num_cogs=4,
    variants=["compass", "pack_rat"]
)
```

---

## Design Philosophy

### Diagnostic Missions

- **Focused**: Each tests a specific skill in isolation
- **Deterministic**: Fixed maps ensure reproducible results
- **Minimal**: Small maps, simple layouts, clear objectives
- **Scalable**: Work well with 1-4 agents

### Integrated Missions

- **Comprehensive**: Combine multiple challenges and variants
- **Procedural**: Different map each run for generalization
- **Challenging**: Require coordination and strategic planning
- **Scalable**: Work well with 2-8 agents

### Evaluation Best Practices

1. Use diagnostic missions to identify specific skill deficits
2. Use integrated missions to evaluate overall performance
3. Run multiple seeds to account for procedural variation
4. Compare against scripted baselines for context

---

**Last Updated:** December 3, 2025

**Diagnostic Missions:** 30+ (various skills and hard variants) **Integrated Missions:** 7 (procedural, with built-in
variants)
