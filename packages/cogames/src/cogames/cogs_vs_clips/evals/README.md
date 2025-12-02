# CoGames Evaluation Environments

This directory contains standardized evaluation missions and difficulty variants for testing CoGames agents.

**Contents:**

- [Evaluation Missions](#evaluation-missions) - 14 distinct environments testing different skills
- [Difficulty Variants](#difficulty-variants) - 13 difficulty levels that can be applied to any mission
- [Quick Reference](#quick-reference) - Tables and usage examples

---

## Evaluation Missions

All evaluation missions are defined in `eval_missions.py` and registered under the `evals` site. Each mission has a
unique map, resource configuration, and gameplay challenge.

### Resource Collection Missions

#### CollectResourcesClassic

**Map:** `evals/eval_collect_resources.map`

**Challenge:** Classic balanced layout with extractors near base

**Configuration:**

- Energy regen: 2/step
- Charger efficiency: 130
- Resource efficiency: Carbon 125, Oxygen 115, Germanium 90, Silicon 125
- Max uses: Unlimited for all extractors
- **Best for:** Standard resource collection testing
- **Recommended agents:** 1-4

**Skills tested:** Efficient routing, resource prioritization, basic coordination

---

#### CollectResourcesSpread

**Map:** `evals/eval_collect_resources_medium.map`

**Challenge:** Scattered extractors requiring more exploration and routing

**Configuration:**

- Energy regen: 2/step
- Charger efficiency: 135
- Resource efficiency: Carbon 130, Oxygen 120, Germanium 95, Silicon 130
- Max uses: Unlimited for all extractors
- **Best for:** Testing exploration and medium-distance routing
- **Recommended agents:** 1-4

**Skills tested:** Exploration, longer-distance routing, rally/chorus glyphing

---

#### CollectFar

**Map:** `evals/eval_collect_resources_hard.map`

**Challenge:** Resources scattered far from base; heavy routing coordination

**Configuration:**

- Energy regen: 2/step
- Charger efficiency: 135
- Resource efficiency: Carbon 130, Oxygen 120, Germanium 100, Silicon 135
- Max uses: Carbon 40, Oxygen 30, Germanium 20, Silicon 25 (limited!)
- **Best for:** Testing long-distance coordination and resource scarcity
- **Recommended agents:** 2-8

**Skills tested:** Long-distance navigation, route coordination, extractor awareness, multi-agent resource sharing

---

### Specialization & Coordination Missions

#### DivideAndConquer

**Map:** `evals/eval_divide_and_conquer.map`

**Challenge:** Resources partitioned into separate regions; agents must specialize and reconvene

**Configuration:**

- Energy regen: 2/step
- Charger efficiency: 130
- Resource efficiency: Carbon 125, Oxygen 120, Germanium 95, Silicon 130
- Max uses: Carbon 25, Oxygen 20, Germanium 10, Silicon 15
- **Best for:** Testing region partitioning and specialization
- **Recommended agents:** 2-8

**Skills tested:** Region partitioning, agent specialization, resource trading/coordination, assembly coordination

---

#### GoTogether

**Map:** `evals/eval_balanced_spread.map`

**Challenge:** Objects favor collective glyphing; agents must travel and return as a pack

**Configuration:**

- Energy regen: 2/step
- Charger efficiency: 140
- Resource efficiency: Carbon 130, Oxygen 125, Germanium 100, Silicon 135
- Max uses: Carbon 30, Oxygen 25, Germanium 10, Silicon 20
- Minimum agents: 2 (enforced)
- **Best for:** Testing synchronized movement and collective glyphing
- **Recommended agents:** 2-4

**Skills tested:** Synchronized travel, collective glyphing, rally coordination, pack behavior

---

#### SingleUseSwarm

**Map:** `evals/eval_single_use_world.map`

**Challenge:** Every extractor can be used exactly once; team must fan out and reconverge

**Configuration:**

- Energy regen: 2/step
- Charger efficiency: 140
- Resource efficiency: Carbon 130, Oxygen 125, Germanium 105, Silicon 135
- Max uses: Carbon 1, Oxygen 1, Germanium 1, Silicon 1 (single use!)
- Minimum agents: 2 (enforced)
- **Best for:** Testing extractor assignment and resource pooling
- **Recommended agents:** 4-8

**Skills tested:** Extractor reservation, resource pooling, assembly coordination, strategic fanning out

---

### Bottleneck & Constraint Missions

#### OxygenBottleneck

**Map:** `evals/eval_oxygen_bottleneck.map`

**Challenge:** Oxygen extraction is significantly slower; forces batching other resources

**Configuration:**

- Energy regen: 2/step
- Charger efficiency: 130
- Resource efficiency: Carbon 115, **Oxygen 60**, Germanium 80, Silicon 120
- Max uses: Charger unlimited, Carbon 120, **Oxygen 30**, Germanium 15, Silicon 120
- **Best for:** Testing resource prioritization and pacing
- **Recommended agents:** 1-4

**Skills tested:** Resource prioritization, batching, waiting/pacing, opportunistic gathering

---

#### EnergyStarved

**Map:** `evals/eval_energy_starved.map`

**Challenge:** Low energy regen and weak chargers; requires careful routing and charging

**Configuration:**

- Energy regen: 1/step
- Inventory regen interval: 2 steps (slower!)
- Charger efficiency: 90 (weak!)
- Resource efficiency: Carbon 125, Oxygen 115, Germanium 100, Silicon 125
- Max uses: Charger unlimited, all extractors unlimited
- **Best for:** Testing energy management and preemptive charging
- **Recommended agents:** 1-4

**Skills tested:** Energy management, preemptive charging, efficient routing, distance-to-charger calculations

---

### Extractor Hub Missions

Large open maps with centralized extractors, testing exploration and navigation efficiency.

#### ExtractorHub30

**Map:** `evals/extractor_hub_30x30.map` (30×30)

**Challenge:** Small hub, quick exploration

**Configuration:**

- Energy regen: 2/step
- Charger efficiency: 125
- Resource efficiency: Carbon 115, Oxygen 110, Germanium 90, Silicon 120
- Max uses: Germanium unlimited, others use mission defaults
- **Recommended agents:** 1-4

**Skills tested:** Basic exploration, hub navigation

---

#### ExtractorHub50

**Map:** `evals/extractor_hub_50x50.map` (50×50)

**Challenge:** Medium hub, moderate exploration

**Configuration:**

- Energy regen: 2/step
- Charger efficiency: 125
- Resource efficiency: Carbon 115, Oxygen 110, Germanium 90, Silicon 120
- Max uses: Germanium unlimited, others use mission defaults
- **Recommended agents:** 1-4

**Skills tested:** Medium-range exploration, efficient pathfinding

---

#### ExtractorHub70

**Map:** `evals/extractor_hub_70x70.map` (70×70)

**Challenge:** Large hub, significant exploration required

**Configuration:**

- Energy regen: 2/step
- Charger efficiency: 130
- Resource efficiency: Carbon 115, Oxygen 110, Germanium 95, Silicon 120
- Max uses: Germanium unlimited, others use mission defaults
- **Recommended agents:** 2-8

**Skills tested:** Large-map exploration, frontier-based navigation

---

#### ExtractorHub80

**Map:** `evals/extractor_hub_80x80.map` (80×80)

**Challenge:** Very large hub, extensive exploration

**Configuration:**

- Energy regen: 2/step
- Charger efficiency: 135
- Resource efficiency: Carbon 115, Oxygen 110, Germanium 95, Silicon 120
- Max uses: Germanium unlimited, others use mission defaults
- **Recommended agents:** 4-8

**Skills tested:** Long-range exploration, timeout prevention

---

#### ExtractorHub100

**Map:** `evals/extractor_hub_100x100.map` (100×100)

**Challenge:** Extreme hub size, timeout risk high

**Configuration:**

- Energy regen: 2/step
- Charger efficiency: 140
- Resource efficiency: Carbon 115, Oxygen 110, Germanium 100, Silicon 120
- Max uses: Germanium unlimited, others use mission defaults
- **Recommended agents:** 4-8

**Skills tested:** Extreme-scale exploration, map-size-aware strategies, multi-agent fanning out

---

## Difficulty Variants

Difficulty variants are modifiers that can be applied to any mission to create varied challenges. Each variant adjusts
extractor max_uses, efficiency, energy_regen, and other parameters.

All difficulty variants are defined in `difficulty_variants.py`.

### Standard Difficulty Progression

#### story_mode

**Description:** Abundant energy/resource output for reliable completion

**Agent Scaling:** Disabled (fixed parameters)

**Parameters:**

- Max uses: 12 for all extractors (generous)
- Efficiency: 140 for all extractors, 150 for charger (high)
- Energy regen: 2/step (abundant)
- **Use case:** Tutorial/practice runs, testing core logic

**Best for:** Verifying agent logic works at all; first-pass testing

---

#### standard

**Description:** Baseline mission parameters (default difficulty)

**Agent Scaling:** Enabled

**Parameters:**

- Uses mission's default values for all parameters
- No multipliers or overrides applied
- **Use case:** Standard evaluation baseline

**Best for:** Default evaluation setting; fair comparison baseline

---

#### hard

**Description:** Tight extractor budgets and no passive regen

**Agent Scaling:** Disabled

**Parameters:**

- Max uses: Carbon 4, Oxygen 4, Germanium 6, Silicon 3 (tight!)
- Efficiency: Carbon 80, Oxygen 65, Germanium 75, Silicon 70, Charger 80 (reduced)
- Energy regen: 0/step (none!)
- Move energy cost: 3 (increased from default)
- **Use case:** Testing resource scarcity and charger usage

**Best for:** Testing preemptive charging, efficient routing, extractor awareness

---

#### brutal

**Description:** Extreme scarcity, reduced inventories, perfection required

**Agent Scaling:** Disabled

**Parameters:**

- Max uses: Carbon 2, Oxygen 2, Germanium 3, Silicon 2 (minimal!)
- Efficiency: Carbon 55, Oxygen 45, Germanium 50, Silicon 50, Charger 60 (very low)
- Energy regen: 0/step
- Move energy cost: 3
- Energy capacity: 70 (reduced from default)
- Cargo capacity: 80 (reduced from default)
- **Use case:** Extreme challenge; near-perfect play required

**Best for:** Stress-testing optimal strategies, finding edge cases

---

### Specialized Difficulty Variants

#### single_use

**Description:** Every extractor can be used exactly once - no second chances

**Agent Scaling:** Disabled

**Parameters:**

- Max uses: 1 for all extractors (single use!)
- Charger efficiency: 120 (compensatory boost)
- Energy regen: 1/step
- **Use case:** Testing extractor assignment and single-use logic

**Best for:** Testing extractor reservation, resource pooling, strategic planning

---

#### speed_run

**Description:** Short clock, cheap movement, efficient extraction

**Agent Scaling:** Enabled

**Parameters:**

- Max uses: 6 for all extractors
- Efficiency: 160 for all extractors and charger (high)
- Energy regen: 2/step
- Move energy cost: 1 (cheap!)
- Max steps: 600 (shortened from default 1000)
- **Use case:** Testing fast execution and efficient strategies

**Best for:** Benchmarking execution speed, optimizing routing

---

#### energy_crisis

**Description:** Zero passive regen and weak chargers - plan every move

**Agent Scaling:** Disabled

**Parameters:**

- Charger efficiency: 50 (very weak!)
- Energy regen: 0/step (none!)
- **Use case:** Testing extreme energy management

**Best for:** Testing preemptive charging, distance-aware routing, energy budgeting

---

### Clipping Difficulty Variants

Clipping variants introduce extractors that start "clipped" (disabled) and require crafting special items to unclip.
Each variant clips a specific resource and provides one immune extractor for crafting the unclip item.

**Unclipping Mechanics:**

1. One extractor starts clipped (disabled)
2. One other extractor is immune (always usable)
3. Agent must gather from immune extractor
4. Agent crafts unclip item at assembler using the "gear" glyph
5. Agent uses unclip item on clipped extractor to reactivate it

**Unclip Item Recipes:**

- **Decoder** (unclips oxygen): Requires 1 carbon
- **Modulator** (unclips carbon): Requires 1 oxygen
- **Resonator** (unclips germanium): Requires 1 silicon
- **Scrambler** (unclips silicon): Requires 1 germanium

---

#### clipped_oxygen

**Description:** Oxygen extractor starts clipped - craft decoder from carbon to unclip

**Immune Extractor:** carbon_extractor

**Challenge:** Must gather carbon → craft decoder → unclip oxygen → assemble hearts

**Skills tested:** Unclipping logic, recipe awareness, assembly glyphing (gear vs heart)

---

#### clipped_carbon

**Description:** Carbon extractor starts clipped - craft modulator from oxygen to unclip

**Immune Extractor:** oxygen_extractor

**Challenge:** Must gather oxygen → craft modulator → unclip carbon → assemble hearts

**Skills tested:** Unclipping logic, oxygen-first strategy

---

#### clipped_germanium

**Description:** Germanium extractor starts clipped - craft resonator from silicon to unclip

**Immune Extractor:** silicon_extractor

**Challenge:** Must gather silicon → craft resonator → unclip germanium → assemble hearts

**Skills tested:** Unclipping logic, silicon-first strategy

---

#### clipped_silicon

**Description:** Silicon extractor starts clipped - craft scrambler from germanium to unclip

**Immune Extractor:** germanium_extractor

**Challenge:** Must gather germanium → craft scrambler → unclip silicon → assemble hearts

**Skills tested:** Unclipping logic, germanium-first strategy

---

#### clipping_chaos

**Description:** Random extractors clip over time - must craft unclip items reactively

**Clip Rate:** 0.10 (10% chance per step)

**Challenge:** Extractors randomly clip during gameplay; must detect and unclip reactively

**Skills tested:** Dynamic clipping detection, reactive crafting, inventory management for multiple gear types

---

#### hard_clipped_oxygen

**Description:** Combines hard mode scarcity with oxygen clipping

**Immune Extractor:** carbon_extractor

**Parameters:**

- Max uses: Carbon 4, Oxygen 4, Germanium 6, Silicon 3
- Efficiency: Carbon 80, Oxygen 65, Germanium 75, Silicon 70, Charger 80
- Energy regen: 0/step
- Move energy cost: 3
- Oxygen starts clipped

**Challenge:** Extreme scarcity + unclipping; most difficult variant

**Skills tested:** All hard mode skills + unclipping under pressure

---

## Quick Reference

### Mission Categories

| Category              | Missions                                        | Key Challenge                      |
| --------------------- | ----------------------------------------------- | ---------------------------------- |
| **Basic Collection**  | CollectResourcesClassic, CollectResourcesSpread | Gather-assemble-deliver loop       |
| **Distance/Scarcity** | CollectFar                                      | Long distances, limited extractors |
| **Coordination**      | DivideAndConquer, GoTogether, SingleUseSwarm    | Multi-agent cooperation            |
| **Bottlenecks**       | OxygenBottleneck, EnergyStarved                 | Resource/energy constraints        |
| **Exploration**       | ExtractorHub30/50/70/80/100                     | Map exploration, large scales      |

### Difficulty Categories

| Category                 | Difficulties                                                                 | Key Challenge                  |
| ------------------------ | ---------------------------------------------------------------------------- | ------------------------------ |
| **Standard Progression** | story_mode, standard, hard, brutal                                           | Increasing scarcity/difficulty |
| **Specialized**          | single_use, speed_run, energy_crisis                                         | Unique constraints             |
| **Clipping**             | clipped_oxygen/carbon/germanium/silicon, clipping_chaos, hard_clipped_oxygen | Unclipping mechanics           |

### Agent Count Recommendations

| Mission                 | 1 Agent    | 2 Agents | 4 Agents | 8 Agents     |
| ----------------------- | ---------- | -------- | -------- | ------------ |
| CollectResourcesClassic | ✅ Good    | ✅ Good  | ✅ Good  | ⚠️ Overkill  |
| CollectResourcesSpread  | ✅ Good    | ✅ Good  | ✅ Good  | ⚠️ Possible  |
| CollectFar              | ⚠️ Hard    | ✅ Good  | ✅ Good  | ✅ Good      |
| DivideAndConquer        | ❌ N/A     | ✅ Good  | ✅ Good  | ✅ Good      |
| GoTogether              | ❌ N/A     | ✅ Good  | ✅ Good  | ⚠️ Possible  |
| SingleUseSwarm          | ❌ N/A     | ⚠️ Hard  | ✅ Good  | ✅ Good      |
| OxygenBottleneck        | ✅ Good    | ✅ Good  | ✅ Good  | ⚠️ Possible  |
| EnergyStarved           | ✅ Good    | ✅ Good  | ⚠️ Hard  | ❌ Very Hard |
| ExtractorHub30          | ✅ Good    | ✅ Good  | ✅ Good  | ⚠️ Overkill  |
| ExtractorHub50          | ✅ Good    | ✅ Good  | ✅ Good  | ⚠️ Possible  |
| ExtractorHub70          | ⚠️ Slow    | ✅ Good  | ✅ Good  | ✅ Good      |
| ExtractorHub80          | ⚠️ Slow    | ✅ Good  | ✅ Good  | ✅ Good      |
| ExtractorHub100         | ❌ Timeout | ⚠️ Hard  | ✅ Good  | ✅ Good      |

---

## Usage Examples

### Playing a Mission

```bash
# Basic mission with default (standard) difficulty
uv run cogames play --mission evals.collect_resources_classic --cogs 2

# Mission with specific difficulty variant
uv run cogames play --mission evals.oxygen_bottleneck --cogs 4 --difficulty hard

# Mission with clipping
uv run cogames play --mission evals.extractor_hub_30 --cogs 1 --difficulty clipped_oxygen

# Multi-agent coordination mission
uv run cogames play --mission evals.go_together --cogs 4 --difficulty standard
```

### Evaluation Script Usage

```bash
# Evaluate a single agent on all missions and difficulties
uv run python packages/cogames/scripts/run_evaluation.py \
  --agent simple \
  --steps 1000 \
  --output eval_simple.json

# Evaluate with specific difficulty filter
uv run python packages/cogames/scripts/run_evaluation.py \
  --agent coordinating \
  --difficulties clipped_oxygen clipped_silicon \
  --steps 1000 \
  --output eval_coordinating_clipped.json
```

### Spanning Eval Suite (Integrated Evals)

```bash
# Evaluate a policy on the integrated eval suite (spanning evals)
uv run python packages/cogames/scripts/run_evaluation.py \
  --agent cogames.policy.nim_agents.agents.ThinkyAgentsMultiPolicy \
  --mission-set integrated_evals \
  --cogs 4 \
  --repeats 2
```

### Testing Specific Scenarios

```bash
# Test single-agent unclipping
uv run cogames play --mission evals.collect_resources_classic -p unclipping --cogs 1 --difficulty clipped_oxygen

# Test multi-agent coordination without clipping
uv run cogames play --mission evals.go_together -p coordinating --cogs 4 --difficulty standard

# Test extreme scarcity
uv run cogames play --mission evals.oxygen_bottleneck -p scripted_baseline --cogs 1 --difficulty brutal

# Test large-scale exploration
uv run cogames play --mission evals.extractor_hub_100 -p coordinating --cogs 8 --difficulty standard

# Test energy management
uv run cogames play --mission evals.energy_starved -p scripted_baseline --cogs 2 --difficulty energy_crisis
```

---

## Design Principles

### Missions

1. **Focused challenges**: Each mission tests specific skills (exploration, coordination, bottlenecks)
2. **Scalable**: Missions work across different agent counts (with recommended ranges)
3. **Observable**: Agents can learn strategies through observation and trial
4. **Balanced baselines**: Default configurations are solvable but challenging

### Difficulty Variants

1. **Composable**: Any difficulty can be applied to any mission
2. **Skill-specific**: Each variant tests specific capabilities (scarcity, speed, unclipping)
3. **Progressive**: Clear difficulty progression from story_mode → standard → hard → brutal
4. **Specialized challenges**: Unique constraints (single_use, clipping) test specific algorithms

### Evaluation Philosophy

1. **No arbitrary thresholds**: Missions are winnable through observation and adaptation
2. **Deterministic mechanics**: Clipping is the only stochastic element (except clipping_chaos)
3. **Multiple solution paths**: Missions don't prescribe a single correct strategy
4. **Failure modes are informative**: Logs/traces should reveal why an agent failed

---

## Common Failure Modes

| Failure Mode            | Likely Cause                     | Test With                                 |
| ----------------------- | -------------------------------- | ----------------------------------------- |
| Timeout (0 hearts)      | Exploration inefficiency         | ExtractorHub100, CollectFar               |
| Low hearts (< expected) | Resource scarcity, poor routing  | hard, brutal difficulties                 |
| Zero hearts (energy)    | Energy management failure        | energy_crisis, EnergyStarved              |
| Zero hearts (clipping)  | Unclipping logic failure         | Any clipped\_\* difficulty                |
| Coordination failure    | Multi-agent collision/contention | GoTogether, SingleUseSwarm with 4+ agents |
| Bottleneck failure      | Resource prioritization failure  | OxygenBottleneck, clipped_oxygen          |

---

## Contributing New Missions

To add a new evaluation mission:

1. Create a new class in `eval_missions.py` inheriting from `_EvalMissionBase`
2. Define `name`, `description`, and `map_name`
3. Set efficiency, max_uses, and energy_regen parameters
4. Add to `EVAL_MISSIONS` list at bottom of file
5. Test across all difficulty variants to verify solvability
6. Document in this README with challenge description and recommended agent counts

To add a new difficulty variant:

1. Create a new `DifficultyLevel` in `difficulty_variants.py`
2. Define overrides/multipliers for extractors, energy, and special mechanics
3. Document in this README with challenge description and use cases

---

## Integrated Evals

A curated set in `integrated_eval.py` balances “scorable” baselines with room to improve. It uses the procedural
`HELLO_WORLD` site and combines shaping variants (e.g., compass, caps, reward shaping) with constraints (e.g.,
bottlenecks, energy pressure, single-use) to avoid common zero-score failure modes.

Included missions (variants shown):

- oxygen_bottleneck: EmptyBase(missing=[oxygen_extractor]), ResourceBottleneck([oxygen]),
  SingleResourceUniform(oxygen_extractor), NeutralFaced, PackRat
- energy_starved: EmptyBase, DarkSide, NeutralFaced, PackRat
- distant_resources: EmptyBase, Compass, DistantResources
- quadrant_buildings: EmptyBase, QuadrantBuildings, Compass, NeutralFaced
- single_use_swarm: EmptyBase, SingleUseSwarm, Compass, PackRat
- vibe_check: HeartChorus, VibeCheckMin2

Why these choices:

- **Scorable**: Agents commonly achieve >0 reward under time limits without trivializing the task.
- **Headroom**: Bottlenecks, distance, and single-use still require coordination and planning to approach ceiling
  scores.
- **Signal**: Reward shaping on vibe tasks and inventory/compass guidance produce informative learning signals.

Run the set:

```bash
uv run python packages/cogames/scripts/run_evaluation.py \
  --agent cogames.policy.nim_agents.agents.ThinkyAgentsMultiPolicy \
  --mission-set integrated_evals \
  --cogs 4 \
  --repeats 2
```

Tip: For quick CI smoke tests, restrict `--experiments` to a small subset of missions.

---

**Last Updated:** November 4, 2025

**Total Missions:** 14

**Total Difficulty Variants:** 13

**Total Test Configurations:** 1,078+ (14 missions × 13 difficulties × variable agent counts)
