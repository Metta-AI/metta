# CoGames Evaluation Environments

This directory contains evaluation missions for testing CoGames agents.

**Contents:**

- [Diagnostic Missions](#diagnostic-missions) - Focused skill tests for specific mechanics
- [Integrated Eval Missions](#integrated-eval-missions) - Procedural missions for training/evaluation
- [Usage Examples](#usage-examples) - How to run missions

---

## Diagnostic Missions

Diagnostic missions are defined in `diagnostic_evals.py` and are playable via
`cogames play --mission evals.diagnostic_*`. Each mission tests a specific skill or mechanic in a controlled
environment.

### Available Diagnostic Missions

See `diagnostic_evals.py` for the complete list. Key missions include:

**Navigation & Delivery:**

- `diagnostic_chest_navigation1/2/3` - Navigate to chest and deposit hearts
- `diagnostic_chest_near` - Chest nearby, test deposit mechanics
- `diagnostic_chest_search` - Find chest through exploration

**Resource Extraction:**

- `diagnostic_extract_missing_carbon/oxygen/germanium/silicon` - Extract specific missing resources

**Assembly:**

- `diagnostic_assembler_near` - Assemble hearts at nearby assembler
- `diagnostic_assembler_search` - Find assembler and craft hearts

**Energy Management:**

- `diagnostic_charge_up` - Test charging mechanics

**Unclipping:**

- `diagnostic_unclip_craft` - Craft unclip items
- `diagnostic_unclip_preseed` - Unclip with pre-seeded inventory

**Complex Scenarios:**

- `diagnostic_radial` - Radial resource layout with chorus assembly
- `diagnostic_agile` - Test agility and quick decision-making
- `diagnostic_memory` - Test memory and state tracking

**Hard Versions:** Most diagnostics have `_hard` variants (e.g., `diagnostic_chest_navigation1_hard`)

---

## Usage Examples

### Playing a Diagnostic Mission

```bash
# Basic diagnostic mission
uv run cogames play --mission evals.diagnostic_chest_navigation1 --cogs 1

# Diagnostic mission with multiple agents
uv run cogames play --mission evals.diagnostic_extract_missing_oxygen --cogs 2

# Hard version of diagnostic
uv run cogames play --mission evals.diagnostic_chest_navigation1_hard --cogs 1

# Unclipping diagnostic
uv run cogames play --mission evals.diagnostic_unclip_craft --cogs 1
```

### Note on Integrated Eval Missions

The integrated eval missions (oxygen_bottleneck, energy_starved, etc.) are **not directly playable** via `cogames play`.
They are used programmatically in training and evaluation scripts. To test similar scenarios, use the diagnostic
missions or the training_facility/hello_world sites with appropriate variants.

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
# Test chest navigation
uv run cogames play --mission evals.diagnostic_chest_navigation1 --cogs 1

# Test resource extraction
uv run cogames play --mission evals.diagnostic_extract_missing_carbon --cogs 1

# Test unclipping mechanics
uv run cogames play --mission evals.diagnostic_unclip_craft --cogs 1

# Test assembly
uv run cogames play --mission evals.diagnostic_assembler_search --cogs 1

# Test with scripted policy
uv run cogames play --mission evals.diagnostic_radial -p scripted_baseline --cogs 2

# Test hard version
uv run cogames play --mission evals.diagnostic_radial_hard -p scripted_baseline --cogs 2
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

## Integrated Eval Missions

Defined in `integrated_evals.py`, these missions use procedural generation with the `HELLO_WORLD` site and mission
variants. They are **not directly playable** via `cogames play` but are used programmatically in training and evaluation
scripts.

### Available Integrated Missions:

- **oxygen_bottleneck** - Oxygen is the limiting resource
- **energy_starved** - Low energy regen and weak chargers
- **distant_resources** - Resources scattered far from base
- **quadrant_buildings** - Buildings placed in four quadrants
- **single_use_swarm** - All extractors are single-use
- **vibe_check** - Vibe-based coordination challenges
- **easy_hearts** - Simplified heart crafting

These missions are used in evaluation scripts with the `run_evaluation.py` tool.

---

**Last Updated:** December 3, 2025

**Diagnostic Missions:** 30+ (various skills and hard variants)

**Integrated Missions:** 7 (procedural, used in training/evaluation)
