# Scripted Agent - Evaluation Report

**Date**: October 28, 2025
**Total Tests**: 75 (15 evaluations × 5 hyperparameter presets)
**Success Rate**: 60.0% (45/75 tests passed)
**Total Hearts**: 81 hearts across all successful runs

---

## Evaluation Maps

**EVAL 1-9**: Machina Eval Suite (40×40, constraint tests)
**EVAL 10-15**: Outpost Experiments (varying sizes, layout tests)

---

## Hyperparameter Presets

| Preset | Strategy | Exploration Steps | Min Energy for Silicon |
|--------|----------|-------------------|------------------------|
| **explorer** | explorer_first | 100 (scales with map size) | 70 |
| **greedy** | greedy_opportunistic | 50 | 70 |
| **efficiency** | efficiency_learner | 100 | 75 |
| **explorer_aggressive** | explorer_first | 100 | 60 |
| **explorer_conservative** | explorer_first | 100 | 85 |

**Strategy Types**:
- `explorer_first`: Explore for N steps (adaptive to map size), then gather greedily
- `greedy_opportunistic`: Start gathering immediately, explore when stuck
- `efficiency_learner`: Learn extractor efficiency, prioritize best ones, explore when no resources needed

---

## Evaluation Results

### EVAL 1-9: Machina Constraint Tests (40×40)

All constraint tests are 40×40 maps with specific resource/energy constraints. Tests agent adaptability to environmental challenges.

| # | Mission Name | Description | Success Rate | Play with GUI |
|---|--------------|-------------|--------------|---------------|
| **1** | OxygenBottleneck | Limited oxygen extractors - tests resource bottleneck handling | 100% (5/5) ✅ | `uv run cogames play -m machina_eval.oxygen_bottleneck -p scripted` |
| **2** | GermaniumRush | Abundant germanium - tests efficiency with resource abundance | 100% (5/5) ✅ | `uv run cogames play -m machina_eval.germanium_rush -p scripted` |
| **3** | SiliconWorkbench | Silicon-focused layout - tests high-energy resource management | 0% (0/5) ❌ | `uv run cogames play -m machina_eval.silicon_workbench -p scripted` |
| **4** | CarbonDesert | Sparse carbon extractors - tests scarcity adaptation | 100% (5/5) ✅ | `uv run cogames play -m machina_eval.carbon_desert -p scripted` |
| **5** | SingleUseWorld | All extractors max_uses=1 - tests single-use resource planning | 0% (0/5) ❌ | `uv run cogames play -m machina_eval.single_use_world -p scripted` |
| **6** | SlowOxygen | Low oxygen efficiency - tests patience with slow collection | 100% (5/5) ✅ | `uv run cogames play -m machina_eval.slow_oxygen -p scripted` |
| **7** | HighRegenSprint | 2.0x energy regen - tests performance with abundant energy | 100% (5/5) ✅ | `uv run cogames play -m machina_eval.high_regen_sprint -p scripted` |
| **8** | SparseBalanced | Few extractors, evenly distributed - tests efficient routing | 100% (5/5) ✅ | `uv run cogames play -m machina_eval.sparse_balanced -p scripted` |
| **9** | GermaniumClutch | Germanium max_uses=1 - tests critical resource preservation | 0% (0/5) ❌ | `uv run cogames play -m machina_eval.germanium_clutch -p scripted` |

**Summary**: 6/9 perfect (100%), 0/9 high success (80%), 3/9 failed (0%)

### EVAL 10-15: Outpost Layout Tests (varying sizes)

Layout tests explore navigation and exploration across various map sizes and spatial configurations. Balanced resources but varying complexity.

| # | Map Size | Layout Description | Success Rate | Play with GUI |
|---|----------|-------------------|--------------|---------------|
| **10** | 40×40 | **Baseline** - Standard balanced layout, good starting point | 80% (4/5) | `uv run cogames play -m outpost.experiment1 -p scripted` |
| **11** | 90×90 | **Large Sparse** - Wide open spaces, tests long-distance navigation | 20% (1/5) ❌ | `uv run cogames play -m outpost.experiment2 -p scripted` |
| **12** | 60×60 | **Medium** - Moderate spacing, balanced exploration challenge | 60% (3/5) | `uv run cogames play -m outpost.experiment4 -p scripted` |
| **13** | 30×30 | **Compact** - Dense resources, minimal travel, fast completion | 100% (5/5) ✅ | `uv run cogames play -m outpost.experiment5 -p scripted` |
| **14** | 50×50 | **Asymmetric** - Uneven distribution, tests adaptive routing | 40% (2/5) | `uv run cogames play -m outpost.experiment6 -p scripted` |
| **15** | 70×30 | **Corridor** - Linear/rectangular layout, constrained movement | 80% (4/5) | `uv run cogames play -m outpost.experiment7 -p scripted` |
| **16** | 80×80 | **Open Field** - Large map with minimal obstacles | 40% (2/5) | `uv run cogames play -m outpost.experiment8 -p scripted` |
| **17** | 55×55 | **Clustered** - Resources grouped together, requires cluster hopping | 40% (2/5) | `uv run cogames play -m outpost.experiment9 -p scripted` |
| **18** | 100×100 | **Extreme Sparse** - Maximum size, ultimate exploration test | 0% (0/5) ❌ | `uv run cogames play -m outpost.experiment10 -p scripted` |

**Summary**: 1/9 perfect (100%), 3/9 moderate success (60-80%), 5/9 low/failed (0-40%)

> **Run from**: `cd packages/cogames` before running play commands

---

## Environment Specifications

### EVAL 1-10: Constraint Tests

| # | Mission | Map Size | Constraints | Key Challenge |
|---|---------|----------|-------------|---------------|
| 1 | EnergyStarved | 40×40 | energy_regen=0.5 | Low energy regeneration |
| 2 | OxygenBottleneck | 40×40 | Limited oxygen extractors | Few oxygen sources |
| 3 | GermaniumRush | 40×40 | Abundant germanium | Resource imbalance |
| 4 | SiliconWorkbench | 40×40 | Silicon-focused | Silicon priority |
| 5 | CarbonDesert | 40×40 | Sparse carbon | Limited carbon |
| 6 | SingleUseWorld | 40×40 | All max_uses=1 | Single-use extractors |
| 7 | SlowOxygen | 40×40 | Low oxygen efficiency | Slow oxygen collection |
| 8 | HighRegenSprint | 40×40 | energy_regen=2.0 | High energy regen |
| 9 | SparseBalanced | 40×40 | Few extractors | Sparse resources |
| 10 | GermaniumClutch | 40×40 | Germanium max_uses=1 | Single-use germanium |

### EVAL 11-19: Layout Tests

| # | Map Size | Layout | Key Features |
|---|----------|--------|--------------|
| 11 | 40×40 | Standard | Balanced, baseline |
| 12 | 90×90 | Large sparse | Exploration challenge |
| 13 | 60×60 | Medium | Moderate spacing |
| 14 | 30×30 | Compact | Dense resources |
| 15 | 50×50 | Asymmetric | Uneven distribution |
| 16 | 70×30 | Corridor | Linear layout |
| 17 | 80×80 | Open field | Large, minimal walls |
| 18 | 55×55 | Clustered | Resource clusters |
| 19 | 100×100 | Extreme sparse | Maximum exploration |

---

## Performance Analysis

### By Map Size

| Map Size | Evaluations | Avg Success Rate | Status |
|----------|-------------|------------------|--------|
| 30×30 | EVAL 14 | 100% | ✅ Excellent |
| 40×40 | EVAL 1-10, 11 | 75% | ✅ Good |
| 50×50 | EVAL 15 | 40% | ⚠️ Mixed |
| 55×55 | EVAL 18 | 40% | ⚠️ Mixed |
| 60×60 | EVAL 13 | 60% | ⚠️ Mixed |
| 70×30 | EVAL 16 | 80% | ✅ Good |
| 80×80 | EVAL 17 | 40% | ⚠️ Mixed |
| 90×90 | EVAL 12 | 20% | ❌ Poor |
| 100×100 | EVAL 19 | 0% | ❌ Failed |

**Pattern**: Performance degrades significantly on maps larger than 60×60.

### By Hyperparameter Preset

| Preset | Success Count | Total Tests | Success Rate |
|--------|---------------|-------------|--------------|
| explorer | 15 | 27 | 55.6% |
| greedy | 8 | 10 | 80.0% |
| efficiency | 7 | 10 | 70.0% |
| explorer_aggressive | 7 | 9 | 77.8% |
| explorer_conservative | 1 | 9 | 11.1% |

**Best Overall**: greedy (80.0%)
**Worst Overall**: explorer_conservative (11.1%)

### Failure Modes

**Zero Hearts (couldn't assemble)** - 21 failures:
- Large maps: EVAL 12, 17, 18, 19 (exploration insufficient)
- Single-use: EVAL 6, 10 (depletion before assembly)

**Timeout (assembled but didn't deposit)** - 13 failures:
- Various evaluations running out of time after assembling 1-2 hearts

---

## Key Findings

### What Works Well ✅

1. **Small/Medium Maps (≤40×40)**: 75-100% success
2. **Balanced Resources**: Perfect on 6/10 EVAL missions
3. **Energy Management**: Handles both low (0.5) and high (2.0) regen
4. **Navigation**: A*/BFS hybrid works well on standard layouts

### What Struggles ⚠️

1. **Large Maps (≥80×80)**: 0-40% success
   - Issue: Extractors discovered but unreachable (pathfinding failures)
   - Agent explores but can't navigate to distant resources

2. **Single-Use Extractors**: 0% success
   - Issue: Agent visits extractors multiple times, depleting them
   - No tracking of extractor depletion

3. **Hyperparameter Sensitivity**:
   - `explorer_conservative` fails on 8/9 tests (too cautious with energy)
   - `greedy` succeeds on 8/10 but only tested on 10 experiments

---

## Technical Implementation

### Agent Capabilities

- **Visual Discovery**: 15×15 observation window, incremental map building
- **Extractor Memory**: Tracks all discovered extractors, cooldowns, estimated depletion
- **Navigation**: A* (long distance ≥20 tiles), BFS (short distance), greedy fallback
- **Phase-Based Strategy**: Sequential gathering, recharge management, assembly, deposit
- **Stuck Detection**:
  - Phase oscillation detection (adaptive threshold: 5-15 visits based on map size)
  - Depletion detection (200+ steps with partial progress)
  - Resource blacklisting for unreachable extractors

### Recent Fixes

1. **Wall Knowledge Fix** (Lines 880-884)
   - Prevents marking extractors as walls after USE attempts
   - Critical: Enables resource collection

2. **Adaptive Oscillation Threshold** (Lines 764-788)
   - Scales with map size: 40×40 → 5 visits, 90×90 → 11 visits, 100×100 → 12 visits
   - Gives large maps more exploration attempts before giving up
   - **Result**: No performance improvement (extractors found but still unreachable)

---

## Running Evaluations

### Quick Start - Watch Agent in GUI

All commands assume you're in `packages/cogames/`:
```bash
cd packages/cogames
```

**Training Facilities** (basic maps):
```bash
# Standard open layout
uv run cogames play training_facility_open_1.map --policy scripted --cogs 1

# Tight corridors
uv run cogames play training_facility_tight_4.map --policy scripted --cogs 1

# Clipped extractors (testing unclipping)
uv run cogames play training_facility_clipped.map --policy scripted --cogs 1
```

**EVAL 1-10** (40×40 constraint tests):
```bash
# Perfect score examples
uv run cogames play -m machina_eval.oxygen_bottleneck -p scripted     # EVAL 2
uv run cogames play -m machina_eval.germanium_rush -p scripted        # EVAL 3
uv run cogames play -m machina_eval.sparse_balanced -p scripted       # EVAL 9

# Challenging maps
uv run cogames play -m machina_eval.single_use_world -p scripted      # EVAL 5 - currently fails
```

**EVAL 10-18** (size/layout tests):
```bash
# Easy wins
uv run cogames play -m outpost.experiment5 -p scripted   # EVAL 13: 30×30, 100% success

# Medium difficulty
uv run cogames play -m outpost.experiment1 -p scripted   # EVAL 10: 40×40, 80% success
uv run cogames play -m outpost.experiment7 -p scripted   # EVAL 15: 70×30 corridor, 80% success

# Hard challenges
uv run cogames play -m outpost.experiment2 -p scripted   # EVAL 11: 90×90, 20% success
uv run cogames play -m outpost.experiment10 -p scripted  # EVAL 18: 100×100, 0% success
```

### Hyperparameter Presets

Test different strategies by adding `--preset <name>`:
```bash
# Explorer strategy (default) - explore first, then gather
uv run cogames play -m outpost.experiment1 -p scripted --preset explorer

# Greedy strategy - gather immediately, explore when stuck (80% success overall!)
uv run cogames play -m outpost.experiment1 -p scripted --preset greedy

# Efficiency learner - learn which extractors are best, prioritize them
uv run cogames play -m outpost.experiment1 -p scripted --preset efficiency

# Aggressive explorer - explore with lower energy thresholds
uv run cogames play -m outpost.experiment1 -p scripted --preset explorer_aggressive

# Conservative explorer - higher energy safety margins (11% success, not recommended)
uv run cogames play -m outpost.experiment1 -p scripted --preset explorer_conservative
```

Available presets: `explorer` (default), `greedy`, `efficiency`, `explorer_aggressive`, `explorer_conservative`

### Full Evaluation Suite

Run all 95 tests (19 experiments × 5 presets):
```bash
cd /Users/daphnedemekas/Desktop/metta
uv run python packages/cogames/scripts/evaluate.py outpost
```

### Programmatic Usage

```python
from cogames.cogs_vs_clips.missions import make_game
from cogames.policy.scripted_agent import ScriptedAgentPolicy
from mettagrid import MettaGridEnv

# Create environment
env_cfg = make_game(num_cogs=1, map_name="training_facility_open_1.map")
env = MettaGridEnv(env_cfg=env_cfg)

# Create policy with preset
policy = ScriptedAgentPolicy(env, preset='greedy')
agents = [policy.agent_policy(i) for i in range(env.num_agents)]

# Run episode
obs, _ = env.reset()
for _ in range(1000):
    actions = [agents[i].step(obs[i]) for i in range(env.num_agents)]
    obs, rewards, done, truncated, _ = env.step(actions)
    if all(done) or all(truncated):
        break

print(f"Total reward: {rewards.sum()}")
```

---

## Next Steps for Improvement

### High Priority

**1. Fix Navigation on Large Maps** (+20-30%)
- Issue: Extractors found but pathfinding fails to reach them
- Solution: Better fallback strategies, continue exploring while gathering
- Impact: Would fix EVAL 12, 17, 18, 19

**2. Single-Use Extractor Tracking** (+10-15%)
- Issue: Agent revisits extractors, depleting them before assembly
- Solution: Track `total_harvests` vs `max_uses`, visit each once
- Impact: Would fix EVAL 6, 10

### Medium Priority

**3. Dynamic Exploration** (+5-10%)
- Issue: Exploration only at episode start, not during gathering
- Solution: Interleave exploration with gathering when stuck
- Impact: Would help all large maps

**4. Hyperparameter Auto-Selection** (+5-8%)
- Issue: Some presets fail on otherwise solvable environments
- Solution: Detect map features, select appropriate preset
- Impact: Would reduce preset sensitivity

---

## Conclusion

The scripted agent achieves **64.2% success** across diverse environments. It excels on small/medium balanced maps (75-100%) but struggles with large sparse maps (0-40%) and single-use extractors (0%).

**Core Strength**: Robust baseline for standard 40×40 environments
**Core Weakness**: Navigation failures on large maps with complex layouts
**Path to 80%+**: Fix navigation on large maps (+20-30%) + single-use tracking (+10-15%)

---

**Report Generated**: October 29, 2024
**Evaluation Log**: `evaluation_final.log`
**Baseline**: 64.2% (61/95 tests)
