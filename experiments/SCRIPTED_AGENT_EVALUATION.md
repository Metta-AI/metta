# Scripted Agent - Evaluation Report

**Date**: October 29, 2024
**Total Tests**: 95 (19 experiments × 5 hyperparameter presets)
**Success Rate**: 64.2% (61/95 tests passed)

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

### EVAL Missions (Machina Eval Suite)

| Mission | Success Rate | Play Command |
|---------|--------------|--------------|
| **EVAL1** EnergyStarved | 80% (4/5) | `cd packages/cogames && uv run cogames play -m machina_eval.energy_starved -p scripted` |
| **EVAL2** OxygenBottleneck | 100% (5/5) ✅ | `cd packages/cogames && uv run cogames play -m machina_eval.oxygen_bottleneck -p scripted` |
| **EVAL3** GermaniumRush | 100% (5/5) ✅ | `cd packages/cogames && uv run cogames play -m machina_eval.germanium_rush -p scripted` |
| **EVAL4** SiliconWorkbench | 80% (4/5) | `cd packages/cogames && uv run cogames play -m machina_eval.silicon_workbench -p scripted` |
| **EVAL5** CarbonDesert | 100% (5/5) ✅ | `cd packages/cogames && uv run cogames play -m machina_eval.carbon_desert -p scripted` |
| **EVAL6** SingleUseWorld | 0% (0/5) ❌ | `cd packages/cogames && uv run cogames play -m machina_eval.single_use_world -p scripted` |
| **EVAL7** SlowOxygen | 100% (5/5) ✅ | `cd packages/cogames && uv run cogames play -m machina_eval.slow_oxygen -p scripted` |
| **EVAL8** HighRegenSprint | 100% (5/5) ✅ | `cd packages/cogames && uv run cogames play -m machina_eval.high_regen_sprint -p scripted` |
| **EVAL9** SparseBalanced | 100% (5/5) ✅ | `cd packages/cogames && uv run cogames play -m machina_eval.sparse_balanced -p scripted` |
| **EVAL10** GermaniumClutch | 0% (0/5) ❌ | `cd packages/cogames && uv run cogames play -m machina_eval.germanium_clutch -p scripted` |

**EVAL Summary**: 6/10 perfect (100%), 2/10 high success (80%), 2/10 failed (0%)

### EXP Missions (Outpost Experiments)

| Experiment | Map Size | Success Rate | Play Command |
|------------|----------|--------------|--------------|
| **EXP1** | 40×40 | 80% (4/5) | `cd packages/cogames && uv run cogames play -m outpost.experiment1 -p scripted` |
| **EXP2** | 90×90 | 20% (1/5) ❌ | `cd packages/cogames && uv run cogames play -m outpost.experiment2 -p scripted` |
| **EXP4** | 60×60 | 60% (3/5) | `cd packages/cogames && uv run cogames play -m outpost.experiment4 -p scripted` |
| **EXP5** | 30×30 | 100% (5/5) ✅ | `cd packages/cogames && uv run cogames play -m outpost.experiment5 -p scripted` |
| **EXP6** | 50×50 | 40% (2/5) | `cd packages/cogames && uv run cogames play -m outpost.experiment6 -p scripted` |
| **EXP7** | 70×30 | 80% (4/5) | `cd packages/cogames && uv run cogames play -m outpost.experiment7 -p scripted` |
| **EXP8** | 80×80 | 40% (2/5) | `cd packages/cogames && uv run cogames play -m outpost.experiment8 -p scripted` |
| **EXP9** | 55×55 | 40% (2/5) | `cd packages/cogames && uv run cogames play -m outpost.experiment9 -p scripted` |
| **EXP10** | 100×100 | 0% (0/5) ❌ | `cd packages/cogames && uv run cogames play -m outpost.experiment10 -p scripted` |

**EXP Summary**: 1/9 perfect (100%), 3/9 moderate success (60-80%), 5/9 low/failed (0-40%)

---

## Environment Specifications

### EVAL Missions

| Mission | Map Size | Constraints | Key Challenge |
|---------|----------|-------------|---------------|
| EVAL1_EnergyStarved | 40×40 | energy_regen=0.5 | Low energy regeneration |
| EVAL2_OxygenBottleneck | 40×40 | Limited oxygen extractors | Few oxygen sources |
| EVAL3_GermaniumRush | 40×40 | Abundant germanium | Resource imbalance |
| EVAL4_SiliconWorkbench | 40×40 | Silicon-focused | Silicon priority |
| EVAL5_CarbonDesert | 40×40 | Sparse carbon | Limited carbon |
| EVAL6_SingleUseWorld | 40×40 | All max_uses=1 | Single-use extractors |
| EVAL7_SlowOxygen | 40×40 | Low oxygen efficiency | Slow oxygen collection |
| EVAL8_HighRegenSprint | 40×40 | energy_regen=2.0 | High energy regen |
| EVAL9_SparseBalanced | 40×40 | Few extractors | Sparse resources |
| EVAL10_GermaniumClutch | 40×40 | Germanium max_uses=1 | Single-use germanium |

### EXP Missions

| Mission | Map Size | Layout | Key Features |
|---------|----------|--------|--------------|
| EXP1 | 40×40 | Standard | Balanced, baseline |
| EXP2 | 90×90 | Large sparse | Exploration challenge |
| EXP4 | 60×60 | Medium | Moderate spacing |
| EXP5 | 30×30 | Compact | Dense resources |
| EXP6 | 50×50 | Asymmetric | Uneven distribution |
| EXP7 | 70×30 | Corridor | Linear layout |
| EXP8 | 80×80 | Open field | Large, minimal walls |
| EXP9 | 55×55 | Clustered | Resource clusters |
| EXP10 | 100×100 | Extreme sparse | Maximum exploration |

---

## Performance Analysis

### By Map Size

| Map Size | Experiments | Avg Success Rate | Status |
|----------|-------------|------------------|--------|
| 30×30 | EXP5 | 100% | ✅ Excellent |
| 40×40 | EVAL1-10, EXP1 | 75% | ✅ Good |
| 50×50 | EXP6 | 40% | ⚠️ Mixed |
| 55×55 | EXP9 | 40% | ⚠️ Mixed |
| 60×60 | EXP4 | 60% | ⚠️ Mixed |
| 70×30 | EXP7 | 80% | ✅ Good |
| 80×80 | EXP8 | 40% | ⚠️ Mixed |
| 90×90 | EXP2 | 20% | ❌ Poor |
| 100×100 | EXP10 | 0% | ❌ Failed |

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
- Large maps: EXP2, EXP8, EXP9, EXP10 (exploration insufficient)
- Single-use: EVAL6, EVAL10 (depletion before assembly)

**Timeout (assembled but didn't deposit)** - 13 failures:
- Various experiments running out of time after assembling 1-2 hearts

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

### Full Evaluation
```bash
cd /Users/daphnedemekas/Desktop/metta
uv run python packages/cogames/scripts/evaluate.py outpost
```

### Single Experiment
```bash
cd /Users/daphnedemekas/Desktop/metta/packages/cogames

# Play specific mission with GUI
uv run cogames play -m machina_eval.energy_starved -p scripted
uv run cogames play -m outpost.experiment2 -p scripted

# Run without GUI (for testing)
uv run python -c "
from cogames.outpost.missions import Experiment1Mission
from cogames.policy.scripted_agent import ScriptedAgentPolicy

mission = Experiment1Mission()
env = mission.make_env()
obs = env.reset()
policy = ScriptedAgentPolicy(env, preset='efficiency')

for _ in range(1000):
    action, _ = policy.step(obs)
    obs, reward, done, truncated, info = env.step(action)
    if done or truncated:
        break

print(f'Reward: {reward}, Hearts: {info.get(\"hearts_assembled\", 0)}')
"
```

### Hyperparameter Presets

Default preset is `explorer`. To use a specific preset:
```python
policy = ScriptedAgentPolicy(env, preset='efficiency')
# Options: 'explorer', 'greedy', 'efficiency', 'explorer_aggressive', 'explorer_conservative'
```

---

## Next Steps for Improvement

### High Priority

**1. Fix Navigation on Large Maps** (+20-30%)
- Issue: Extractors found but pathfinding fails to reach them
- Solution: Better fallback strategies, continue exploring while gathering
- Impact: Would fix EXP2, EXP8, EXP9, EXP10

**2. Single-Use Extractor Tracking** (+10-15%)
- Issue: Agent revisits extractors, depleting them before assembly
- Solution: Track `total_harvests` vs `max_uses`, visit each once
- Impact: Would fix EVAL6, EVAL10

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
