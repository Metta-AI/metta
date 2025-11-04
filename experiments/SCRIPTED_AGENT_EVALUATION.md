# Scripted Agent Evaluation Report

**Date**: November 4, 2025
**Agents Evaluated**: `SimpleBaseline`, `UnclippingAgent`, `CoordinatingAgent`
**Total Tests**: 1,078 configurations across 14 missions, 13 difficulty variants, and multiple agent counts

---

## Executive Summary

- ✅ **Three distinct agents tested**: `SimpleBaseline` (single-agent, non-clipped), `UnclippingAgent` (single-agent, all difficulties), and `CoordinatingAgent` (multi-agent, all difficulties)
- ✅ **Overall success rate**: **21.8%** (235/1,078 tests passed)
- 📊 **Multi-agent challenges**: `CoordinatingAgent` achieves **18.2%** overall, with performance degrading as agent count increases (22.2% → 19.2% → 13.2% for 2/4/8 agents)
- 🔓 **Unclipping works**: `UnclippingAgent` achieves **50%** success on `clipped_oxygen` and `clipped_silicon` difficulties (single-agent)
- ⚠️ **Coordination on clipped maps**: `CoordinatingAgent` with unclipping inheritance achieves only **12.3%** on clipped difficulties, suggesting multi-agent unclipping coordination needs improvement
- 🎯 **Best difficulty**: `energy_crisis` and `standard` both achieve **50%** success rate
- 💔 **Hardest challenges**: `brutal` (0%), `clipped_carbon` (0%), `clipping_chaos` (2%), and large hub maps remain unsolved

---

## Overall Statistics

| Metric | Value |
|--------|-------|
| **Total Tests** | 1,078 |
| **Successes** | 235 (21.8%) |
| **Average Reward** | 1.03 hearts/agent |
| **Agents Tested** | 3 (SimpleBaseline, UnclippingAgent, CoordinatingAgent) |
| **Missions** | 14 evaluation environments |
| **Difficulty Variants** | 13 |
| **Agent Counts** | 1, 2, 4, 8 |

---

## Performance by Agent

### Summary Table

| Agent | Tests | Success Rate | Avg Reward | Avg Hearts | Agent Counts |
|-------|-------|--------------|------------|------------|--------------|
| **SimpleBaseline** | 98 | **35.7%** | 2.51 | 2.51 | 1 |
| **UnclippingAgent** | 182 | **30.2%** | 1.90 | 1.90 | 1 |
| **CoordinatingAgent** | 798 | **18.2%** | 0.66 | 0.66 | 2, 4, 8 |

---

### SimpleBaseline Agent

**Purpose**: Single-agent baseline for non-clipped environments
**Tests**: 98 (7 difficulties × 14 missions, 1 agent)
**Overall Success**: 35/98 (35.7%)

#### Performance by Difficulty

| Difficulty | Success Rate | Avg Reward |
|------------|--------------|------------|
| energy_crisis | **57.1%** | 3.79 |
| standard | **42.9%** | 3.07 |
| story_mode | **42.9%** | 2.79 |
| speed_run | **42.9%** | 2.86 |
| hard | **35.7%** | 2.43 |
| single_use | **28.6%** | 2.64 |
| brutal | **0.0%** | 0.00 |

#### Top Performing Missions

| Mission | Success Rate | Avg Hearts |
|---------|--------------|------------|
| OxygenBottleneck | **85.7%** (6/7) | 6.00 |
| CollectResourcesBase | **85.7%** (6/7) | 4.43 |
| CollectResourcesClassic | **85.7%** (6/7) | 4.71 |
| CollectResourcesSpread | **85.7%** (6/7) | 6.43 |
| ExtractorHub30 | **57.1%** (4/7) | 2.14 |

#### Bottom Performing Missions

| Mission | Success Rate |
|---------|--------------|
| ExtractorHub80 | 0.0% (0/7) |
| ExtractorHub100 | 0.0% (0/7) |
| DivideAndConquer | 0.0% (0/7) |
| GoTogether | 0.0% (0/7) |
| SingleUseSwarm | 0.0% (0/7) |
| EnergyStarved | 0.0% (0/7) |

**Key Observations**:
- Excels at resource collection missions with straightforward layouts
- Struggles with coordination-required missions (GoTogether, DivideAndConquer)
- Cannot handle single-use or large extractor hub maps effectively
- Strong performance on oxygen-bottleneck scenarios (85.7%)

---

### UnclippingAgent

**Purpose**: Single-agent with unclipping capability
**Tests**: 182 (13 difficulties × 14 missions, 1 agent)
**Overall Success**: 55/182 (30.2%)

#### Performance by Difficulty

| Difficulty | Success Rate | Avg Reward | Notes |
|------------|--------------|------------|-------|
| clipped_oxygen | **50.0%** | 3.07 | ✅ Unclipping works |
| clipped_silicon | **50.0%** | 3.21 | ✅ Unclipping works |
| standard | **50.0%** | 3.21 | |
| energy_crisis | **42.9%** | 3.00 | |
| story_mode | **42.9%** | 3.00 | |
| speed_run | **42.9%** | 2.93 | |
| hard | **42.9%** | 2.50 | |
| single_use | **35.7%** | 2.93 | |
| hard_clipped_oxygen | **28.6%** | 0.36 | 🔶 Harder clipping variant |
| clipped_germanium | **7.1%** | 0.50 | 🔴 Unclipping struggles |
| clipped_carbon | **0.0%** | 0.00 | 🔴 Unclipping fails |
| clipping_chaos | **0.0%** | 0.00 | 🔴 Multi-resource clipping fails |
| brutal | **0.0%** | 0.00 | |

#### Top Performing Missions

| Mission | Success Rate | Avg Hearts |
|---------|--------------|------------|
| CollectResourcesBase | **69.2%** (9/13) | 4.23 |
| CollectResourcesClassic | **69.2%** (9/13) | 3.46 |
| CollectResourcesSpread | **69.2%** (9/13) | 5.77 |
| OxygenBottleneck | **61.5%** (8/13) | 2.77 |
| ExtractorHub50 | **53.8%** (7/13) | 2.46 |

#### Bottom Performing Missions

| Mission | Success Rate |
|---------|--------------|
| ExtractorHub80 | 0.0% (0/13) |
| ExtractorHub100 | 0.0% (0/13) |
| CollectFar | 0.0% (0/13) |
| GoTogether | 0.0% (0/13) |
| SingleUseSwarm | 0.0% (0/13) |
| EnergyStarved | 0.0% (0/13) |

**Key Observations**:
- Successfully unclips and uses oxygen/silicon extractors (50% success)
- Struggles with carbon and germanium unclipping (0-7% success)
- `clipping_chaos` (multiple clipped resources) completely unsolved
- Similar mission performance profile to SimpleBaseline but with added unclipping capability
- Single-agent unclipping is functional but needs refinement for harder variants

---

### CoordinatingAgent

**Purpose**: Multi-agent with coordination and unclipping
**Tests**: 798 (13 difficulties × 14 missions × 3 agent counts: 2, 4, 8)
**Overall Success**: 145/798 (18.2%)

#### Performance by Agent Count

| Agent Count | Success Rate | Avg Reward | Tests |
|-------------|--------------|------------|-------|
| **2 agents** | **22.2%** | 1.11 | 266 |
| **4 agents** | **19.2%** | 0.62 | 266 |
| **8 agents** | **13.2%** | 0.23 | 266 |

**Observation**: Performance degrades with more agents, suggesting coordination overhead and resource contention issues.

#### Performance by Difficulty

| Difficulty | Success Rate | Avg Reward | Notes |
|------------|--------------|------------|-------|
| standard | **52.4%** | 2.14 | Best non-clipped |
| energy_crisis | **50.0%** | 2.12 | |
| story_mode | **50.0%** | 1.52 | |
| hard | **40.5%** | 1.69 | |
| speed_run | **40.5%** | 1.17 | |
| single_use | **38.1%** | 1.48 | Better than single-agent! |
| hard_clipped_oxygen | **14.3%** | 0.18 | 🔶 Multi-agent unclipping |
| clipped_silicon | **9.5%** | 0.58 | 🔴 Coordination + unclipping weak |
| clipped_oxygen | **8.3%** | 0.26 | 🔴 Coordination + unclipping weak |
| clipped_germanium | **2.4%** | 0.12 | 🔴 Fails |
| clipping_chaos | **2.4%** | 0.02 | 🔴 Fails |
| clipped_carbon | **0.0%** | 0.00 | 🔴 Fails |
| brutal | **0.0%** | 0.00 | 🔴 Fails |

#### Top Performing Missions

| Mission | Success Rate | Avg Hearts | Agent Counts |
|---------|--------------|------------|--------------|
| CollectResourcesClassic | **40.4%** (23/57) | 1.77 | 2, 4, 8 |
| CollectResourcesSpread | **35.1%** (20/57) | 1.21 | 2, 4, 8 |
| GoTogether | **33.3%** (19/57) | 0.68 | 2, 4, 8 |
| CollectResourcesBase | **29.8%** (17/57) | 1.21 | 2, 4, 8 |
| OxygenBottleneck | **26.3%** (15/57) | 0.61 | 2, 4, 8 |

#### Bottom Performing Missions

| Mission | Success Rate | Notes |
|---------|--------------|-------|
| ExtractorHub100 | 8.8% (5/57) | Large maps difficult |
| CollectFar | 5.3% (3/57) | Distance coordination fails |
| SingleUseSwarm | 5.3% (3/57) | Single-use logic weak |
| EnergyStarved | 0.0% (0/57) | Energy management fails |
| DivideAndConquer | 0.0% (0/57) | Region partitioning fails |

**Key Observations**:
- Multi-agent coordination works well on non-clipped, straightforward missions
- Performance on clipped difficulties is poor (8-14% vs 50% for single-agent UnclippingAgent)
- Suggests multi-agent unclipping coordination is not effectively implemented
- Better at `single_use` missions than single-agent (38.1% vs 28-35%)
- Coordination overhead reduces effectiveness as agent count increases
- `GoTogether` mission shows coordination is functional (33.3%)

---

## Performance by Difficulty Variant

| Difficulty | Success Rate | Avg Reward | Tests | Best Agent |
|------------|--------------|------------|-------|------------|
| **energy_crisis** | **50.0%** | 2.63 | 70 | SimpleBaseline (57.1%) |
| **standard** | **50.0%** | 2.54 | 70 | CoordinatingAgent (52.4%) |
| **story_mode** | **47.1%** | 2.07 | 70 | CoordinatingAgent (50.0%) |
| **speed_run** | **41.4%** | 1.86 | 70 | SimpleBaseline (42.9%) |
| **hard** | **40.0%** | 2.00 | 70 | CoordinatingAgent (40.5%) |
| **single_use** | **35.7%** | 2.00 | 70 | CoordinatingAgent (38.1%) |
| **hard_clipped_oxygen** | **16.3%** | 0.20 | 98 | CoordinatingAgent (14.3%) |
| **clipped_silicon** | **15.3%** | 0.96 | 98 | UnclippingAgent (50.0%) |
| **clipped_oxygen** | **14.3%** | 0.66 | 98 | UnclippingAgent (50.0%) |
| **clipped_germanium** | **3.1%** | 0.17 | 98 | UnclippingAgent (7.1%) |
| **clipping_chaos** | **2.0%** | 0.02 | 98 | CoordinatingAgent (2.4%) |
| **clipped_carbon** | **0.0%** | 0.00 | 98 | All fail |
| **brutal** | **0.0%** | 0.00 | 70 | All fail |

**Key Insights**:
- Non-clipped difficulties perform well (35-50% success)
- Single-agent unclipping works for oxygen/silicon (50%)
- Multi-agent unclipping severely underperforms (8-14% vs 50%)
- Carbon unclipping completely fails across all agents
- `brutal` difficulty remains completely unsolved

---

## Performance by Mission

| Mission | Success Rate | Avg Reward | Tests | Best Configuration |
|---------|--------------|------------|-------|--------------------|
| **CollectResourcesClassic** | **49.4%** | 2.90 | 77 | SimpleBaseline (85.7%) |
| **CollectResourcesSpread** | **45.5%** | 3.49 | 77 | SimpleBaseline (85.7%) |
| **CollectResourcesBase** | **41.6%** | 3.31 | 77 | SimpleBaseline (85.7%) |
| **OxygenBottleneck** | **37.7%** | 1.47 | 77 | SimpleBaseline (85.7%) |
| **ExtractorHub50** | **28.6%** | 0.86 | 77 | UnclippingAgent (53.8%) |
| **GoTogether** | **24.7%** | 0.35 | 77 | CoordinatingAgent (33.3%) |
| **ExtractorHub30** | **23.4%** | 0.56 | 77 | SimpleBaseline (57.1%) |
| **ExtractorHub70** | **22.1%** | 0.70 | 77 | CoordinatingAgent (26.3%) |
| **ExtractorHub80** | **14.3%** | 0.57 | 77 | CoordinatingAgent (19.3%) |
| **ExtractorHub100** | **6.5%** | 0.10 | 77 | CoordinatingAgent (8.8%) |
| **CollectFar** | **5.2%** | 0.08 | 77 | CoordinatingAgent (5.3%) |
| **SingleUseSwarm** | **3.9%** | 0.04 | 77 | CoordinatingAgent (5.3%) |
| **DivideAndConquer** | **2.6%** | 0.05 | 77 | CoordinatingAgent (2.6%) |
| **EnergyStarved** | **0.0%** | 0.00 | 77 | All fail |

**Mission Categories**:
- ✅ **Resource Collection (Base/Classic/Spread)**: 41-49% success, best with SimpleBaseline single-agent
- ✅ **Oxygen Bottleneck**: 37.7% success, demonstrates resource prioritization
- 🔶 **Medium Extractor Hubs (30/50/70)**: 22-28% success, scales moderately
- 🔴 **Large Extractor Hubs (80/100)**: 6-14% success, navigation/exploration fails
- 🔴 **Coordination-Required (GoTogether, DivideAndConquer)**: 2-25% success, needs improvement
- 🔴 **Complex Constraints (EnergyStarved, SingleUseSwarm, CollectFar)**: 0-5% success, largely unsolved

---

## Observations & Recommendations

### What Works Well ✅

1. **Single-agent baseline is solid**: 35.7% success on non-clipped missions, with 85.7% on resource collection missions
2. **Unclipping logic functional**: UnclippingAgent achieves 50% success on oxygen/silicon clipping (single-agent)
3. **Multi-agent coordination exists**: CoordinatingAgent shows coordination (GoTogether: 33.3%, better than 0% for single-agent)
4. **Resource collection missions**: All agents perform well on straightforward gather-assemble-deliver loops
5. **Mouth coordination works**: Agents spread around extractors/assemblers instead of queuing

### Critical Issues 🔴

1. **Multi-agent unclipping fails**: CoordinatingAgent only achieves 8-14% on clipped difficulties vs 50% for single-agent UnclippingAgent
   - **Root cause**: Multi-agent coordination of unclipping resources (who gathers decoder parts, who unclips) is not effectively implemented
   - **Recommendation**: Implement explicit unclipping coordination (assign one agent to unclip, others to gather other resources)

2. **Carbon/germanium unclipping broken**: 0-7% success across all agents
   - **Root cause**: Unknown - needs investigation of extractor types, recipes, or pathfinding issues
   - **Recommendation**: Debug carbon/germanium unclipping logic specifically

3. **Brutal difficulty unsolved**: 0% success across all agents, all missions
   - **Root cause**: Likely extreme resource scarcity, energy constraints, or time pressure
   - **Recommendation**: Profile `brutal` missions to understand specific failure modes

4. **Large maps fail**: ExtractorHub100 only 6.5% success
   - **Root cause**: Exploration inefficiency, timeout before finding all resources
   - **Recommendation**: Improve frontier-based exploration, consider map-size-aware exploration strategies

5. **Agent scaling degrades performance**: 2 agents (22.2%) → 4 agents (19.2%) → 8 agents (13.2%)
   - **Root cause**: Resource contention, collision, and coordination overhead
   - **Recommendation**: Improve collision avoidance, implement better resource reservation/assignment

6. **EnergyStarved mission: 0% success**
   - **Root cause**: Energy management logic insufficient for low-regen environments
   - **Recommendation**: Implement preemptive recharging based on distance-to-charger calculations

7. **DivideAndConquer: 2.6% success**
   - **Root cause**: Agents don't partition regions effectively
   - **Recommendation**: Implement explicit region assignment for multi-agent scenarios

### Improvements for Next Iteration 🔧

**High Priority**:
1. Fix multi-agent unclipping coordination (critical blocker for clipped multi-agent missions)
2. Debug carbon/germanium unclipping failures
3. Improve large-map exploration efficiency
4. Add preemptive energy management for EnergyStarved

**Medium Priority**:
5. Implement region-partitioning for DivideAndConquer
6. Improve collision avoidance and unstick logic
7. Add explicit single-use extractor tracking and assignment

**Low Priority**:
8. Profile and fix `brutal` difficulty
9. Optimize coordination overhead to reduce performance degradation at higher agent counts
10. Add better mouth selection logic for heavily contended stations

---

## Quick Play Commands

Test any mission locally:

```bash
# SimpleBaseline (single agent, no clipping)
uv run cogames play --mission evals.<mission_name> -p simple_baseline --cogs 1

# UnclippingAgent (single agent, with clipping)
uv run cogames play --mission evals.<mission_name> -p unclipping --cogs 1

# CoordinatingAgent (multi-agent, with clipping)
uv run cogames play --mission evals.<mission_name> -p coordinating --cogs 4
```

**Example missions**:
```bash
# Best performers
uv run cogames play --mission evals.collect_resources_classic -p simple_baseline --cogs 1
uv run cogames play --mission evals.oxygen_bottleneck -p simple_baseline --cogs 1

# Test unclipping (single-agent)
uv run cogames play --mission evals.extractor_hub_30 -p unclipping --cogs 1 --difficulty clipped_oxygen

# Test coordination (multi-agent)
uv run cogames play --mission evals.go_together -p coordinating --cogs 4
uv run cogames play --mission evals.collect_resources_spread -p coordinating --cogs 4

# Challenge missions (currently failing)
uv run cogames play --mission evals.energy_starved -p coordinating --cogs 4
uv run cogames play --mission evals.divide_and_conquer -p coordinating --cogs 4
uv run cogames play --mission evals.extractor_hub_100 -p coordinating --cogs 8
```

---

## Evaluation Reproduction

```bash
# Run full evaluation suite (1,078 tests, ~30-45 minutes)
cd /Users/daphnedemekas/Desktop/metta

# All three agents
uv run python packages/cogames/scripts/evaluate_scripted_agents.py \
  --steps 1000 \
  --output eval_results_complete.json

# Individual agents
uv run python packages/cogames/scripts/evaluate_scripted_agents.py \
  --agent simple \
  --steps 1000 \
  --output eval_results_simple.json

uv run python packages/cogames/scripts/evaluate_scripted_agents.py \
  --agent unclipping \
  --steps 1000 \
  --output eval_results_unclipping.json

uv run python packages/cogames/scripts/evaluate_scripted_agents.py \
  --agent coordinating \
  --steps 1000 \
  --output eval_results_coordinating.json
```

**Results Files**:
- `eval_results_baseline_agents.json` - Original run (all agents, non-clipped for CoordinatingAgent)
- `eval_results_coordinating_clipped.json` - CoordinatingAgent on clipped difficulties only
- Combined total: 1,078 test configurations

---

## Agent Architecture Summary

### SimpleBaselineAgent
- **Phases**: GATHER → ASSEMBLE → DELIVER → RECHARGE
- **Exploration**: Frontier-based with target persistence
- **Key Features**: Opportunistic gathering, goal-driven phase transitions, mouth coordination (N/S/E/W adjacency)
- **Limitations**: Single-agent only, no unclipping, no explicit coordination

### UnclippingAgent (extends SimpleBaseline)
- **Added Phases**: CRAFT_UNCLIP → UNCLIP
- **Key Features**: Recognizes clipped extractors, crafts decoder/resonator/modulator/scrambler, unclips extractors
- **Limitations**: Single-agent only, carbon/germanium unclipping unreliable

### CoordinatingAgent (extends UnclippingAgent)
- **Key Features**: Multi-agent, spreads agents around stations (mouth selection), unstick mechanism (random motion)
- **Limitations**: Multi-agent unclipping coordination poor, performance degrades with more agents, resource contention

---

**End of Report**
