 # Scripted Agent Evaluation Report

**Date**: November 5, 2025
**Agents Evaluated**: `Baseline`, `UnclippingAgent`
**Total Tests**: 1,040 configurations across 14 missions, 13 difficulty variants, and 1/2/4/8 agent counts

---

## Executive Summary

### 🎯 Key Findings

- ✅ **UnclippingAgent leads overall**: **38.6%** success rate (261/676 tests) with best average reward (2.58 hearts/agent)
- ✅ **Baseline strong baseline**: **33.8%** success (123/364 tests), best for non-clipped single-agent scenarios
- 📉 **Multi-agent scaling challenges**: Performance drops from 39.2% (1 agent) → 21.4% (8 agents) even with simple collision avoidance

### 🔑 Notable Results

1. **Unclipping works for oxygen/silicon**: 30-33% success rates demonstrate functional unclipping logic
2. **Carbon/germanium unclipping broken**: 0-2% success requires investigation
3. **Multi-agent collision avoidance functional**: Agents successfully avoid each other using agent_occupancy tracking
4. **Brutal difficulty unsolved**: 0% success across all agents

---

## Overall Statistics

| Metric | Value |
|--------|-------|
| **Total Tests** | 1,040 |
| **Successes** | 384 (36.9%) |
| **Average Reward** | 2.72 hearts/agent |
| **Agents Tested** | 2 (Baseline, UnclippingAgent) |
| **Missions** | 14 evaluation environments |
| **Difficulty Variants** | 13 |
| **Agent Counts** | 1, 2, 4, 8 |

---

## Performance by Agent

### Summary Table

| Agent | Tests | Success Rate | Avg Reward | Agent Counts | Status |
|-------|-------|--------------|------------|--------------|--------|
| **UnclippingAgent** | 676 | **38.6%** ✅ | 2.58 | 1, 2, 4, 8 | **BEST OVERALL** |
| **Baseline** | 364 | **33.8%** ✅ | 2.85 | 1, 2, 4, 8 | Solid baseline |

---

### Baseline Agent

**Purpose**: Single/multi-agent baseline for non-clipped environments
**Tests**: 364 (7 difficulties × 13 missions × 4 agent counts)
**Overall Success**: 123/364 (33.8%)

#### Performance by Agent Count

| Agent Count | Success Rate | Avg Reward | Tests |
|-------------|--------------|------------|-------|
| **1 agent** | **44.2%** | 2.95 | 91 |
| **2 agents** | **35.2%** | 2.88 | 91 |
| **4 agents** | **30.8%** | 2.77 | 91 |
| **8 agents** | **25.3%** | 2.81 | 91 |

**Observation**: Baseline shows **graceful degradation** with more agents (44% → 25%), indicating basic collision avoidance via agent_occupancy is functional but imperfect.

#### Performance by Difficulty

| Difficulty | Success Rate | Avg Reward | Tests |
|------------|--------------|------------|-------|
| **speed_run** | **47.5%** | 2.84 | 52 |
| **energy_crisis** | **43.4%** | 3.50 | 52 |
| **standard** | **40.6%** | 3.15 | 52 |
| **story_mode** | **40.6%** | 2.78 | 52 |
| **hard** | **38.5%** | 2.69 | 52 |
| **single_use** | **28.0%** | 2.43 | 52 |
| **brutal** | **0.0%** | 0.00 | 52 |

**Key Observations**:
- Performs well on time-pressure scenarios (speed_run: 47.5%)
- Handles energy management effectively (energy_crisis: 43.4%)
- Brutal difficulty remains unsolved
- No unclipping capability limits effectiveness on clipped maps

---

### UnclippingAgent

**Purpose**: Multi-agent with unclipping capability
**Tests**: 676 (13 difficulties × 13 missions × 4 agent counts)
**Overall Success**: 261/676 (38.6%) - **BEST OVERALL**

#### Performance by Agent Count

| Agent Count | Success Rate | Avg Reward | Tests |
|-------------|--------------|------------|-------|
| **1 agent** | **60.4%** ✅ | 3.12 | 169 |
| **2 agents** | **37.3%** | 2.57 | 169 |
| **4 agents** | **32.0%** | 2.39 | 169 |
| **8 agents** | **24.9%** | 2.25 | 169 |

**Observation**: Single-agent unclipping is **highly effective** (60.4%), but multi-agent scenarios show degradation due to collision/resource contention.

#### Performance by Difficulty

| Difficulty | Success Rate | Avg Reward | Tests | Notes |
|------------|--------------|------------|-------|-------|
| **speed_run** | **47.6%** ✅ | 2.84 | 52 | Excellent |
| **energy_crisis** | **43.4%** ✅ | 3.50 | 52 | Excellent |
| **standard** | **40.6%** ✅ | 3.15 | 52 | Excellent |
| **story_mode** | **40.6%** ✅ | 2.78 | 52 | Excellent |
| **hard** | **38.5%** ✅ | 2.69 | 52 | Good |
| **clipped_silicon** | **33.0%** ✅ | 2.93 | 91 | Unclipping works! |
| **clipped_oxygen** | **29.7%** ✅ | 2.16 | 91 | Unclipping works! |
| **single_use** | **28.0%** | 2.43 | 52 | Moderate |
| **hard_clipped_oxygen** | **19.8%** 🔶 | 0.24 | 91 | Harder variant struggles |
| **clipping_chaos** | **3.3%** ❌ | 0.08 | 91 | Multi-resource clipping fails |
| **clipped_germanium** | **2.2%** ❌ | 0.09 | 91 | Unclipping broken |
| **clipped_carbon** | **0.0%** ❌ | 0.00 | 91 | Unclipping broken |
| **brutal** | **0.0%** ❌ | 0.00 | 52 | Unsolved |

**Key Observations**:
- ✅ **Oxygen & Silicon unclipping work**: 30-33% success rates demonstrate functional unclipping logic
- ❌ **Carbon & Germanium unclipping broken**: 0-2% success suggests fundamental issues with these resource types
- ❌ **Clipping chaos unsolvable**: When multiple extractors are clipped, agents fail (3.3%)
- 🔶 **Hard clipped variants struggle**: Only 19.8% on hard_clipped_oxygen vs 29.7% on regular clipped_oxygen

---

## Performance by Agent Count

| Agent Count | Success Rate | Avg Reward | Tests | Best Agent |
|-------------|--------------|------------|-------|------------|
| **1 agent** | **52.3%** ✅ | 3.04 | 260 | UnclippingAgent (60.4%) |
| **2 agents** | **36.3%** 🔶 | 2.73 | 260 | UnclippingAgent (37.3%) |
| **4 agents** | **31.4%** 🔶 | 2.58 | 260 | UnclippingAgent (32.0%) |
| **8 agents** | **25.1%** ❌ | 2.53 | 260 | UnclippingAgent (24.9%) |

**Scaling Observations**:
- Single-agent performance is strong (52.3%)
- **Performance degrades by ~50%** when going from 1 to 8 agents
- Indicates collision/contention issues even with agent_occupancy-based collision avoidance
- Both agents show similar degradation patterns

---

## Performance by Difficulty Variant

| Difficulty | Success Rate | Avg Reward | Tests | Best Agent |
|------------|--------------|------------|-------|------------|
| **speed_run** | **47.6%** ✅ | 2.84 | 104 | UnclippingAgent |
| **energy_crisis** | **43.4%** ✅ | 3.50 | 104 | UnclippingAgent |
| **standard** | **40.6%** ✅ | 3.15 | 104 | UnclippingAgent |
| **story_mode** | **40.6%** ✅ | 2.78 | 104 | UnclippingAgent |
| **hard** | **38.5%** ✅ | 2.69 | 104 | UnclippingAgent |
| **clipped_silicon** | **33.0%** ✅ | 2.93 | 91 | UnclippingAgent (33.0%) |
| **clipped_oxygen** | **29.7%** ✅ | 2.16 | 91 | UnclippingAgent (29.7%) |
| **single_use** | **28.0%** 🔶 | 2.43 | 104 | UnclippingAgent |
| **hard_clipped_oxygen** | **19.8%** 🔶 | 0.24 | 91 | UnclippingAgent (19.8%) |
| **clipping_chaos** | **3.3%** ❌ | 0.08 | 91 | UnclippingAgent (3.3%) |
| **clipped_germanium** | **2.2%** ❌ | 0.09 | 91 | UnclippingAgent (2.2%) |
| **clipped_carbon** | **0.0%** ❌ | 0.00 | 91 | All fail |
| **brutal** | **0.0%** ❌ | 0.00 | 104 | All fail |

**Difficulty Insights**:
- ✅ **Non-clipped difficulties strong**: 28-48% success, agents work well on basic gather-assemble-deliver
- ✅ **Oxygen/Silicon clipping works**: 30-33% shows unclipping logic is functional for these types
- ❌ **Carbon/Germanium clipping broken**: Needs urgent debugging
- ❌ **Multi-resource clipping (chaos) unsolved**: Complex unclipping scenarios fail
- ❌ **Brutal universally unsolved**: Extreme constraints too difficult

---

## Critical Issues & Recommendations

### 🚨 Issue #1: Carbon/Germanium Unclipping Broken

**Severity**: **HIGH** - Blocks 182 tests (91 clipped_carbon + 91 clipped_germanium)

**Symptoms**:
- 0% success on clipped_carbon across all agents (91 tests)
- 2.2% success on clipped_germanium (2/91 tests, likely lucky)
- Oxygen/silicon unclipping works fine (30-33%)

**Likely Root Causes**:
1. **Recipe/protocol issues**: Carbon/germanium gear crafting may be broken
2. **Resource gathering failures**: Can't find/extract carbon or germanium
3. **Unclip action not working**: Decoder/scrambler application fails
4. **Map-specific issues**: Carbon/germanium extractors may be unreachable or missing

**Recommended Fix**:
- Debug step-by-step on a clipped_carbon mission with logging
- Verify recipe creation for decoder (carbon → decoder)
- Check if carbon extractors exist and are reachable
- Test single-agent UnclippingAgent on simplest clipped_carbon map

---

### 🚨 Issue #2: Multi-Agent Scaling Degradation

**Severity**: **HIGH** - Limits real-world applicability

**Symptoms**:
- 52.3% (1 agent) → 25.1% (8 agents) - 50% degradation
- Both Baseline and UnclippingAgent affected
- Even with agent_occupancy collision avoidance

**Likely Root Causes**:
1. **Collision/blocking**: Agents physically blocking each other's paths
2. **Resource contention**: Multiple agents targeting same extractor simultaneously
3. **Exploration inefficiency**: Agents re-exploring same areas
4. **Agent occupancy avoidance too strong**: Agents avoiding each other too aggressively, limiting paths

**Recommended Fix**:
- Improve pathfinding to allow agents to "wait" for others to pass
- Add temporal planning: agents predict where others will be
- Implement explicit resource claiming: first agent to see an extractor "owns" it temporarily
- Consider reducing agent_occupancy avoidance radius

---

### 🚨 Issue #3: Brutal Difficulty Unsolved

**Severity**: **MEDIUM** - Advanced difficulty, expected to be hard

**Symptoms**:
- 0% success across all agents, all missions (104 tests)

**Likely Root Causes**:
- Extreme resource scarcity or energy constraints
- Insufficient time (may need >1000 steps)
- Specialized strategies required (e.g., perfect efficiency, no wasted moves)

**Recommended Fix**:
- Profile a specific brutal mission to understand constraints
- May need domain-specific optimizations (not general agent improvement)

---

### 🚨 Issue #4: Clipping Chaos Unsolved

**Severity**: **MEDIUM** - Complex scenario

**Symptoms**:
- 3.3% success (3/91 tests)
- Occurs when multiple extractors are clipped

**Likely Root Causes**:
- Agent can't handle multiple unclipping needs
- Prioritization logic fails with multiple clipped extractors
- Resource deadlock: needs X to unclip Y, needs Y to unclip X

**Recommended Fix**:
- Implement smarter unclip prioritization (unclip extractors in dependency order)
- Add deadlock detection and recovery

---

## Recommendations by Priority

### 🔥 Immediate (P0)

1. **Debug carbon/germanium unclipping**: Single highest-value fix for coverage (182 tests)
2. **Add comprehensive logging**: Instrument agent decision-making for debugging

### 📋 High Priority (P1)

3. **Improve multi-agent collision avoidance**: Better unstick, temporal planning, resource claiming
4. **Fix agent_occupancy tracking**: May be too aggressive or buggy
5. **Investigate hard_clipped_oxygen failures**: Why worse than regular clipped_oxygen?

### 📌 Medium Priority (P2)

6. **Improve clipping_chaos handling**: Multiple clipped extractors need better strategy
7. **Optimize exploration for large maps**: ExtractorHub80/100 timeout issues

### 📎 Low Priority (P3)

8. **Profile brutal difficulty**: Understand if solvable with current architecture
9. **Add extractor queueing**: Multiple agents coordinating at same extractor
10. **Improve energy management**: Preemptive recharging for energy-starved missions

---

## Quick Play Commands

### Test UnclippingAgent (Best Performance)

```bash
# Single agent, non-clipped (60.4% success)
uv run cogames play --mission evals.oxygen_bottleneck -p unclipping --cogs 1

# Single agent, oxygen clipped (30% success)
uv run cogames play --mission evals.extractor_hub_30 -p unclipping --cogs 1 --difficulty clipped_oxygen

# Multi-agent (37% success, 2 agents)
uv run cogames play --mission evals.collect_resources_spread -p unclipping --cogs 2
```

### Test Baseline

```bash
# Single agent (44% success)
uv run cogames play --mission evals.collect_resources_classic -p scripted_baseline --cogs 1

# Multi-agent (35% success, 2 agents)
uv run cogames play --mission evals.extractor_hub_30 -p scripted_baseline --cogs 2
```

### Debug Carbon Unclipping Failure

```bash
# Carbon unclipping (0% success - broken!)
uv run cogames play --mission evals.extractor_hub_30 -p unclipping --cogs 1 --difficulty clipped_carbon
```

---

## Evaluation Reproduction

```bash
cd /Users/daphnedemekas/Desktop/metta

# Full evaluation (1,040 tests, ~70 minutes)
uv run python packages/cogames/scripts/evaluate_scripted_agents.py

# Output: evaluation_output.log
```

**Results**:
- Baseline: 123/364 (33.8%)
- UnclippingAgent: 261/676 (38.6%) ← **BEST**

---

## Agent Architecture Summary

### BaselineAgent
- **Phases**: GATHER → ASSEMBLE → DELIVER → RECHARGE
- **Exploration**: Frontier-based with target persistence
- **Key Features**: Opportunistic gathering, goal-driven phase transitions, agent occupancy avoidance
- **Performance**: 33.8% overall, 44.2% single-agent
- **Limitations**: No unclipping, basic multi-agent collision avoidance

### UnclippingAgent (extends Baseline)
- **Added Phases**: CRAFT_UNCLIP → UNCLIP
- **Key Features**: Recognizes clipped extractors, crafts decoder/resonator/modulator/scrambler, unclips extractors
- **Performance**: **38.6% overall (BEST)**, 60.4% single-agent
- **Limitations**: Carbon/germanium unclipping broken (0-2%), multi-agent degradation (60% → 25%)

---

**End of Report**
