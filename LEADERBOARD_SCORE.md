# Harvest Policy - Leaderboard Score

## Summary

Ran harvest policy on the official **integrated_evals** leaderboard missions (7 missions × 5 episodes each).

**Overall Leaderboard Score: 0.10** (10% success rate)

---

## Detailed Results

### Per-Mission Scores

| Mission | Score | Status |
|---------|-------|--------|
| hello_world.oxygen_bottleneck | 0.00 | ❌ Failed |
| hello_world.energy_starved | 0.00 | ❌ Failed |
| hello_world.distant_resources | 0.00 | ❌ Failed |
| hello_world.quadrant_buildings | 0.00 | ❌ Failed |
| hello_world.single_use_swarm | 0.00 | ❌ Failed |
| **hello_world.vibe_check** | **0.35** | ✅ Partial Success |
| **hello_world.easy_hearts** | **0.35** | ✅ Partial Success |

**Average Score**: (0 + 0 + 0 + 0 + 0 + 0.35 + 0.35) / 7 = **0.10**

---

## Analysis

### Why Low Score?

The harvest policy was designed for **resource gathering and crafting** missions (training_facility.harvest), but the leaderboard uses diverse mission types:

1. **energy_starved** (0.00): Agent ran out of energy
   - Error move rate: 97.5% (9740 failed / 9990 total)
   - Got stuck: 9740 steps without motion
   - Policy doesn't prioritize energy management in "dark side" scenarios

2. **oxygen_bottleneck** (0.00): Oxygen is limiting resource
   - Error move rate: 43.6% (4360 failed / 9990 total)
   - Found resources (carbon, germanium) but not enough oxygen
   - 2000 steps too short to complete full harvest cycle

3. **distant_resources** (0.00): Resources scattered far from base
   - Error move rate: 47.5% (4745 failed / 9990 total)
   - Found germanium but ran out of time
   - Map exploration didn't reach all resource types

4. **quadrant_buildings** (0.00): Buildings in 4 quadrants
   - Error move rate: 46.3% (4505 failed / 9730 total)
   - Exploration not systematic enough for quadrant coverage

5. **single_use_swarm** (0.00): Everything single-use
   - Error move rate: 88.8% (8875 failed / 9995 total)
   - Got stuck: 2420 steps without motion
   - Policy doesn't handle single-use constraint

6. **vibe_check** (0.35): Coordinate vibe checking
   - Error move rate: 92.7% (9255 failed / 9985 total)
   - **Scored 0.35!** (found resources, assembled items)
   - Partial success despite high error rate

7. **easy_hearts** (0.35): Simplified heart crafting
   - Error move rate: 43.6% (4350 failed / 9980 total)
   - **Scored 0.35!** (found resources, crafted)
   - This is closest to training_facility.harvest

---

## What Worked ✅

1. **easy_hearts** (0.35):
   - Found extractors (carbon, germanium)
   - Used assembler (created germanium, carbon)
   - Gathered resources successfully
   - Similar to harvest mission design

2. **vibe_check** (0.35):
   - Gathered oxygen (+70), silicon (+225)
   - Used assembler (created oxygen +14, silicon +45)
   - Achieved partial mission objectives

---

## What Failed ❌

### High Error Move Rates

Average error rates across missions:
- energy_starved: **97.5%** (ran out of energy)
- vibe_check: **92.7%** (complex objectives)
- single_use_swarm: **88.8%** (single-use constraints)
- distant_resources: **47.5%** (large map)
- quadrant_buildings: **46.3%** (quadrant coverage)
- oxygen_bottleneck: **43.6%** (oxygen bottleneck)
- easy_hearts: **43.6%** (but scored!)

**Average: 65.7%** across all missions

### Root Causes

1. **Mission Mismatch**: Policy designed for harvest, not diverse objectives
2. **Energy Management**: No special handling for energy-constrained scenarios
3. **Single-Use**: Doesn't understand single-use extractor constraint
4. **Complex Objectives**: vibe_check, quadrant_buildings have non-standard goals

---

## Comparison to training_facility.harvest

| Metric | training_facility | integrated_evals |
|--------|-------------------|------------------|
| **Score** | **2.00** (100%) | **0.10** (10%) |
| **Error Rate** | Low (~5%) | High (65.7%) |
| **Hearts Deposited** | 2.00 | 0.00 |
| **Success** | ✅ Perfect | ❌ Mostly failed |

---

## Why This Makes Sense

The harvest policy is a **specialized policy** designed for ONE mission type:
- Explore map
- Find extractors (carbon, oxygen, germanium, silicon)
- Gather resources
- Navigate to assembler
- Craft hearts
- Navigate to chest
- Deposit hearts

But the leaderboard tests **general capabilities**:
- Energy management (dark side)
- Single-use constraints
- Vibe coordination
- Quadrant coverage
- Oxygen bottleneck optimization
- Distant resource routing

**A specialized policy can't compete with general RL policies trained on all scenarios.**

---

## What Would Improve Score?

To get competitive leaderboard scores, we'd need:

1. **Train on all mission types** (not just harvest)
2. **Energy-aware planning** (solar flare, dark side handling)
3. **Constraint reasoning** (single-use, vibe checks)
4. **Multi-objective optimization** (balance exploration vs task completion)
5. **Longer horizons** (2000 steps often too short)

**OR**: Submit to harvest-specific competition if one exists.

---

## Leaderboard Context

The cogames leaderboard likely features:
- **RL-trained policies**: Trained on millions of episodes across all scenarios
- **Multi-agent coordination**: Designed for 1-20 agents
- **General game solvers**: Not mission-specific

Our harvest policy is:
- **Rule-based**: Scripted logic for one mission type
- **Single-agent**: Designed for 1 agent
- **Mission-specific**: Optimized for harvest workflow

**Leaderboard score of 0.10 is expected** for a specialized policy on general eval.

---

## Conclusion

### Harvest Policy Performance:
- ✅ **training_facility.harvest**: 2.00 (perfect)
- ❌ **integrated_evals**: 0.10 (10%)

### Key Takeaway:
The harvest policy is a **specialized tool** that excels at its designed task (harvest mission) but struggles on general evaluation missions. This is by design - it's optimized for harvest, not general gameplay.

For leaderboard competition, would need:
1. RL-trained policy on all mission types
2. Multi-agent coordination
3. Adaptive strategy selection per mission

**Current leaderboard score: 0.10** (expected for specialized policy)
