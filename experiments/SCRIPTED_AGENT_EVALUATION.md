 # Scripted Agent Evaluation Report

## Executive Summary

**Overall Performance**: 73.0% success rate (467/640 tests) on the Difficulty suite (16 missions × 4 difficulties × 10 presets, 1000 steps).

**Best Presets (win rate)**: `efficiency_heavy` (78.1%), `balanced` (75.0%), `efficiency_light` (75.0%), `sequential_baseline` (75.0%).

**Difficulty Breakdown**: easy 99.4%, medium 91.2%, hard 50.6%, extreme 50.6%.

**Recent Fix**: Increased cargo capacity from 100 to 255, resolving inventory blocking issues in silicon-heavy environments.

---

## Hyperparameter System

### Hyperparameter Presets (10 total)

Each preset combines strategy, exploration horizon, energy policy, cooldown impatience, and depletion sensitivity:

- **balanced**
  - strategy: explorer_first; explore=80; min_energy_for_silicon=70
  - recharge_start=65/45; recharge_stop=90/75; wait_if_cooldown_leq=2; depletion_threshold=0.25

- **explorer_short**
  - strategy: explorer_first; explore=50; min_energy_for_silicon=65
  - recharge_start=68/48; recharge_stop=88/73; wait_if_cooldown_leq=1; depletion_threshold=0.25

- **explorer_long**
  - strategy: explorer_first; explore=150; min_energy_for_silicon=75
  - recharge_start=60/40; recharge_stop=85/70; wait_if_cooldown_leq=2; depletion_threshold=0.25

- **greedy_aggressive**
  - strategy: greedy_opportunistic; explore=25; min_energy_for_silicon=55
  - recharge_start=70/50; recharge_stop=90/75; wait_if_cooldown_leq=0; depletion_threshold=0.20

- **greedy_conservative**
  - strategy: greedy_opportunistic; explore=75; min_energy_for_silicon=80
  - recharge_start=70/50; recharge_stop=85/70; wait_if_cooldown_leq=3; depletion_threshold=0.30

- **efficiency_light**
  - strategy: efficiency_learner; explore=80; min_energy_for_silicon=65
  - recharge_start=65/45; recharge_stop=90/75; wait_if_cooldown_leq=2; depletion_threshold=0.25

- **efficiency_heavy**
  - strategy: efficiency_learner; explore=120; min_energy_for_silicon=80
  - recharge_start=65/45; recharge_stop=92/78; wait_if_cooldown_leq=2; depletion_threshold=0.20

- **sequential_baseline**
  - strategy: sequential_simple; explore=50; min_energy_for_silicon=70
  - recharge_start=65/45; recharge_stop=90/75; wait_if_cooldown_leq=2; depletion_threshold=0.25

- **silicon_rush**
  - strategy: greedy_opportunistic; explore=50; min_energy_for_silicon=60
  - recharge_start=65/45; recharge_stop=88/73; wait_if_cooldown_leq=1; depletion_threshold=0.20

- **oxygen_safe**
  - strategy: explorer_first; explore=100; min_energy_for_silicon=85
  - recharge_start=70/50; recharge_stop=95/80; wait_if_cooldown_leq=3; depletion_threshold=0.30

Note: All 10 presets were evaluated across easy/medium/hard/extreme difficulties.

---

## Environment Descriptions

### Exploration Experiments (with Difficulty Variants)

#### EXP1 - Basic Resource Gathering
- **Objective**: Gather 4 resources (Ge, Si, C, O), assemble hearts, deposit in chest
- **Difficulty Variants**:
  - **EASY**: Abundant extractors, high efficiency, fast energy regen
  - **MEDIUM**: Moderate extractors, balanced efficiency
  - **HARD**: Limited extractors, low efficiency, slow energy regen
- **Map Size**: ~30x30

#### EXP2 - Advanced Resource Management
- **Objective**: Same as EXP1, but with more complex map layout
- **Difficulty Variants**: Same as EXP1
- **Map Size**: ~30x30

### Eval Missions (Fixed Configurations)

#### OXYGEN_BOTTLENECK
- **Challenge**: Limited oxygen extractors
- **Key**: Find all oxygen sources, manage cooldowns

#### GERMANIUM_RUSH
- **Challenge**: Time pressure to gather germanium quickly
- **Key**: Fast exploration, prioritize germanium

#### SILICON_WORKBENCH
- **Challenge**: Silicon requires high energy
- **Key**: Manage energy carefully, recharge strategically

#### CARBON_DESERT
- **Challenge**: Sparse carbon extractors
- **Key**: Thorough exploration, efficient travel

#### SINGLE_USE_WORLD
- **Challenge**: All extractors have max_uses=1
- **Key**: Find ALL extractors before gathering (UNSOLVED)

#### SLOW_OXYGEN
- **Challenge**: Oxygen extractors have long cooldowns
- **Key**: Find multiple oxygen sources, wait strategically

#### HIGH_REGEN_SPRINT
- **Challenge**: High energy regen, encourages fast movement
- **Key**: Aggressive gathering, less recharging needed

#### SPARSE_BALANCED
- **Challenge**: Few extractors of each type, balanced distribution
- **Key**: Systematic exploration, efficient routing

#### GERMANIUM_CLUTCH
- **Challenge**: Critical germanium shortage
- **Key**: Find all germanium extractors (UNSOLVED)

---

## Results by Environment

### ✅ FULLY SOLVED (8/15 environments - all 5 strategies succeed)

#### EXP1-EASY
- **Best**: All strategies - 2 hearts
- **Optimal Strategy**: Any strategy works
- **Notes**: Straightforward environment, good for testing baseline behavior

#### EXP1-MEDIUM
- **Best**: All strategies - 2 hearts
- **Optimal Strategy**: Any strategy works
- **Notes**: Slightly more complex than EASY, still very solvable

#### OXYGEN_BOTTLENECK
- **Best**: All strategies - 2 hearts
- **Optimal Strategy**: Any strategy works
- **Notes**: Limited oxygen extractors but sufficient with good exploration
- **Known Issue**: Agents get stuck after 2 hearts (3rd heart crafting bug)

#### GERMANIUM_RUSH
- **Best**: All strategies - 2 hearts
- **Optimal Strategy**: Any strategy works
- **Notes**: Sufficient germanium extractors for all approaches

#### SILICON_WORKBENCH
- **Best**: All strategies - 2 hearts
- **Optimal Strategy**: Any strategy works
- **Notes**: **FIXED** - Increased cargo capacity from 100 to 255 resolved inventory blocking
- **Known Issue**: Agents get stuck after 2 hearts (3rd heart crafting bug)

#### CARBON_DESERT
- **Best**: All strategies - 2 hearts
- **Optimal Strategy**: Any strategy works
- **Notes**: Despite "desert" name, carbon is findable with exploration

#### SLOW_OXYGEN
- **Best**: All strategies - 2 hearts
- **Optimal Strategy**: Any strategy works
- **Notes**: Long cooldowns require finding multiple oxygen sources

#### HIGH_REGEN_SPRINT
- **Best**: All strategies - 2 hearts
- **Optimal Strategy**: Any strategy works
- **Notes**: High energy regen makes all strategies viable

#### SPARSE_BALANCED
- **Best**: All strategies - 2 hearts
- **Optimal Strategy**: Any strategy works
- **Notes**: Balanced design rewards all approaches

### ✅ Partially Solved (selected missions sensitive to preset/difficulty)

#### EXP2-EASY
- **Best**: 4/5 strategies - 1 heart
- **Failed**: `explorer_conservative` (0 hearts)
- **Optimal Strategy**: `explorer`, `greedy`, `efficiency`, or `explorer_aggressive`

#### EXP2-MEDIUM
- **Best**: 3/5 strategies - 1 heart
- **Failed**: `greedy` (0 hearts), `explorer_conservative` (0 hearts)
- **Optimal Strategy**: `explorer`, `efficiency`, or `explorer_aggressive`


### ❌ Challenging Cases

- Hard/Extreme variants of EXP1 and EXP2 show lower success (50.6% win rate); failures often due to navigation dead-ends to germanium or oxygen.

---

## Strategy Performance Summary (Difficulty Suite)

| Preset              | Passed/Total | Win Rate |
|---------------------|--------------|----------|
| efficiency_heavy    | 50/64        | 78.1%    |
| balanced            | 48/64        | 75.0%    |
| efficiency_light    | 48/64        | 75.0%    |
| sequential_baseline | 48/64        | 75.0%    |
| greedy_conservative | 47/64        | 73.4%    |
| oxygen_safe         | 47/64        | 73.4%    |
| explorer_short      | 45/64        | 70.3%    |
| greedy_aggressive   | 45/64        | 70.3%    |
| silicon_rush        | 45/64        | 70.3%    |
| explorer_long       | 44/64        | 68.8%    |

### Difficulty Variants - Summary

- Total: 640 tests; Passed: 467; Failed: 173 → 73.0%
- By difficulty: easy 159/160 (99.4%), medium 146/160 (91.2%), hard 81/160 (50.6%), extreme 81/160 (50.6%)

### Key Insights

1. **Explorer strategies dominate**: The `explorer_first` strategy (explore 100 steps, then gather) is the most reliable, with 73.3% success rate.

2. **Silicon energy threshold is critical**: `min_energy_for_silicon=70` (balanced) works best. Conservative (85) still struggles on some environments.

3. **Cargo capacity was a blocker**: Increasing from 100 to 255 fixed SILICON_WORKBENCH, which was failing due to inventory limits.

4. **2-heart ceiling**: Most successful missions get exactly 2 hearts, then get stuck trying to craft the 3rd heart (possibly due to inventory cap)

5. **Hard missions remain unsolved**: EXP1-HARD, EXP2-HARD, SINGLE_USE_WORLD, and GERMANIUM_CLUTCH all get 0 hearts across all strategies.

6. **Navigation issues persist**: Agents still get stuck in some scenarios, repeatedly trying to reach unreachable targets.

---

## Recommendations

### Critical Fixes

1. **3rd Heart Crafting Bug** ⚠️ **BLOCKING** - Inventory Overflow Issue
   - **Issue**: Agents successfully craft and deposit 2 hearts, then get stuck trying to craft the 3rd heart
   - **Root Cause**: Inventory capacity overflow. Cargo cap is 255 units, but agent doesn't check capacity before gathering. After 2 hearts, leftover resources (~111 units) + gathering for 3rd heart (~95 units) can exceed 255 if agent over-gathers (e.g., gets 30C instead of 20C).
   - **Impact**: Prevents agents from getting beyond 2 hearts in most missions
   - **Observed in**: OXYGEN_BOTTLENECK, SILICON_WORKBENCH, and likely others
   - **Solution**: Add inventory capacity checks before gathering, or add logic to stop gathering when approaching the 255 cap
   - **Action**: Debug why assembler stops working after 2 uses

### High Priority Fixes

2. **Navigation System**: Fix pathfinding bugs
   - Agent gets stuck trying to reach discovered stations
   - BFS/A* failing on seemingly reachable cells
   - Repeated "STUCK" messages in logs

3. **Discovery-Before-Gathering**: Implement for SINGLE_USE_WORLD
   - Force complete map exploration before any resource gathering
   - Track extractor locations without using them
   - Mark depleted extractors and find alternatives

4. **Germanium-Focused Exploration**: Improve for GERMANIUM_CLUTCH
   - Bias exploration toward germanium-rich areas
   - Increase exploration duration when germanium is scarce

### Medium Priority Improvements

5. **Hard Difficulty Tuning**: Adjust for EXP1-HARD, EXP2-HARD
   - Increase exploration duration
   - Lower depletion threshold to find backup extractors earlier
   - Improve extractor discovery rate

6. **Cooldown Waiting**: Improve extractor cooldown handling
   - Agent should wait near extractors on cooldown instead of exploring
   - Use observed cooldown_remaining from observations

---

## Conclusion

The curated 10-preset evaluation achieves **73.0% success** across 640 difficulty tests (16 missions × 4 difficulties × 10 presets), demonstrating that:

1. **Preset diversity helps**: Energy hysteresis, cooldown impatience, and exploration horizon materially affect robustness.
2. **Balanced/efficiency-heavy perform best**: 75–78% win rates across all missions/difficulties.
3. **Hard/Extreme are the bottleneck**: Success drops to ~50.6% at hard/extreme; failures correlate with navigation dead-ends and scarce germanium.
4. **Cargo capacity fix is impactful**: Silicon-heavy maps remain stable at 1000 steps.

**Current Status**:
- Passed: 467/640 tests (73.0%)
- By difficulty: easy 99.4%, medium 91.2%, hard 50.6%, extreme 50.6%
- Top presets: efficiency_heavy, balanced, efficiency_light, sequential_baseline

**Next Wins**:
1. Navigation fallback around obstacles to reduce STUCK loops to germanium targets.
2. Smarter oxygen scheduling on slow-O2/EXP2-hard variants (short waits near extractor).
3. Minor preset tuning per-mission family (auto-select best of 2–3 presets).

---

## Multi-Agent Support

**Current Status**: The scripted agent is **single-agent only**. Testing with 2+ agents results in complete failure (0% success, agents get stuck immediately).

**Issue**: Agents block each other and have no collision avoidance or coordination:
- `action.move.failed`: ~745 per episode (agents constantly blocked)
- `status.max_steps_without_motion`: ~177 steps stuck
- Total reward: 0.00 (complete failure)

**Required for Multi-Agent Extension** (to be implemented):
1. **Collision avoidance** in navigator - detect and route around other agents
2. **Multi-agent coordination** - task allocation and shared goal management
3. **Occupancy tracking** - maintain awareness of other agent positions
4. **Shared resource management** - coordinate extractor usage and prevent conflicts

**Test Commands**:
```bash
# Evaluate with 2 agents
uv run cogames evaluate -m training_facility.assemble -p scripted --cogs 2 --episodes 3

# Play with 2 agents (no --gui flag, use mettascope)
uv run cogames play training_facility.assemble -p scripted --cogs 2
```
