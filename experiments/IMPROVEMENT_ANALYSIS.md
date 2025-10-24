# Scripted Agent Improvement Analysis

**Based on**: Comprehensive evaluation of 10 experiments × 4 configs = 40 runs
**Current Performance**: 70% success rate (28/40)
**Date**: October 24, 2025

---

## 1. Performance Analysis

### Current Results Summary

| Metric | Value | Notes |
|--------|-------|-------|
| **Overall Success** | 28/40 (70%) | Strong baseline |
| **Working Experiments** | 7/10 | Exp 1,4,5,6,7,8,9 |
| **Failing Experiments** | 3/10 | Exp 2,3,10 |
| **Best Config** | aggressive | 1.70 avg reward |
| **Config Diversity** | LOW | All achieve same 7/10 success |

### Key Insight
**All 4 hyperparameter configurations achieve identical pass/fail results** - they succeed on the same 7 experiments and fail on the same 3. This suggests:
- Failures are due to **fundamental algorithm limitations**, not tuning
- Hyperparameters only affect **speed/efficiency** on solvable experiments
- Need **algorithmic improvements**, not just parameter tweaks

---

## 2. Critical Issues Analysis

### Issue #1: Navigation (Exp 2) - **HIGHEST PRIORITY**
**Symptom**: Agent collects all resources (C=20, O=20, G=6, Si=50) but fails to assemble
**Root Cause**: 80x80 maze with complex walls; BFS pathfinding is inefficient

**Evidence from diagnostics:**
```
Agent at (27, 34) → Assembler at (44, 44)
Manhattan distance: 27 tiles
Agent movements: (27,34)→(27,35)→(27,36)→(27,37)→(27,36)→(27,35) [backtracking!]
Result: Runs out of time/energy navigating maze
```

**Impact**: Complete failure on large complex maps
**Priority**: ⭐⭐⭐⭐⭐ (Blocks 1/10 experiments)

**Proposed Solutions:**
1. **A* Pathfinding** (Recommended)
   - Use Manhattan distance heuristic
   - Much faster than BFS on large maps
   - Finds optimal paths through mazes
   - Effort: ~2-3 hours implementation

2. **Path Caching**
   - Cache successful paths between key locations
   - Reuse paths for return trips
   - Effort: ~1 hour

3. **Early Pathfinding Validation**
   - Before committing to target, verify path exists
   - Reject targets with paths >80 steps on large maps
   - Effort: ~30 minutes

**Expected Improvement**: +10% success rate (Exp 2 solved)

---

### Issue #2: Resource Timing (Exp 3) - **HIGH PRIORITY**
**Symptom**: Stuck at 19/20 oxygen, full energy, can't progress
**Root Cause**: No waiting strategy for cooldowns; 75% efficiency = 15 O2/harvest

**Evidence:**
```
Steps 981-1000: Agent at full energy, oxygen=19 (need 20)
All oxygen extractors on 100-turn cooldown
Agent explores aimlessly instead of waiting
```

**Impact**: Fails on low-efficiency + cooldown scenarios
**Priority**: ⭐⭐⭐⭐ (Blocks 1/10 experiments)

**Proposed Solutions:**
1. **Cooldown Awareness** (Recommended)
   - Track extractor cooldown timers
   - Calculate when next harvest available
   - Wait near extractor if cooldown < 50 turns
   - Effort: ~2 hours

2. **Multi-Source Strategy**
   - Rotate between multiple extractors of same type
   - Already partially implemented in extractor memory
   - Needs better "wait for cooldown" vs "find alternative" logic
   - Effort: ~1 hour

3. **Partial Resource Collection**
   - If at 19/20 oxygen, go do other tasks and return
   - Requires better multi-objective planning
   - Effort: ~3-4 hours (complex)

**Expected Improvement**: +10% success rate (Exp 3 solved)

---

### Issue #3: Efficiency Extremes (Exp 10) - **MEDIUM PRIORITY**
**Symptom**: 50% efficiency + 80x80 map = slow resource collection
**Root Cause**: Combination of navigation + low efficiency

**Impact**: Edge case failure
**Priority**: ⭐⭐⭐ (Blocks 1/10, but hard to solve)

**Note**: Likely solved by fixing Issues #1 and #2

---

## 3. Hyperparameter Diversity Analysis

### Current Configurations

| Config | energy_buffer | prefer_nearby | cooldown_tolerance | Distinguishing Factor |
|--------|---------------|---------------|-------------------|----------------------|
| baseline | 20 | True | 20 | Balanced, safe |
| conservative | 30 | True | 10 | Extra cautious |
| aggressive | 10 | False | 30 | Explore farther |
| silicon_focused | 15 | True | 25 | Silicon priority |

### Problem: Low Differentiation
**All configs achieve identical 7/10 success rate**

Current parameters affect:
- ✅ **Speed** (aggressive finishes faster)
- ✅ **Reward quantity** (aggressive gets more hearts)
- ❌ **Success/failure** (no impact on hard experiments)

### Recommended New Configurations

**Config 1: "Pathfinder"** (for complex mazes)
```python
Hyperparameters(
    energy_buffer=5,              # Minimal buffer, trust regen
    prefer_nearby=False,          # Willing to travel far
    exploration_strategy="greedy", # Prioritize known targets
    max_path_length=150,          # NEW: Accept longer paths
    path_recalc_frequency=20,     # NEW: Replan often
)
```
**Target**: Exp 2 (complex navigation)

**Config 2: "Patient Harvester"** (for cooldowns/low efficiency)
```python
Hyperparameters(
    energy_buffer=15,
    cooldown_tolerance=100,        # NEW: Willing to wait
    max_wait_turns=200,           # Increased from 50
    depletion_threshold=0.5,      # More tolerant of depleted extractors
    enable_cooldown_waiting=True, # NEW: Wait at extractors
)
```
**Target**: Exp 3 (oxygen bottleneck)

**Config 3: "Efficiency Optimizer"** (for resource scarcity)
```python
Hyperparameters(
    track_efficiency=True,
    efficiency_weight=0.8,        # Strongly prefer efficient extractors
    resource_batching=True,       # NEW: Batch collection trips
    prefer_nearby=True,
    assembly_priority=True,       # NEW: Prioritize assembly when ready
)
```
**Target**: Exp 10 (low efficiency)

**Config 4: "Sprint"** (for easy maps)
```python
Hyperparameters(
    energy_buffer=5,
    prefer_nearby=True,
    cooldown_tolerance=5,         # Don't wait, find alternatives
    exploration_strategy="frontier",
    aggressive_assembly=True,     # NEW: Assembly ASAP
)
```
**Target**: Maximize hearts on easy experiments

---

## 4. Navigation Quality Assessment

### Current Navigation: BFS (Breadth-First Search)

**Strengths:**
- ✅ Finds shortest path (if one exists)
- ✅ Simple, reliable
- ✅ Works well on small-medium maps (30x30 to 70x70)

**Weaknesses:**
- ❌ **Slow on large maps** (80x80+)
- ❌ **No heuristic guidance** (explores all directions equally)
- ❌ **Memory intensive** (stores all visited nodes)
- ❌ **Doesn't handle dynamic obstacles well**
- ❌ **No path quality metrics** (can't prefer "safer" paths)

### Navigation Performance by Map Size

| Map Size | BFS Performance | Success Rate |
|----------|----------------|--------------|
| 30x30 | Excellent | 100% |
| 50x50 | Good | 100% |
| 70x70 | Good | 100% |
| 80x80 | **Poor** | 0% (on complex mazes) |
| 100x100 | Acceptable | 100% (on open maps) |

**Key Finding**: BFS works fine on open maps (Exp 8,9: 100x100), but **fails on 80x80 mazes** (Exp 2,10)

### Recommended Improvements

#### **Option A: A* Pathfinding** (Recommended)
**Pros:**
- Much faster on large maps
- Guided by heuristic (Manhattan distance)
- Still finds optimal paths
- Industry standard

**Cons:**
- Slightly more complex
- Need to implement priority queue

**Implementation Estimate:** 2-3 hours
**Expected Impact:** +20% success rate on large/complex maps

**Example A* implementation:**
```python
def _astar_next_step(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[Tuple[int, int]]:
    """A* pathfinding with Manhattan distance heuristic."""
    from heapq import heappush, heappop

    def heuristic(pos: Tuple[int, int]) -> int:
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    open_set = [(heuristic(start), 0, start)]  # (f_score, g_score, position)
    came_from = {}
    g_score = {start: 0}

    while open_set:
        _, current_g, current = heappop(open_set)

        if current == goal:
            # Reconstruct path
            return self._reconstruct_first_step(came_from, start, goal)

        for next_pos in self._neighbors4(*current):
            if self._occ[next_pos[0]][next_pos[1]] == self.OCC_WALL:
                continue

            tentative_g = current_g + 1
            if next_pos not in g_score or tentative_g < g_score[next_pos]:
                g_score[next_pos] = tentative_g
                f_score = tentative_g + heuristic(next_pos)
                heappush(open_set, (f_score, tentative_g, next_pos))
                came_from[next_pos] = current

    return None  # No path found
```

#### **Option B: Hybrid BFS + A***
- Use BFS for short distances (<20 tiles)
- Use A* for long distances (≥20 tiles)
- Best of both worlds

#### **Option C: Jump Point Search**
- Advanced A* variant
- Extremely fast on grid maps
- More complex to implement
- Only needed if A* still too slow

---

## 5. Recommended Implementation Priority

### Phase 1: Critical Fixes (Week 1)
**Goal**: 90% success rate

1. **Implement A* Pathfinding** [2-3 hours]
   - Replace `_bfs_next_step_optimistic` with A*
   - Keep BFS as fallback for short distances
   - **Expected**: +10% (Exp 2 solved)

2. **Add Cooldown Waiting** [2 hours]
   - Track cooldown timers in ExtractorMemory
   - Wait near extractor if cooldown < threshold
   - **Expected**: +10% (Exp 3 solved)

3. **Test and Validate** [1 hour]
   - Rerun full evaluation
   - Verify no regressions on working experiments

**Total Effort**: 5-6 hours
**Expected Result**: 36/40 success (90%)

### Phase 2: Optimization (Week 2)
**Goal**: Maximize hearts on working experiments

4. **Add New Hyperparameter Configs** [1 hour]
   - Implement "Pathfinder", "Patient", "Optimizer" configs
   - Test on all experiments

5. **Dynamic Parameter Adjustment** [2-3 hours]
   - Detect map size and adjust energy_buffer
   - Detect low efficiency and increase wait tolerance
   - **Expected**: Better rewards, same success rate

6. **Path Caching** [1 hour]
   - Cache assembler→chest, spawn→chargers paths
   - Reuse for efficiency

**Total Effort**: 4-5 hours
**Expected Result**: 36/40 success, 2.0+ avg reward

### Phase 3: Advanced Features (Week 3+)
**Goal**: Handle edge cases, multi-agent coordination

7. **Multi-Objective Planning**
   - Batch resource collection
   - Opportunistic gathering en route

8. **Advanced Energy Management**
   - Predictive energy modeling
   - Dynamic charging strategies

9. **Multi-Agent Awareness**
   - Avoid contention for extractors
   - Coordinate resource collection

---

## 6. Expected Performance Improvements

### After Phase 1 (Critical Fixes)
| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Success Rate | 70% | **90%** | +20% |
| Working Experiments | 7/10 | **9/10** | +2 |
| Avg Reward | 1.35 | **1.50** | +0.15 |
| Exp 2 (maze) | ❌ | ✅ | FIXED |
| Exp 3 (cooldown) | ❌ | ✅ | FIXED |
| Exp 10 (extreme) | ❌ | ⚠️ | Maybe |

### After Phase 2 (Optimization)
| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Success Rate | 90% | **90%** | - |
| Avg Reward | 1.50 | **2.00+** | +0.50 |
| Max Hearts (Exp 5) | 3 | **4-5** | Better efficiency |

---

## 7. Specific Code Changes Needed

### 1. Replace BFS with A* in execute_phase
```python
# In _execute_phase, line ~794
# OLD:
step = self._bfs_next_step_optimistic((state.agent_row, state.agent_col), goal)

# NEW:
distance = abs(goal[0] - state.agent_row) + abs(goal[1] - state.agent_col)
if distance > 20:  # Use A* for long distances
    step = self._astar_next_step((state.agent_row, state.agent_col), goal)
else:  # Use BFS for short distances (faster for small search space)
    step = self._bfs_next_step_optimistic((state.agent_row, state.agent_col), goal)
```

### 2. Add Cooldown Waiting Logic
```python
# In _find_best_extractor_for_phase
best = self.extractor_memory.find_best_extractor(resource_type, current_pos, state.step_count, self.hyperparams)

if best is None:
    # NEW: Check if any extractors are on cooldown
    on_cooldown = [e for e in all_extractors if not e.is_available(state.step_count)]
    if on_cooldown:
        nearest_cooldown = min(on_cooldown, key=lambda e: e.cooldown_remaining(state.step_count))
        if nearest_cooldown.cooldown_remaining(state.step_count) < self.hyperparams.max_wait_turns:
            logger.info(f"[Phase1] Waiting for {resource_type} cooldown ({nearest_cooldown.cooldown_remaining(state.step_count)} turns)")
            return nearest_cooldown.position  # Go wait there

    # No available extractors - need to explore
    logger.info(f"[Phase1] No available {resource_type} extractors, exploring")
    return None
```

### 3. Add Assembly Priority Check
```python
# In _determine_phase, after has_all_resources check
if has_all_resources and state.energy >= self.ENERGY_REQ:
    # NEW: Verify can reach assembler before committing
    assembler_pos = self._station_positions.get("assembler")
    if assembler_pos and state.agent_row != -1:
        path_exists = self._astar_next_step((state.agent_row, state.agent_col), assembler_pos)
        if path_exists is None:
            logger.warning(f"[Phase1] Have all resources but can't path to assembler!")
            # Try to explore toward assembler direction
            return GamePhase.EXPLORE  # or find closer charger

    return GamePhase.ASSEMBLE_HEART
```

---

## 8. Conclusion & Recommendations

### Primary Recommendation: **Implement Critical Fixes (Phase 1)**
**Effort**: 5-6 hours
**Impact**: 70% → 90% success rate
**ROI**: Very High

The current hyperparameters are **sufficient** - the problem is **algorithmic limitations** in:
1. ❌ Navigation (BFS on large mazes)
2. ❌ Resource timing (no cooldown awareness)

### Secondary Recommendation: **Add Diverse Hyperparameter Configs**
**Effort**: 1 hour
**Impact**: Better rewards, more strategic diversity
**ROI**: Medium

New configs should target **specific scenarios**:
- "Pathfinder" for mazes
- "Patient" for cooldowns
- "Optimizer" for scarcity

### Not Recommended: **Pure hyperparameter tuning of existing configs**
Current configs already well-tuned. Tweaking energy_buffer or prefer_nearby won't solve the 3 failing experiments.

---

## Next Steps

1. **Implement A* pathfinding** (priority 1)
2. **Add cooldown waiting** (priority 2)
3. **Rerun evaluation** to validate improvements
4. **If 90%+ achieved**: Add optimization features (Phase 2)
5. **If still <90%**: Investigate remaining failures with detailed diagnostics

**Expected Timeline**: 1-2 weeks to 90% success rate

