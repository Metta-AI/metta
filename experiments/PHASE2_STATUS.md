# Phase 2 Implementation Status

**Date**: October 24, 2025  
**Branch**: Scripted Agent Improvements

---

## Summary

Implemented A* pathfinding and cooldown waiting logic for the scripted agent. These improvements partially work but need further refinement.

### Test Results: 1/3 Success (33.3%)

| Experiment | Before | After | Status | Notes |
|------------|--------|-------|--------|-------|
| **Exp 1** (Baseline) | ✅ | ✅ | No regression | Works as expected |
| **Exp 2** (80x80 maze) | ❌ | ⚠️ | Partial improvement | Collects all resources but fails assembly |
| **Exp 3** (Low efficiency) | ❌ | ⚠️ | Partial improvement | Waits for cooldown but times out |

---

## What Was Implemented

### 1. A* Pathfinding ✅ (Partially Working)

**Changes Made:**
- Added `_astar_next_step()` method using Manhattan distance heuristic
- Added `_choose_pathfinding()` to intelligently select BFS vs A* based on distance
- Updated `_execute_phase()` to use new pathfinding
- Added hyperparameters: `use_astar`, `astar_threshold`

**Impact:**
- **Exp 2**: Agent successfully navigates 80x80 maze and collects ALL resources (C=20, O=20, G=6, Si=50)
- **Exp 1**: No regression, still works

**Remaining Issue:**
- Agent fails to discover or navigate to assembler after collecting resources
- Needs: Better station discovery during exploration, or earlier assembler targeting

---

### 2. Cooldown Waiting Logic ✅ (Partially Working)

**Changes Made:**
- Added `cooldown_remaining()` method to track extractor cooldowns
- Modified `_find_best_extractor_for_phase()` to identify and wait near extractors on cooldown
- Added hyperparameters: `enable_cooldown_waiting`, `max_cooldown_wait`

**Impact:**
- **Exp 3**: Agent correctly waits at oxygen extractor when on cooldown
- Agent gathers 19/20 oxygen (needs 20 total, gets 15 per harvest due to 75% efficiency)

**Remaining Issue:**
- 100-turn cooldown + 1000-step limit = timeout before second oxygen harvest
- Needs: Either longer step limit, or multi-source oxygen gathering strategy

---

## Detailed Problem Analysis

### Exp 2: Assembler Discovery Problem

**Observations:**
```
Final state: C=20, O=20, G=6, Si=50, E=33
All resources collected ✓
No heart assembled ✗
```

**Root Cause Hypotheses:**
1. **Assembler not discovered**: Agent explored for resources but never saw assembler
2. **Assembler navigation failure**: Discovered but can't path to it (walls blocking all approaches)
3. **Phase transition issue**: Has resources but not transitioning to ASSEMBLE_HEART phase

**Diagnosis Needed:**
- Check if assembler is in `self._station_positions` at end of run
- Check agent's phase at steps 990-1000
- Check assembler location vs agent's explored area

**Potential Fixes:**
1. **Priority exploration**: Target assembler/chest early in exploration
2. **Bidirectional search**: When resources full, actively search for assembler
3. **Longer step limit**: 2000 steps instead of 1000 for large maps

---

### Exp 3: Cooldown Timing Problem

**Observations:**
```
Final state: C=21, O=19, G=6, Si=54, E=100
Missing 1 oxygen (19/20) ✗
Stuck waiting at oxygen extractor
```

**Root Cause:**
- 75% efficiency → 15 O2 per harvest (need 2 harvests for 20 total)
- 100-turn cooldown between harvests
- 1000-step limit insufficient for: explore + harvest1 + wait 100 + harvest2 + assemble

**Calculation:**
- First oxygen: Steps 0-200 (explore + harvest)
- Cooldown wait: Steps 200-300 (100 turns)
- Second oxygen: Steps 300-320 (navigate back)
- Other resources: Steps 320-600
- Assembly: Steps 600+
- **Total needed**: ~700+ steps → Should work!

**Actual Issue:**
Agent is stuck in a loop at the extractor, waiting forever without re-attempting harvest.
Cooldown waiting logic is working, but agent never re-checks if extractor is available.

**Fix Needed:**
- After waiting N turns, re-evaluate extractor availability
- Don't wait forever - explore for alternative oxygen sources if wait > threshold

---

## Recommended Next Steps

### Priority 1: Fix Exp 3 (Easier)

**Problem**: Agent waits at oxygen extractor but never re-attempts harvest

**Fix**:
```python
# In _find_best_extractor_for_phase, after cooldown waiting logic:
if cooldown_time <= self.hyperparams.max_cooldown_wait:
    # NEW: Only wait if we haven't been here recently
    if (target_pos not in state.recent_wait_positions or 
        steps_since_wait > cooldown_time):
        logger.info(f"Waiting for {resource_type} cooldown ({cooldown_time} turns)")
        state.waiting_since_step = state.step_count
        state.wait_target = target_pos
        return target_pos
    else:
        # Waited long enough, try exploring for alternatives
        logger.info(f"Waited too long, exploring for alternative {resource_type}")
        return None
```

**Expected Impact**: Exp 3 should succeed (70%+ success rate → 80%)

---

### Priority 2: Fix Exp 2 (Harder)

**Problem**: Agent doesn't discover or reach assembler after collecting resources

**Fix Options:**

**Option A: Early Assembler Discovery** (Recommended)
```python
# In _determine_phase:
if (has_all_resources or approaching_full_inv) and not self._station_positions.get("assembler"):
    # Force exploration toward map center (likely assembler location)
    return GamePhase.EXPLORE_FOR_ASSEMBLER  # New phase
```

**Option B: Increase Step Limit**
```python
# In test script:
max_steps = 2000  # For large maps (80x80+)
```

**Option C: Prioritize Critical Stations**
```python
# In exploration, prefer frontiers near critical stations (assembler, chest)
# Use heuristic: explore toward map center early
```

**Expected Impact**: Exp 2 should succeed (80%+ success rate → 90%)

---

### Priority 3: Run Full Evaluation

After fixes, rerun comprehensive evaluation on all 10 experiments to measure true improvement.

**Expected Results:**
- Before: 28/40 (70%)
- After Phase 2 fixes: 32-36/40 (80-90%)

---

## Code Changes Summary

### Files Modified:
1. **`packages/cogames/src/cogames/policy/scripted_agent_outpost.py`**
   - Added A* pathfinding
   - Added cooldown waiting logic
   - Updated hyperparameters

### New Hyperparameters:
```python
# Pathfinding
use_astar: bool = True
astar_threshold: int = 20

# Cooldown waiting
enable_cooldown_waiting: bool = True
max_cooldown_wait: int = 100
```

### Lines Changed: ~100 lines added/modified

---

## Conclusion

Phase 2 improvements are **partially successful**:
- ✅ A* pathfinding works well for navigation (Exp 2 collects all resources)
- ✅ Cooldown waiting logic detects cooldowns correctly (Exp 3 waits at extractor)
- ❌ Final assembly step needs work (Exp 2 can't find assembler)
- ❌ Cooldown re-attempt logic needs refinement (Exp 3 waits forever)

**Next**: Implement Priority 1 & 2 fixes, then rerun full evaluation.

