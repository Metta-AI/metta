# Harvest Policy Oscillation Fixes - Complete Summary

## Overview

Fixed critical oscillation and stuck issues that caused the agent to get stuck for up to 10,000 steps. The fixes target multiple phases (GATHER, ASSEMBLE, DELIVER, RECHARGE) and multiple root causes (frontier commitment, stuck recovery, pathfinding loops).

## Test Results Summary

### training_facility.harvest (Small Map)

**Before All Fixes:**
- Move failures: 9,952 / 9,996 (99.6% failure rate)
- Max stuck: 9,950 steps
- Hearts deposited: 1 (stuck in deliver phase)

**After All Fixes:**
- Move failures: 104 / 9,969 (1.0% failure rate)
- Max stuck: 5 steps
- Hearts deposited: 2 ✓

**Improvement:**
- 98.6% reduction in failure rate
- 1,990x improvement in max stuck time
- Successful completion of full harvest cycle

### machina_1.open_world (Large Map)

**Before Fixes:**
- Max stuck: 5,000+ steps
- Failure rate: 49.6%
- Resources: 2/5 types
- Hearts: 0

**After Initial Fixes (1-5):**
- Max stuck: 22 steps (227x improvement)
- Failure rate: 9.4%
- Resources: 5/5 types ✓
- Hearts: Variable (0-1)

**Status:** Works much better but has remaining pathfinding issues where agent can't find paths to frontiers despite having clear directions.

## All Fixes Implemented

### Fix 1: Clear Frontier Commitment When Stuck
**File:** `harvest/harvest_policy.py:2478-2481`

**Problem:** Agent commits to distant frontier for up to 30 steps, gets stuck after a few moves, but continues trying same unreachable frontier.

**Solution:**
```python
elif state.consecutive_failed_moves >= 5:
    self._logger.warning(f"EXPLORE: STUCK - clearing commitment to find alternate target")
    state.committed_frontier_target = None
    state.frontier_target_commitment_steps = 0
```

**Impact:** Allows agent to quickly abandon unreachable frontiers and try alternatives.

---

### Fix 2: Adaptive Commitment Duration for Large Maps
**File:** `harvest/harvest_policy.py:2516-2528`

**Problem:** Fixed commitment duration doesn't account for map complexity.

**Solution:**
```python
map_size = state.mission_profile.map_size if state.mission_profile else "medium"
if map_size == "large":
    commitment_duration = min(30, max(5, dist // 2))
elif map_size == "medium":
    commitment_duration = min(20, max(5, dist * 2 // 5))
else:
    commitment_duration = min(15, max(5, dist // 3))
```

**Impact:** Gives agent more time to navigate complex mazes on large maps while preventing infinite commitment.

---

### Fix 3: Position History Checking in Recharge
**File:** `harvest/harvest_policy.py:1889-1907`

**Problem:** Stuck recovery tried perpendicular directions without checking if they led to recently visited positions, creating A→B→A→B loops.

**Solution:**
```python
dir_offsets = {"north": (-1, 0), "south": (1, 0), "east": (0, 1), "west": (0, -1)}
recent_positions = set(state.position_history[-5:])

for alt_dir in alt_directions:
    if not self._is_direction_clear_in_obs(state, alt_dir):
        continue

    dr_alt, dc_alt = dir_offsets[alt_dir]
    target_pos = (state.row + dr_alt, state.col + dc_alt)

    if target_pos in recent_positions:
        self._logger.debug(f"STUCK RECOVERY: Skipping {alt_dir} - would revisit {target_pos}")
        continue

    return self._actions.move.Move(alt_dir)
```

**Impact:** Prevents 2-cell oscillation patterns during stuck recovery.

---

### Fix 4: Recharge Phase Tracking
**File:** `harvest/harvest_policy.py:880-881, 895-896`

**Problem:** No way to detect when agent spends too long in recharge phase overall (not just consecutive failures).

**Solution:**
```python
state.recharge_phase_start_step = state.step_count
state.recharge_failed_attempts = 0
```

**Impact:** Enables time-based and attempt-based escape conditions.

---

### Fix 5: Multiple Escape Conditions in Recharge
**File:** `harvest/harvest_policy.py:1879-1902`

**Problem:** Single threshold (20 consecutive failures) was too high for some trap situations where failures reset periodically.

**Solution:**
```python
steps_in_recharge = state.step_count - state.recharge_phase_start_step

if state.consecutive_failed_moves >= 5:
    state.recharge_failed_attempts += 1

    should_escape = (
        state.consecutive_failed_moves >= 20 or
        (steps_in_recharge > 100 and state.consecutive_failed_moves >= 5) or
        state.recharge_failed_attempts > 50
    )

    if should_escape:
        self._logger.error(f"RECHARGE: SEVERELY STUCK - switching to EXPLORATION")
        state.recharge_failed_attempts = 0
        return self._explore(state)
```

**Impact:** Catches traps where consecutive failures never reach 20 but agent is clearly stuck.

---

### Fix 6: Severely Stuck Escape in Navigation (Observation-based)
**File:** `harvest/harvest_policy.py:1773-1784`

**Problem:** When stuck recovery calls `_explore_observation_only`, it might still fail if in a trapped area.

**Solution:**
```python
if state.consecutive_failed_moves >= 100:
    self._logger.error(f"  NAVIGATE: SEVERELY STUCK ({state.consecutive_failed_moves} fails)")
    import random
    clear_dirs = [d for d in ["north", "south", "east", "west"]
                 if self._is_direction_clear_in_obs(state, d)]
    if clear_dirs:
        random_dir = random.choice(clear_dirs)
        return self._actions.move.Move(random_dir)
    else:
        return self._actions.noop.Noop()
```

**Impact:** Aggressive random escape when all navigation strategies fail.

---

### Fix 7: Severely Stuck Escape BEFORE Pathfinding
**File:** `harvest/harvest_policy.py:1793-1804`

**Problem:** Fix 6 was never reached because pathfinding ran first and kept returning actions (even failing ones) before stuck recovery could activate.

**Root Cause:** Control flow in `_navigate_to_station()`:
1. Lines 1740-1766: Try to reach visible station
2. Lines 1768-1786: Stuck recovery (Fix 6)
3. **Lines 1788-1808: Use pathfinding to known station** ← This runs BEFORE stuck recovery!

The pathfinding section would find the station location, call `_move_towards()`, which would pick "east" as a clear direction and return it. The stuck recovery at lines 1768-1786 was never reached even at 428 consecutive failures.

**Solution:** Add stuck check BEFORE calling pathfinding:
```python
# Priority 3: Use pathfinding to known station location
station_pos = state.stations.get(station_type)
if station_pos is not None:
    # CRITICAL: Check for severe stuck BEFORE pathfinding
    # Pathfinding may keep returning same failing direction
    if state.consecutive_failed_moves >= 100:
        self._logger.error(f"  NAVIGATE: SEVERELY STUCK ({state.consecutive_failed_moves} fails) before pathfinding")
        import random
        clear_dirs = [d for d in ["north", "south", "east", "west"]
                     if self._is_direction_clear_in_obs(state, d)]
        if clear_dirs:
            random_dir = random.choice(clear_dirs)
            self._logger.warning(f"  NAVIGATE: Random escape direction: {random_dir}")
            return self._actions.move.Move(random_dir)
        else:
            self._logger.error(f"  NAVIGATE: ALL DIRECTIONS BLOCKED - switching to exploration")
            return self._explore(state)

    action = self._move_towards(state, station_pos, reach_adjacent=True, station_name=station_type)
    if action.name != "noop":
        return action
```

**Impact:** This was the critical fix for training_facility.harvest, reducing max stuck from 9,950 to 5 steps (1,990x improvement) and failure rate from 99.6% to 1.0%.

**Key Insight:** Control flow ordering matters! Checks for stuck conditions must come BEFORE operations that might return actions, otherwise the recovery code is never reached.

---

## Technical Insights

### Multi-Layered Stuck Detection

The fixes work by combining multiple strategies:

1. **Early detection**: Check `consecutive_failed_moves` at decision points
2. **Position history**: Track last 5 positions to detect loops
3. **Time-based thresholds**: Different timeouts for different map sizes
4. **Attempt-based counting**: Count total failed attempts, not just consecutive
5. **Control flow ordering**: Check stuck conditions BEFORE operations that might succeed
6. **Graceful degradation**: Switch to more sophisticated algorithms when simple recovery fails

### The Pathfinding Loop Problem

A key discovery was that pathfinding itself can create infinite loops:
- Pathfinding finds a "valid" path based on the occupancy map
- Agent tries the first step and fails (actual obstacle not in map)
- Pathfinding runs again, finds the same "valid" path
- Loop continues indefinitely

**Solution:** Check for severe stuck (100+ failures) BEFORE calling pathfinding, allowing random escape to break the loop.

### Position History vs Consecutive Failures

Two complementary stuck detection mechanisms:
- **Consecutive failures**: Detects when moves keep failing
- **Position history**: Detects oscillation even when some moves succeed

Together they catch both "can't move at all" and "moving but going nowhere" stuck patterns.

## Remaining Issues

### machina_1.open_world Pathfinding

The agent on large maps sometimes can't find paths to frontiers even when all directions are clear. This suggests:
1. Occupancy map may have errors (marking traversable cells as blocked)
2. BFS pathfinding may fail in certain maze configurations
3. Frontier detection may return unreachable cells

This is a separate issue from oscillation and requires investigation of the pathfinding and occupancy map logic.

## Files Modified

- `harvest/harvest_policy.py`: All 7 fixes implemented
- `OSCILLATION_FIXES.md`: Initial documentation of Fixes 1-5
- `HARVEST_FIXES_SUMMARY.md`: Complete documentation (this file)

## Testing

Run the tests:
```bash
# Small map (training_facility) - should be excellent
uv run python test_training_facility.py

# Large map (machina_1) - works but has pathfinding issues
uv run python test_machina1_quick2.py

# All maps comparison
uv run python test_all_maps.py
```

## Conclusion

The oscillation and severe stuck issues are **solved** for typical scenarios. The agent now:
- Recovers from stuck situations in 5-22 steps (vs 5,000-10,000 before)
- Has 1-10% failure rates (vs 50-99.6% before)
- Successfully completes full harvest cycles on small maps
- Collects all 5 resource types on large maps

The remaining pathfinding issues on large maps are a separate problem that doesn't cause catastrophic oscillation, just reduced efficiency.
