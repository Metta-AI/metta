# Oscillation Fixes for Harvest Policy on Large Maps

## Summary

Fixed critical oscillation issues that caused the agent to get stuck for thousands of steps on large maps. Reduced max stuck time from **5,000+ steps to 11 steps** (450x improvement).

## Problems Identified

### 1. Frontier Commitment Oscillation
**Symptom**: Agent commits to a distant frontier for 5-30 steps, gets stuck after a few moves, but continues trying the same unreachable frontier.

**Root Cause**: Commitment system didn't check if agent was stuck (`consecutive_failed_moves`). Even when stuck, pathfinding might find alternate routes, so commitment never cleared.

### 2. Recharge Phase Oscillation
**Symptom**: Agent oscillates between 2-3 positions (e.g., (229,251) ↔ (229,252)) for thousands of steps while trying to reach charger.

**Root Cause**: Stuck recovery tried "perpendicular" directions without checking if they led to recently visited positions, creating infinite A→B→A→B loops.

## Solutions Implemented

### Fix 1: Clear Frontier Commitment When Stuck
**File**: `harvest/harvest_policy.py:2478-2481`

```python
elif state.consecutive_failed_moves >= 5:
    self._logger.warning(f"EXPLORE: STUCK - clearing commitment to find alternate target")
    state.committed_frontier_target = None
    state.frontier_target_commitment_steps = 0
```

**Impact**: Allows agent to quickly abandon unreachable frontiers and try alternatives.

### Fix 2: Adaptive Commitment Duration for Large Maps
**File**: `harvest/harvest_policy.py:2516-2528`

```python
map_size = state.mission_profile.map_size if state.mission_profile else "medium"
if map_size == "large":
    commitment_duration = min(30, max(5, dist // 2))
elif map_size == "medium":
    commitment_duration = min(20, max(5, dist * 2 // 5))
else:
    commitment_duration = min(15, max(5, dist // 3))
```

**Impact**: Gives agent more time to navigate complex mazes on large maps while preventing infinite commitment.

### Fix 3: Position History Checking in Recharge Stuck Recovery
**File**: `harvest/harvest_policy.py:1889-1907`

```python
# OSCILLATION FIX: Check if direction leads to recently visited position
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

**Impact**: Prevents A→B→A→B oscillation patterns during stuck recovery.

### Fix 4: Severe Stuck Escape to Exploration
**File**: `harvest/harvest_policy.py:1870-1876`

```python
if state.consecutive_failed_moves >= 20:
    self._logger.error(f"RECHARGE: SEVERELY STUCK - switching to EXPLORATION")
    return self._explore(state)
```

**Impact**: After extended stuck, uses sophisticated exploration algorithms (frontier-based, wall-following) instead of simple perpendicular moves.

## Test Results

### Before Fixes
```
Failed moves: 4,960 / 10,000 (49.6%)
Max stuck: 5,000+ steps
Resources: 2/5 types
Behavior: Infinite oscillation between 2 cells
```

### After Fixes
```
Failed moves: 809 / 2,922 (27.7%)
Max stuck: 11 steps
Resources: 5/5 types ✓
Behavior: Systematic exploration, no severe oscillation ✓
```

## Key Metrics

- **Max stuck reduction**: 5,000+ → 11 steps (450x improvement)
- **Failure rate reduction**: 49.6% → 27.7% (44% improvement)
- **Resource collection**: 2/5 → 5/5 types (complete)

## Testing

Run oscillation analysis:
```bash
uv run python test_oscillation_analysis.py
uv run python test_recharge_oscillation.py
```

## Related Files Modified

- `harvest/harvest_policy.py`: All four fixes implemented
- `test_oscillation_analysis.py`: Analysis tool for detecting oscillation patterns
- `test_recharge_oscillation.py`: Specific test for recharge phase oscillation

## Technical Details

The fixes work by combining multiple strategies:

1. **Early detection**: Check `consecutive_failed_moves` at decision points
2. **Position history**: Track last 5 positions to detect loops
3. **Adaptive thresholds**: Different timeouts for different map sizes
4. **Graceful degradation**: Switch to more sophisticated algorithms when simple recovery fails

This multi-layered approach ensures the agent can escape both simple (2-cell) and complex (3-5 cell) oscillation patterns while still maintaining goal-directed behavior.
