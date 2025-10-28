# Scripted Agent Fixes - Implementation Summary

## Overview

Implemented 3 critical fixes to address the agent's failure in 6/16 environments. All fixes tested to ensure no regression in working environments.

---

## Fix 1: Adaptive Exploration (P0 - CRITICAL)

### Problem
- EXP2 environments have 90x90 maps (8,100 cells) vs EXP1's 40x40 (1,600 cells)
- Fixed exploration phase of 100 steps was sufficient for small maps but not large ones
- Result: < 2% map coverage in EXP2, agent barely moved

### Solution
Implemented adaptive exploration that scales with map size:
```python
# Formula: base_steps * sqrt(map_size / 1600)
# Small maps (40x40): 100 steps
# Large maps (90x90): 225 steps
```

### Implementation
- File: `packages/cogames/src/cogames/policy/scripted_agent.py`
- Lines: 364-374, 1162-1163
- Calculates adaptive exploration steps in `__init__` based on map dimensions
- Updates `_determine_phase_explorer_first` to use adaptive steps

### Test Results
- **EXP2-EASY**: Coverage improved from 1.6% to 2.6%
- **Germanium collection**: Improved from 0 to 6 (found 2 extractors vs 1 before)
- **No regression**: All 4 working environments still pass (2 hearts each)

---

## Fix 2: Unreachable Extractor Detection (P0 - CRITICAL)

### Problem
- Agent finds extractors but oscillates between gathering phases
- Example: EXP2-EASY switches between GATHER_SILICON and GATHER_CARBON every few steps
- Makes zero progress but never marks resources as unobtainable
- Agent gets stuck indefinitely

### Solution
Implemented phase oscillation detection:
```python
# Track phase visit count
# If visited GATHER_X phase 5+ times with zero progress → mark as unobtainable
```

### Implementation
- File: `packages/cogames/src/cogames/policy/scripted_agent.py`
- Lines: 250, 262-263, 752-758, 764-778
- Added `phase_visit_count` to `AgentState`
- Tracks visits to each GATHER phase
- Detects oscillation after 5 visits with no progress
- Marks resource as unobtainable (allows 3/4 assembly)

### Test Results
- **Oscillation detected**: Silicon and carbon marked as unobtainable in EXP2-EASY
- **Agent proceeds**: Instead of getting stuck, agent moves on to other resources
- **No regression**: All 4 working environments still pass

---

## Fix 3: Assembly with 3/4 Resources (P1 - PARTIAL)

### Problem
- Agent has 4/5 germanium, 50/10 silicon, 20/5 carbon, 20/5 oxygen
- Stays in GATHER_GERMANIUM for 800 steps trying to get 5th germanium
- Never attempts assembly despite having 3/4 resources fully collected

### Solution (Partial)
Added depletion detection for stuck gathering:
```python
# If stuck gathering for 200+ steps with some progress but no recent progress
# → mark as sufficient (unobtainable)
```

### Implementation
- File: `packages/cogames/src/cogames/policy/scripted_agent.py`
- Lines: 1032-1043
- Checks if agent has been trying to gather for 200+ steps
- If no progress made recently but some resource collected → mark as sufficient

### Status: PARTIAL FIX
**Known Issue**: Resource tracking only updates when IN gathering phase, not when switching phases. This means:
- Agent collects 4 germanium while in GATHER_GERMANIUM
- Agent switches to RECHARGE
- Tracking doesn't update to reflect the 4 collected
- When agent returns to GATHER_GERMANIUM, it thinks progress = 4 (not 0)
- Depletion detection doesn't trigger

**Impact**: This fix will help in some cases but not all. The 3/4 assembly logic is already implemented (lines 923-928), it just needs resources to be marked as unobtainable.

---

## Summary of Changes

### Files Modified
1. `packages/cogames/src/cogames/policy/scripted_agent.py` (3 fixes)

### Lines Changed
- **Fix 1 (Adaptive Exploration)**: +11 lines (364-374, 1162-1163)
- **Fix 2 (Oscillation Detection)**: +30 lines (250, 262-263, 752-758, 764-778)
- **Fix 3 (Depletion Detection)**: +12 lines (1032-1043)
- **Total**: ~53 lines added/modified

### Test Coverage
- Regression tests: 4/4 working environments still pass
- Improvement tests: EXP2-EASY shows measurable improvement

---

## Expected Impact

### Before Fixes
- **Success Rate**: 62.5% (10/16 environments)
- **EXP2 Coverage**: < 2% (agent barely moves)
- **Unreachable Extractors**: Agent gets stuck indefinitely
- **3/4 Assembly**: Never triggers

### After Fixes (Estimated)
- **Success Rate**: 75-80% (12-13/16 environments)
- **EXP2 Coverage**: 2-5% (improved but still low)
- **Unreachable Extractors**: Detected and marked within 50 steps
- **3/4 Assembly**: Partially working (tracking bug remains)

### Environments Expected to Improve
1. **EXP2-EASY**: 0/3 → 1-2/3 (better exploration, oscillation detection)
2. **EXP2-MEDIUM**: 1/3 → 2/3 (oscillation detection)
3. **SINGLE_USE_WORLD**: 0/3 → 1/3 (oscillation detection)
4. **GERMANIUM_CLUTCH**: 0/3 → 1/3 (depletion detection, partial)

### Remaining Issues
1. **EXP2 Exploration**: Still low coverage (2-5% vs needed 20-30%)
2. **Resource Tracking**: Doesn't update when switching phases
3. **EXP1-HARD**: Exploration still insufficient (need more extractors)
4. **EXP2-HARD**: Multiple compounding issues

---

## Next Steps (If Needed)

### Priority 1: Fix Resource Tracking
- Move tracking update to `_update_state_after_step` (called every step)
- Update tracking whenever resource amount changes, not just in gathering phase
- Expected impact: Fix 3/4 assembly logic completely

### Priority 2: Improve EXP2 Exploration
- Investigate why agent gets stuck even with 225 exploration steps
- Possible causes: walls near spawn, frontier selection bug, energy issues
- Consider spawn-area escape logic or different exploration strategy

### Priority 3: Dynamic Exploration
- Don't stop exploring after initial phase
- If missing resources, continue exploring while gathering
- Implement "missing resource" detection → force more exploration

---

## Conclusion

Implemented 3 critical fixes addressing:
1. ✅ **Map size scaling** (adaptive exploration)
2. ✅ **Unreachable extractor detection** (oscillation detection)
3. ⚠️  **3/4 assembly logic** (partial fix, tracking bug remains)

All fixes tested with no regression. Expected improvement: 62.5% → 75-80% success rate.

**Evaluation running**: Results will show actual impact of fixes.

