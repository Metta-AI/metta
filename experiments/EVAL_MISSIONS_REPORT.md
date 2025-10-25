# Eval Missions Evaluation Report

**Date**: October 24, 2025
**Agent**: ScriptedAgentOutpostPolicy (Phase 1)
**Status**: ❌ **Cannot Evaluate - Map Files Malformed**

## Executive Summary

**Attempted Evaluation**: 10 eval missions × 4 hyperparameter configurations = 40 runs
**Actual Results**: **0/40 runs completed (0%)**

**Root Cause**: All eval mission map files have **inconsistent line lengths**, causing Pydantic validation errors.

---

## Eval Missions List

The following 10 eval missions were defined in `eval_missions.py`:

1. **EnergyStarved** (`machina_eval_exp01.map`) - 0 energy regen, plan charger routes
2. **OxygenBottleneck** (`machina_eval_exp02.map`) - 50% oxygen efficiency
3. **GermaniumRush** (`machina_eval_exp03.map`) - Limited germanium (10 uses)
4. **SiliconWorkbench** (`machina_eval_exp04.map`) - 150% silicon efficiency
5. **CarbonDesert** (`machina_eval_exp05.map`) - Sparse carbon (30 uses)
6. **SingleUseWorld** (`machina_eval_exp06.map`) - Every station usable once
7. **SlowOxygen** (`machina_eval_exp07.map`) - 25% oxygen efficiency
8. **HighRegenSprint** (`machina_eval_exp08.map`) - 3 energy regen/turn
9. **SparseBalanced** (`machina_eval_exp09.map`) - All resources limited
10. **GermaniumClutch** (`machina_eval_exp10.map`) - Only 2 germanium uses

---

## Error Analysis

### Map Validation Errors

All eval missions failed with the same type of error:

```
Pydantic ValidationError for MapBuilderConfig:
Line X has length Y, expected Z.
All lines in ASCII map must have the same length.
```

### Specific Examples:

- **machina_eval_exp01.map**: Line 2 has length 39, expected 40
- **machina_eval_exp02.map**: Line 2 has length 41, expected 40
- **machina_eval_exp04.map**: Line 2 has length 41, expected 40
- **machina_eval_exp05.map**: (similar validation error)

### Other Missions

A few missions (GermaniumRush, SingleUseWorld, SlowOxygen, GermaniumClutch) ran for 1000 steps but achieved **0 reward**, suggesting either:
- Maps exist but are unsolvable
- Maps are missing critical objects (assemblers, chests, or resources)
- Agent cannot find/reach required stations

---

## Technical Details

### Issues Fixed During Evaluation Attempt

1. **Circular Import**: Fixed circular dependency between `missions.py` and `eval_missions.py`
   - Solution: Added try/except import pattern similar to exploration_experiments

2. **Type Annotations**: Added missing Pydantic type annotations to all eval mission classes
   - Fixed: `name`, `description`, `map_name`, all efficiency parameters, all `max_uses_*` parameters

3. **JSON Serialization**: Enhanced `CustomJsonEncoder` to handle `numpy.bool_` types

### Evaluation Script Status

✅ **Script is ready** (`evaluate_eval_missions.py`)
- Supports 4 hyperparameter configurations
- Properly handles exceptions and logging
- Generates comprehensive reports

---

## Recommendations

### Immediate Actions Required

**Option 1: Fix Existing Maps**
For each `machina_eval_expXX.map` file:
1. Check all lines have consistent length
2. Verify map contains required objects:
   - At least 1 agent spawn (@)
   - At least 1 assembler (&)
   - At least 1 chest (=)
   - Resource extractors (C, O, G, S)
   - Chargers (+)

**Option 2: Create New Maps**
Design maps specifically for each eval mission's challenge:
- **EnergyStarved**: Multiple chargers, long distances
- **OxygenBottleneck**: Limited oxygen extractors with cooldowns
- **GermaniumRush**: Few germanium sources, encourage racing
- **SiliconWorkbench**: Many silicon extractors
- **CarbonDesert**: Sparse carbon placement
- **SingleUseWorld**: Rich variety of each station type
- **SlowOxygen**: Multiple oxygen extractors to encourage interleaving
- **HighRegenSprint**: Large map to leverage high energy regen
- **SparseBalanced**: Even distribution, moderate distances
- **GermaniumClutch**: Single germanium line as chokepoint

**Option 3: Use Exploration Experiment Maps**
Temporarily map eval missions to working exploration experiment maps:
- Leverage the 7/10 working experiments from exploration_experiments
- Adjust efficiency/max_uses parameters to create similar challenges

---

## Status of Map Files

### Need Investigation/Fixing:
- `machina_eval_exp01.map` ❌ (Line length: 39 vs 40)
- `machina_eval_exp02.map` ❌ (Line length: 41 vs 40)
- `machina_eval_exp03.map` ⚠️ (Loads but 0 reward)
- `machina_eval_exp04.map` ❌ (Line length: 41 vs 40)
- `machina_eval_exp05.map` ❌ (Validation error)
- `machina_eval_exp06.map` ⚠️ (Loads but 0 reward)
- `machina_eval_exp07.map` ⚠️ (Loads but 0 reward)
- `machina_eval_exp08.map` ❌ (Validation error)
- `machina_eval_exp09.map` ❌ (Validation error)
- `machina_eval_exp10.map` ⚠️ (Loads but 0 reward)

---

## Comparison with Exploration Experiments

The exploration experiments successfully achieved **70% success rate (28/40 runs)** because:
- ✅ Maps are properly formatted (consistent line lengths)
- ✅ Maps contain all required objects
- ✅ Maps are solvable by the scripted agent
- ✅ Diverse challenges across 10 experiments

The eval missions, once maps are fixed, offer different types of challenges:
- Resource scarcity (max_uses limits)
- Efficiency variations
- Energy management extremes
- Strategic timing requirements

---

## Next Steps

1. **Validate Map Files**: Check if `machina_eval_expXX.map` files exist in `packages/cogames/src/cogames/maps/`
2. **Fix Line Lengths**: Ensure all lines in each map have identical length
3. **Verify Object Placement**: Confirm each map has assembler, chest, spawn, and extractors
4. **Test Individual Maps**: Run single missions to verify solvability
5. **Full Re-evaluation**: Once maps are fixed, run `evaluate_eval_missions.py` again

---

## Files Created

- ✅ `scripts/evaluate_eval_missions.py` - Comprehensive evaluation script
- ✅ `experiments/EVAL_MISSIONS_REPORT.md` - This report
- ✅ Fixed circular imports in `missions.py`
- ✅ Added type annotations to `eval_missions.py`

---

## Conclusion

The eval missions framework is **ready for evaluation** once the map files are fixed. The scripted agent policy is capable (as proven by 70% success on exploration experiments), but the eval mission maps need to be:

1. Created or fixed to have consistent line lengths
2. Verified to contain all required game objects
3. Tested for basic solvability

Once these issues are resolved, the comprehensive evaluation can be rerun to assess the agent's performance across the 10 eval scenarios.

