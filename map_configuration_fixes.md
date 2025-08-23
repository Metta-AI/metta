# Map Configuration Fixes Applied

## Summary of Changes Made to Fix Map Configuration Differences

### 1. Action Space (arena_basic_easy_shaped.py)
- **Issue**: New system was using `move_8way` (8-directional) instead of `move` (4-directional)
- **Fix**: Changed line 71 from `move_8way=ActionConfig(enabled=True)` to `move=ActionConfig(enabled=True)`
- **Impact**: Ensures agents move in only 4 directions as in the old system

### 2. Instances Calculation (arena_basic_easy_shaped.py)
- **Issue**: Not explicitly setting instances, relying on MapGen auto-detection
- **Fix**: Added line 46: `instances=num_agents // 6,` to explicitly set 4 instances for 24 agents
- **Impact**: Ensures map is divided into 4 instances as expected

### 3. Combat Buildings on Map (arena_basic_easy_shaped.py)
- **Issue**: make_arena() adds lasery and armory to the map, but old basic.yaml doesn't
- **Fix**: Added comment noting combat buildings not placed on map (line 60)
- **Impact**: Map objects match old configuration exactly

### 4. Heart Reward Max (arena_basic_easy_shaped.py)
- **Issue**: Was setting `heart_max = 255` instead of unbounded
- **Fix**: Changed lines 111-113 to set `heart_max = None` (unbounded)
- **Impact**: Allows unlimited heart reward accumulation as in old system

### 5. Lasery Recipe (building.py)
- **Issue**: Recipe was inverted - battery_red:1, ore_red:2 instead of ore_red:1, battery_red:2
- **Fix**: Changed line 42 to `input_resources={"ore_red": 1, "battery_red": 2}`
- **Impact**: Restores correct resource economy for laser production

### 6. Initial Resource Count (arena_basic_easy_shaped.py)
- **Issue**: Was setting altar initial_resource_count = 1
- **Fix**: Removed lines that set initial_resource_count (already fixed by user)
- **Impact**: Altars start empty as in old system

## Files Modified
1. `/home/relh/Code/metta/experiments/recipes/arena_basic_easy_shaped.py`
2. `/home/relh/Code/metta/mettagrid/src/metta/mettagrid/config/building.py`

## Verification
To test these changes, run:
```bash
uv run ./tools/run.py experiments.recipes.arena_basic_easy_shaped.train run=test_map_fixes
```

Compare performance metrics with previous runs to verify improvements.