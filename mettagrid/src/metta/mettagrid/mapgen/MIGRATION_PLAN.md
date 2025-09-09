# Scene Classes Migration Plan: String-based to Int-based Grid Support

This document outlines the systematic approach for updating all 49 scene classes to support both legacy string-based
grids and new int-based grids during the migration period.

## Migration Status

### ✅ Completed (5 files)

- `enhanced_scene.py` - New enhanced base class with dual-format support
- `scenes/enhanced_random.py` - New enhanced Random scene implementation
- `scenes/enhanced_mean_distance.py` - New enhanced MeanDistance scene implementation
- `scenes/random.py` - Updated in-place to support both formats
- `scenes/mean_distance.py` - Updated in-place to support both formats

### 🔄 In Progress (0 files)

### ⏳ Remaining (44 files)

- `scenes/varied_terrain.py` - Complex terrain generation
- `scenes/yaml.py` - YAML-based map loading
- `scenes/nop.py` - No-operation scene
- `scenes/radial_maze.py` - Radial maze generation
- `scenes/random_scene.py` - Random scene selection
- `scenes/random_objects.py` - Random object placement
- `scenes/auto.py` - Automatic scene generation
- `scenes/spiral.py` - Spiral pattern generation
- `scenes/layout.py` - Area layout management
- `scenes/mirror.py` - Mirror/symmetry operations
- `scenes/make_connected.py` - Connectivity algorithms
- `scenes/remove_agents.py` - Agent removal operations
- `scenes/maze.py` - Maze generation
- `scenes/inline_ascii.py` - Inline ASCII map parsing
- `scenes/wfc.py` - Wave Function Collapse
- `scenes/random_dcss_scene.py` - DCSS-style random generation
- `scenes/transplant_scene.py` - Scene transplantation
- `scenes/copy_grid.py` - Grid copying operations
- `scenes/convchain.py` - Convolution chain operations
- `scenes/random_yaml_scene.py` - Random YAML scene selection
- `scenes/multi_left_and_right.py` - Multi-directional layout
- `scenes/room_grid.py` - Room-based grid generation
- `scenes/grid_altars.py` - Altar grid placement
- `scenes/ascii.py` - ASCII map parsing
- `scenes/bsp.py` - Binary Space Partitioning

Plus 19 additional test scene files in `tests/mapgen/scenes/`

## Migration Approaches

### Option 1: In-Place Migration (Recommended for Core Scenes)

Update existing scene files to support both formats by:

1. **Add format detection**:

   ```python
   def __init__(self, *args, **kwargs):
       super().__init__(*args, **kwargs)
       self._grid_is_int = self.grid.dtype == np.uint8
       self._empty_value = ObjectTypes.EMPTY if self._grid_is_int else "empty"
   ```

2. **Update grid operations**:

   ```python
   # Before:
   self.grid[pos] = "wall"
   empty_mask = self.grid == "empty"

   # After:
   if self._grid_is_int:
       self.grid[pos] = ObjectTypes.WALL
   else:
       self.grid[pos] = "wall"
   empty_mask = self.grid == self._empty_value
   ```

3. **Add object type mapping**:
   ```python
   if self._grid_is_int:
       try:
           from metta.mettagrid.type_mapping import TypeMapping
           type_mapping = TypeMapping()
           type_id = type_mapping.get_type_id(obj_name)
           self.grid[pos] = type_id
       except KeyError:
           print(f"Warning: Unknown object {obj_name}, skipping")
   else:
       self.grid[pos] = obj_name
   ```

### Option 2: Enhanced Scene Creation (Recommended for Complex Scenes)

Create new enhanced versions alongside existing ones:

1. Inherit from `EnhancedScene` base class
2. Use type-safe utility methods
3. Better error handling and validation
4. Maintain backward compatibility

## Implementation Priority

### High Priority (Core Functionality) - Week 1

1. `scenes/room_grid.py` - Fundamental room generation
2. `scenes/maze.py` - Basic maze algorithms
3. `scenes/layout.py` - Area layout management
4. `scenes/ascii.py` - ASCII map parsing
5. `scenes/bsp.py` - Binary space partitioning

### Medium Priority (Common Features) - Week 2

6. `scenes/random_objects.py` - Random placement patterns
7. `scenes/copy_grid.py` - Grid copying utilities
8. `scenes/mirror.py` - Symmetry operations
9. `scenes/make_connected.py` - Connectivity algorithms
10. `scenes/remove_agents.py` - Agent management

### Lower Priority (Specialized Features) - Week 3

11. `scenes/varied_terrain.py` - Complex terrain
12. `scenes/wfc.py` - Wave Function Collapse
13. `scenes/convchain.py` - Convolution chains
14. `scenes/radial_maze.py` - Radial patterns
15. `scenes/spiral.py` - Spiral patterns

### Lowest Priority (Utils/Tests) - Week 4

16. All test scene files in `tests/mapgen/scenes/`
17. Utility scenes like `nop.py`
18. Specialized DCSS and YAML scenes

## Standard Migration Pattern

Each scene file should follow this pattern:

```python
# 1. Add imports
from metta.mettagrid.object_types import ObjectTypes

# 2. Add format detection to __init__
def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._grid_is_int = self.grid.dtype == np.uint8
    self._empty_value = ObjectTypes.EMPTY if self._grid_is_int else "empty"

# 3. Update all grid assignments
def place_object(self, pos, obj_name):
    if self._grid_is_int:
        try:
            from metta.mettagrid.type_mapping import TypeMapping
            type_mapping = TypeMapping()
            type_id = type_mapping.get_type_id(obj_name)
            self.grid[pos] = type_id
        except (KeyError, ImportError):
            print(f"Warning: Unknown object {obj_name}, skipping")
    else:
        self.grid[pos] = obj_name

# 4. Update all comparisons
def find_empty_cells(self):
    return self.grid == self._empty_value

# 5. Update agent placement
def place_agent(self, pos, agent_type="agent.agent"):
    if self._grid_is_int:
        if agent_type == "agent.agent":
            self.grid[pos] = ObjectTypes.AGENT_DEFAULT
        else:
            try:
                type_mapping = TypeMapping()
                type_id = type_mapping.get_type_id(agent_type)
                self.grid[pos] = type_id
            except (KeyError, ImportError):
                self.grid[pos] = ObjectTypes.AGENT_DEFAULT
    else:
        self.grid[pos] = agent_type
```

## Testing Strategy

1. **Dual Format Testing**: Each updated scene should be tested with both string and int grids
2. **Regression Testing**: Ensure existing functionality is preserved
3. **Performance Testing**: Verify int-based scenes perform better
4. **Integration Testing**: Test scenes within MapGen system

## Validation Checklist

For each migrated scene file:

- [ ] Imports `ObjectTypes`
- [ ] Adds `__init__` with format detection
- [ ] Updates all direct grid assignments
- [ ] Updates all grid comparisons
- [ ] Handles agent placement appropriately
- [ ] Includes error handling for unknown objects
- [ ] Maintains backward compatibility
- [ ] Passes both string and int grid tests
- [ ] Includes migration note in docstring

## Benefits After Migration

1. **Memory Efficiency**: ~95% memory reduction for int-based grids
2. **Performance**: Faster integer comparisons vs string comparisons
3. **Type Safety**: Constants prevent typos and runtime errors
4. **Maintainability**: Centralized object type definitions
5. **Validation**: Type checking against available objects

## Next Steps

1. Begin with high-priority core scenes
2. Create comprehensive test suite for dual-format support
3. Update MapGen system to handle both grid types
4. Provide migration documentation and examples
5. Performance benchmark before/after migration





