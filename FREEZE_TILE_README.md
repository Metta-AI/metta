# Freeze Tile Implementation

## Overview

The freeze tile is a new terrain feature in MettaGrid that affects agent movement behavior. When an agent steps on a freeze tile, it forces the agent to continue moving in the same direction on their next action.

## Behavior

1. **Activation**: When an agent moves onto a freeze tile, the tile captures the direction the agent was moving
2. **Forced Movement**: On the agent's next action, regardless of what action they try to take, they will be forced to move in the captured direction
3. **Release Conditions**:
   - **Case 1**: The agent successfully moves to an empty square (normal release)
   - **Case 2**: The agent cannot move in the forced direction due to an obstacle, which immediately releases them from the freeze effect

## Implementation Details

### Files Modified

- `mettagrid/mettagrid/objects/constants.hpp` - Added `FreezeTileT` object type
- `mettagrid/mettagrid/objects/freeze_tile.hpp` - New freeze tile class implementation
- `mettagrid/mettagrid/objects/agent.hpp` - Added freeze tile state tracking fields
- `mettagrid/mettagrid/actions/move.hpp` - Modified move action to handle freeze tile logic
- `mettagrid/mettagrid/mettagrid_c.cpp` - Added freeze tile parsing for map creation
- `mettagrid/mettagrid/observation_encoder.hpp` - Added freeze tile to observation encoding

### Agent State Fields

Two new fields were added to the Agent class:
- `bool on_freeze_tile` - Tracks if the agent is currently affected by a freeze tile
- `unsigned char freeze_tile_direction` - Stores the forced movement direction

### Movement Logic

The move action was enhanced to:
1. Check if the agent is on a freeze tile
2. If so, override the requested movement direction with the stored freeze direction
3. Handle obstacle collision by releasing the freeze effect
4. Update freeze tile state when moving to new locations

## Usage in Maps

To add freeze tiles to your maps, use the string `"freeze_tile"` in your grid definition:

```python
grid = [
    ["empty", "empty", "empty"],
    ["agent.player", "freeze_tile", "empty"],
    ["empty", "empty", "empty"]
]
```

## Configuration

Freeze tiles require basic configuration in your environment config:

```python
config = {
    "objects": {
        "freeze_tile": {
            "hp": 1
        }
    }
}
```

## Example

See `freeze_tile_demo.py` for a complete example demonstrating the freeze tile behavior.

## Technical Notes

- Freeze tiles are implemented as ground-layer objects that agents can move onto
- The freeze effect only lasts for one action after stepping on the tile
- Agents are immediately released if they cannot move in the forced direction
- The implementation integrates cleanly with the existing MettaGrid action system
