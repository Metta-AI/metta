# MettaGrid Observation System - Technical Manual

This document provides a technical explanation of the observation format that policies should expect when interacting
with MettaGrid. It covers the token-based observation structure, coordinate encoding, feature types, and how
observations are generated from grid objects.

## Overview

MettaGrid uses a **token-based observation system** where observations are represented as a variable-length sequence of
tokens. Each token encodes a single feature value at a specific location within an agent's observation window.

Note that these tokens are not to be confused with tokens used in a transformer architecture. When there might be
ambiguity, we try to use the term Observation Token.

## Observation Token Structure

### Binary Format

Each observation token is exactly **3 bytes** in size, with the following structure:

```cpp
struct ObservationToken {
    uint8_t location;    // Packed (row, col) coordinates
    uint8_t feature_id;  // Feature type identifier
    uint8_t value;       // Feature value
};
```

The structure is packed with `alignas(1)` to ensure no padding, guaranteeing exactly 3 bytes per token.

### Empty Tokens

- **Empty token byte**: `0xFF` (255)
- When `location == 0xFF`, the token is considered empty/invalid
- Empty tokens are used to pad observation arrays to a fixed size
- Policies should ignore tokens where `location == 0xFF`

### Observation Array Format

Observations are provided as NumPy arrays with shape `(num_agents, num_tokens, 3)`:

- **Dimension 0**: Agent index
- **Dimension 1**: Token index (variable length, padded with empty tokens)
- **Dimension 2**: Token components `[location, feature_id, value]`

## Coordinate Encoding

### Packed Coordinate Format

Coordinates are packed into a single byte using a nibble-based encoding scheme:

- **Upper 4 bits (high nibble)**: Row coordinate (0-14)
- **Lower 4 bits (low nibble)**: Column coordinate (0-14)
- **Special value `0xFF`**: Empty/invalid coordinate
- Observation windows are typically 11x11 centered on the agent; so the agent is at `0x55`.
- Coordinates are **egocentric** (relative to the agent's position)

### Coordinate System

- **Row (r/y)**: Vertical coordinate, increases downward
- **Column (c/x)**: Horizontal coordinate, increases rightward
- **Center location**: Used for agent-specific features (inventory, global state)

## Observation Feature Types

Feature IDs are assigned sequentially starting from 0. The exact feature set depends on the game configuration, but
follows this general structure:

### Core Features (Always Present)

These features are always included in the feature set:

1. **`agent:group`** (ID: 0)
   - Normalization: 10.0
   - Agent's group/team identifier
   - Value range: 0-10

2. **`agent:frozen`** (ID: 1)
   - Normalization: 1.0
   - Whether agent is frozen (cannot act)
   - Value: 0 (not frozen) or 1 (frozen)

### Global Observation Features

These features appear at the agent's center location and provide global state:

6. **`episode_completion_pct`** (ID: 5)
   - Normalization: 255.0
   - Episode completion percentage (0-255)
   - Only populated if `global_obs.episode_completion_pct` is enabled

7. **`last_action`** (ID: 6)
   - Normalization: 10.0
   - Last action taken by the agent
   - Only populated if `global_obs.last_action` is enabled

8. **`last_reward`** (ID: 8)
   - Normalization: 100.0
   - Last reward received by the agent
   - Only populated if `global_obs.last_reward` is enabled

### Agent-Specific Features

10. **`vibe`** (ID: 9)
    - Normalization: 255.0
    - Agent's current vibe value
    - Only emitted if `vibe > 0`
    - Appears at agent's location

11. **`agent:compass`** (ID: 10)
    - Normalization: 1.0
    - Compass direction toward assembler
    - Only populated if `global_obs.compass` is enabled

### Object Features

12. **`tag`** (ID: 11)
    - Normalization: 10.0
    - Tag ID associated with an object
    - Multiple tags emit multiple tokens with the same feature ID

13. **`cooldown_remaining`** (ID: 12)
    - Normalization: 255.0
    - Remaining cooldown time for objects (assemblers, generators, etc.)
    - Only emitted if `cooldown_remaining > 0`
    - Value capped at 255

14. **`clipped`** (ID: 13)
    - Normalization: 1.0
    - Whether an assembler is clipped (cannot be used)
    - Value: 0 or 1
    - Only emitted if `is_clipped == true`

15. **`remaining_uses`** (ID: 14)
    - Normalization: 255.0
    - Remaining uses for objects with use limits
    - Only emitted if `max_uses > 0`
    - Value capped at 255

### Inventory Features

For each resource in the game configuration, an inventory feature is created:

- **`inv:{resource_name}`** (ID: starts after object features)
  - Normalization: 100.0
  - Amount of resource in agent's inventory
  - Only emitted for non-zero inventory amounts
  - Appears at agent's center location
  - Example: `inv:food`, `inv:wood`, `inv:stone`

### Protocol Detail Features (Optional)

If `protocol_details_obs` is enabled in the configuration, assemblers emit additional features:

- **`protocol_input:{resource_name}`** (ID: after inventory features)
  - Normalization: 100.0
  - Required input resource amount for current protocol
  - Only emitted for non-zero values

- **`protocol_output:{resource_name}`** (ID: after protocol inputs)
  - Normalization: 100.0
  - Output resource amount for current protocol
  - Only emitted for non-zero values

## Observation Generation

### From Grid Objects

Each grid object type implements an `obs_features()` method that returns a vector of `PartialObservationToken`
structures:

```cpp
struct PartialObservationToken {
    ObservationType feature_id;
    ObservationType value;
};
```

The observation encoder then:

1. Takes the object's grid location
2. Converts it to egocentric coordinates relative to the observing agent
3. Packs the coordinates into the `location` byte
4. Creates full `ObservationToken` structures with `(location, feature_id, value)`

### Agent Observations

Agents emit the following features (at their own location):

- `agent:group` - Group ID
- `agent:frozen` - Frozen status (0 or 1)
- `vibe` - Vibe value (if > 0)
- `inv:{resource}` - Inventory items (one token per non-zero resource)
- `tag` - Tag IDs (one token per tag)

### Assembler Observations

Assemblers emit the following features (at their location):

- `cooldown_remaining` - Remaining cooldown (if > 0)
- `clipped` - Clipped status (if clipped)
- `remaining_uses` - Remaining uses (if max_uses > 0)
- `protocol_input:{resource}` - Protocol inputs (if protocol_details_obs enabled)
- `protocol_output:{resource}` - Protocol outputs (if protocol_details_obs enabled)
- `tag` - Tag IDs (one token per tag)
- `vibe` - Vibe value (if > 0)

### Observation Window

Observations are generated for objects within the agent's observation window:

- **Window size**: Configurable via `obs_width` and `obs_height` (typically 11x11)
- **Center**: Agent's current position
- **Coordinate system**: Egocentric (relative to agent)
- **Scanning order**: Objects are scanned in Manhattan distance order (closest first)

## Feature Normalization

Each feature has an associated normalization factor used for scaling values:

- Values should be divided by the normalization factor to get normalized [0, 1] range
- Example: `inv:food` with normalization 100.0 means value 50 represents 0.5 normalized
- Policies typically normalize values: `normalized_value = raw_value / normalization_factor`

## Policy Implementation Guidelines

### Token Processing

1. **Filter empty tokens**: Ignore tokens where `location == 0xFF`
2. **Unpack coordinates**: Extract row/col from packed location byte
3. **Convert to egocentric**: Coordinates are already egocentric relative to agent
4. **Group by location**: Multiple features can appear at the same location
5. **Normalize values**: Divide values by their feature's normalization factor

### Example Token Processing

```python
def process_observations(obs_array: np.ndarray) -> dict:
    """
    obs_array shape: (num_tokens, 3)
    Returns: Dictionary mapping (row, col) -> {feature_name: value}
    """
    result = {}

    for token in obs_array:
        location_byte = token[0]
        feature_id = token[1]
        value = token[2]

        # Skip empty tokens
        if location_byte == 0xFF:
            continue

        # Unpack coordinates
        row = (location_byte >> 4) & 0x0F
        col = location_byte & 0x0F

        # Get feature name from feature_id (requires feature mapping)
        feature_name = feature_id_to_name[feature_id]

        # Store in result
        if (row, col) not in result:
            result[(row, col)] = {}
        result[(row, col)][feature_name] = value

    return result
```

### Common Patterns

1. **Agent self-observation**: Features at center location `(obs_height//2, obs_width//2)`
2. **Spatial features**: Features at non-center locations represent nearby objects
3. **Multiple features per location**: Objects can emit multiple tokens at the same location
4. **Variable token count**: Number of tokens varies based on visible objects and inventory

## Data Types

- **ObservationType**: `uint8_t` (0-255)
- **Location byte**: `uint8_t` (packed coordinates or 0xFF for empty)
- **Feature ID**: `uint8_t` (0-255, but typically much smaller)
- **Value**: `uint8_t` (0-255, feature-dependent)

## References

- **C++ Implementation**: `packages/mettagrid/cpp/include/mettagrid/core/grid_object.hpp`
- **Observation Encoder**: `packages/mettagrid/cpp/include/mettagrid/systems/observation_encoder.hpp`
- **Packed Coordinates**: `packages/mettagrid/cpp/include/mettagrid/systems/packed_coordinate.hpp`
- **Feature Definitions**: `packages/mettagrid/python/src/mettagrid/config/id_map.py`
- **Python Bindings**: `packages/mettagrid/python/src/mettagrid/mettagrid_c.pyi`
