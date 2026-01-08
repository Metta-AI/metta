# TECHNICAL MANUAL: Sensor Systems and Command Protocols

## Field Awareness and Command Engine (FACE) Technical Documentation

Welcome, Cognitive!

This document provides technical specifications for your onboard sensor systems and command protocols. Understanding
these systems is critical for effective operation in the field. Your FACE processes environmental data and translates
your decisions into executable commands.

This report covers the technical details of how you perceive the world and how you issue commands. In different
environments, specific **details** may vary.

---

## SENSOR SYSTEMS OVERVIEW

Your FACE uses a **token-based observation system** to process environmental data. Rather than processing raw visual
feeds, your sensors emit discrete data tokens representing specific features at specific locations within your
observation window. This design allows for efficient processing while maintaining critical spatial awareness.

> **Technical Note**: These observation tokens are distinct from tokens used in transformer architectures. When
> ambiguity might arise, we refer to them as Observation Tokens.

### Observation Token Structure

Your sensor array provides observations as structured data arrays with dimensions `(num_cogs, num_tokens, 3)`:

- **Dimension 0**: Cog index (your position in the team)
- **Dimension 1**: Token index (variable length, padded with empty tokens)
- **Dimension 2**: Token components `[location, feature_id, value]`

Each token encodes a single feature value at a specific location within your observation window.

### Empty Tokens

Your sensors use a special marker value `0xFF` (255) to indicate empty or invalid tokens. When `location == 0xFF`, the
token should be ignored. Empty tokens are used to pad observation arrays to a fixed size for efficient batch processing.

### Coordinate Encoding

Your observation window uses a packed coordinate system to efficiently encode spatial information:

- **Upper 4 bits (high nibble)**: Row coordinate (0-14)
- **Lower 4 bits (low nibble)**: Column coordinate (0-14)
- **Special value `0xFF`**: Empty/invalid coordinate

Observation windows are typically 11x11 centered on your position; you are located at `0x55` (row 5, column 5).
Coordinates are **egocentric** (relative to your position), not absolute map coordinates.

#### Coordinate System

- **Row (r/y)**: Vertical coordinate, increases downward
- **Column (c/x)**: Horizontal coordinate, increases rightward
- **Center location**: Used for cog-specific features (inventory, global state)

### Observation Features

Your sensors can detect various features in the environment. Feature IDs are assigned sequentially starting from 0. The
exact feature set depends on your mission configuration (available resources, protocol details, etc.).

> **Critical**: Feature IDs may differ between mission configurations. Ensure your cognitive models are configured for
> the specific mission parameters, or implement dynamic feature mapping using the `IdMap` system.

#### Getting Feature Information

You can query your FACE for available observation features using the `IdMap` system:

```python
from mettagrid.config.mettagrid_config import MettaGridConfig

# Access your mission configuration
config = MettaGridConfig(...)

# Get the IdMap for this configuration
id_map = config.game.id_map()

# Get all observation features
features = id_map.features()

# Each feature provides:
# - id: int - The feature ID used in observation tokens
# - name: str - Human-readable feature name (e.g., "inv:food", "agent:group")
# - normalization: float - Normalization factor for this feature
```

#### Feature Set Structure

The following features may be available in your sensor data. Note that specific feature IDs depend on your mission
configuration (number of resources, whether protocol details are enabled, etc.), so always use `IdMap` to get the exact
feature IDs for your configuration.

| Feature Name                      | Description                                                                     | Objects with this Feature | Notes                                                                           |
| --------------------------------- | ------------------------------------------------------------------------------- | ------------------------- | ------------------------------------------------------------------------------- |
| `agent:group`                     | Cog's group/team identifier                                                     | cogs                      | Teams not currently in use.                                                     |
| `agent:frozen`                    | Whether cog is frozen or not. Frozen cogs cannot act                            | cogs                      | Freezing not currently in use.                                                  |
| `episode_completion_pct`          | Portion of the episode completed, from 0 (start) to 255 (end). Not a percentage | self                      |                                                                                 |
| `last_action`                     | Last action taken by the cog                                                    | self                      |                                                                                 |
| `last_reward`                     | Last reward received by the cog                                                 | self                      |                                                                                 |
| `vibe`                            | Cog's current vibe                                                              | any object                | Values can be found in `vibes.VIBES`                                            |
| `agent:compass`                   | Compass direction toward assembler                                              | self                      |                                                                                 |
| `tag`                             | Tags associated with an object (e.g., "wall", "oxygen_extractor", "blue")       | any object                | Values can be found in `IdMap.tag_names()`. Multiple tags emit multiple tokens. |
| `cooldown_remaining`              | Remaining cooldown time for objects                                             | assembler, extractors     | Value capped at 255                                                             |
| `clipped`                         | Whether an assembler is clipped or not                                          | extractors                |                                                                                 |
| `remaining_uses`                  | Remaining uses for objects with use limits                                      | extractors                | Value capped at 255. Only emitted if `max_uses > 0`                             |
| `inv:{resource_name}`             | Amount of resource in the object                                                | cogs, chests              | One feature per resource (e.g., `inv:food`, `inv:wood`, `inv:stone`)            |
| `protocol_input:{resource_name}`  | Required input resource amount for current protocol                             | assembler, extractors     | One feature per resource                                                        |
| `protocol_output:{resource_name}` | Output resource amount for current protocol                                     | assembler, extractors     | One feature per resource                                                        |

---

## COMMAND PROTOCOLS

Your FACE translates your decisions into executable commands using a **discrete action space**. Each command is
represented as a single integer index (action ID) that corresponds to a fully qualified action variant such as
`move_north`, `attack_3`, or `change_vibe_happy`.

Verb/argument combinations are flattened during environment initialization, so you only need to emit a scalar
`action_id` per cog.

### Action Format

Commands are provided as structured data arrays with shape `(num_cogs,)`:

- **Type**: `int32` (or `np.int32`)
- **Range**: `0 <= action_id < num_actions`
- **Per cog**: Each cog emits a single action ID

### Action ID Assignment

Action IDs are assigned sequentially starting from 0 based on the order in which actions are registered during
environment initialization. The exact action set depends on your mission configuration (enabled actions, allowed
directions, etc.).

> Action IDs may differ between mission configurations. Ensure your cognitive models are configured for the specific
> mission parameters, or implement dynamic action mapping using action names.

### Available Command Types

The following command types are available in MettaGrid. The exact set depends on your mission configuration.

#### Noop Command

- **Name**: `noop`
- **Description**: Do nothing. Always available and typically has action ID 0.
- **Resource requirements**: None

#### Move Commands

- **Name pattern**: `move_{direction}`
- **Directions**: `north`, `south`, `east`, `west` (and optionally `northeast`, `northwest`, `southeast`, `southwest` if
  diagonals are enabled). Moving into an object triggers interaction with that object.
- **Description**: Move one cell in the specified direction
- **Resource requirements**: Consumes energy (configurable)

Example command names:

- `move_north`
- `move_south`
- `move_east`
- `move_west`

#### Change Vibe Commands

- **Name pattern**: `change_vibe_{vibe_name}`
- **Vibes**: Depends on configuration (e.g., `happy`, `sad`, `angry`, etc.)
- **Description**: Change your current vibe to the specified vibe. Vibes can be used for communication, and also impacts
  environmental interactions.

Example command names:

- `change_vibe_happy`
- `change_vibe_sad`
- `change_vibe_neutral`

> **Note**: The available vibes are configurable via `change_vibe.vibes`.

---

## COMMAND EXECUTION FLOW

When a command is executed, the following validation occurs:

1. **Action Index Validation**: `0 <= action_id < num_actions`
2. **Action Space Validation**: Action ID must be within `env.action_space.n`
3. **Resource Validation**: Cog must have required resources (if any)
4. **Action Execution**: Attempt the action

### Invalid Commands

If a command is invalid (out of range, insufficient resources, etc.), the command is silently ignored and you
effectively perform a noop.

---

## FINAL NOTES

**Understanding your sensor systems and command protocols is essential for effective field operations.**

Your success depends on:

- Proper interpretation of observation tokens
- Efficient command selection and execution
- Understanding the relationship between feature IDs and action IDs
- Adapting to different mission configurations

_Stay aware. Stay coordinated. Stay operational._

---

_END TRANSMISSION_
