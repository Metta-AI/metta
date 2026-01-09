# MettaGrid Observation System - Technical Manual

This document provides a technical explanation of the observation format that policies should expect when interacting
with MettaGrid. It covers the token-based observation structure, coordinate encoding, feature types, and how
observations are generated from grid objects.

> For information about using observations via the Python Simulator API (e.g., `agent.observation`, `ObservationToken`,
> `AgentObservation`), see [simulator_api.md](simulator_api.md#observation-system).

## Overview

MettaGrid uses a **token-based observation system** where observations are represented as a variable-length sequence of
fixed-length tokens. Each token encodes a single feature value at a specific location within an agent's observation
window. Generally only non-zero-valued tokens will be emitted. E.g., if an agent has none of a particular resource, no
token will be emitted for this observation.

> These tokens are not to be confused with tokens used in a transformer architecture. When there might be ambiguity, we
> try to use the term Observation Token.

## Observation Token Structure

### Observation Array Format

Observations are provided as NumPy arrays with shape `(num_agents, num_tokens, 3)`:

- **Dimension 0**: Agent index
- **Dimension 1**: Token index (variable length, padded with empty tokens)
- **Dimension 2**: Token components `[location, feature_id, value]`

### Empty Tokens

- **Empty token byte**: `0xFF` (255)
- When `location == 0xFF`, the token is considered empty/invalid
- Empty tokens are used to pad observation arrays to a fixed size
- Policies should ignore tokens where `location == 0xFF`

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

## Observation Features

Feature IDs are assigned sequentially starting from 0. The exact feature set depends on the game configuration
(resources, protocol details, etc.). To determine which features are available in a specific game configuration, use the
`IdMap` class.

> feature_ids used to represent these features may be different between configurations. Policy authors should either
> ensure that the same feature configuration exists between training and evaluation, or should ensure that trained
> policies can appropriately deal with changing feature_ids (e.g., by accepting an IdMap and using this to set up an
> appropriate front-end permutation layer).

### Getting Feature Information from Configuration

The `IdMap` class provides access to all observation features for a given game configuration. You can obtain an `IdMap`
from a `GameConfig`:

```python
from mettagrid.config.mettagrid_config import MettaGridConfig

# Create or load a configuration
config = MettaGridConfig(...)

# Get the IdMap for this configuration
id_map = config.game.id_map()

# Get all observation features
features = id_map.features()

# Each feature is an ObservationFeatureSpec with:
# - id: int - The feature ID used in observation tokens
# - name: str - Human-readable feature name (e.g., "inv:food", "agent:group")
# - normalization: float - Normalization factor for this feature
```

### ObservationFeatureSpec

Each feature is represented by an `ObservationFeatureSpec` object:

```python
class ObservationFeatureSpec:
    id: int           # Feature ID (0, 1, 2, ...).
    name: str         # Feature name (e.g., "inv:food", "agent:group")
    normalization: float  # Normalization factor (e.g., 100.0, 255.0)
```

### Feature Set Structure

The feature set follows this general structure (exact IDs depend on configuration). Note that the specific feature IDs
depend on your game configuration (number of resources, whether protocol details are enabled, etc.), so always use
`IdMap` to get the exact feature IDs for your configuration.

| Feature Name                      | Description                                                                     | Objects with this Feature | Notes                                                                           |
| --------------------------------- | ------------------------------------------------------------------------------- | ------------------------- | ------------------------------------------------------------------------------- |
| `agent:group`                     | Agent's group/team identifier                                                   | agents                    |                                                                                 |
| `agent:frozen`                    | Whether agent is frozen or not. Frozen agents cannot act                        | agents                    |                                                                                 |
| `episode_completion_pct`          | Portion of the episode completed, from 0 (start) to 255 (end). Not a percentage | self                      |                                                                                 |
| `last_action`                     | Last action taken by the agent                                                  | self                      |                                                                                 |
| `last_reward`                     | Last reward received by the agent                                               | self                      |                                                                                 |
| `vibe`                            | Agent's current vibe                                                            | any object                | Values can be found in `vibes.VIBES`                                            |
| `agent:compass`                   | Compass direction toward assembler                                              | self                      |                                                                                 |
| `tag`                             | Tags associated with an object (e.g., "wall", "oxygen_extractor", "blue")       | any object                | Values can be found in `IdMap.tag_names()`. Multiple tags emit multiple tokens. |
| `cooldown_remaining`              | Remaining cooldown time for objects                                             | assembler, extractors     | Value capped at 255                                                             |
| `clipped`                         | Whether an assembler is clipped or not                                          | extractors                |                                                                                 |
| `remaining_uses`                  | Remaining uses for objects with use limits                                      | extractors                | Value capped at 255. Only emitted if `max_uses > 0`                             |
| `inv:{resource_name}`             | Base inventory amount (amount % token_value_base)                               | agents, chests            | One feature per resource. See [Inventory Encoding](#inventory-encoding) below.  |
| `inv:{resource_name}:p1`          | Power 1 component ((amount / B) % B)                                            | agents, chests            | Only emitted if amount >= B. See [Inventory Encoding](#inventory-encoding).     |
| `inv:{resource_name}:p2`          | Power 2 component ((amount / B²) % B)                                           | agents, chests            | Only emitted if amount >= B². See [Inventory Encoding](#inventory-encoding).    |
| `protocol_input:{resource_name}`  | Required input resource amount for current protocol                             | assembler, extractors     | One feature per resource                                                        |
| `protocol_output:{resource_name}` | Output resource amount for current protocol                                     | assembler, extractors     | One feature per resource                                                        |

### Inventory Encoding

Inventory values are encoded using a multi-token scheme with a configurable base (`ObsConfig.token_value_base`, default
256). This allows representing large amounts while keeping individual token values bounded. The number of tokens is
dynamically computed based on the maximum inventory value (uint16_t max = 65535).

- **`inv:{resource}`**: Base value = `amount % B` (always emitted if amount > 0)
- **`inv:{resource}:p1`**: Power 1 = `(amount / B) % B` (only emitted if amount >= B)
- **`inv:{resource}:p2`**: Power 2 = `(amount / B²) % B` (only emitted if amount >= B²)
- etc.

Where B = `token_value_base` (default 256).

The full value is reconstructed as: `base + p1 * B + p2 * B² + ...`

**Examples with token_value_base=256 (default):**

| Amount | `inv:food` | `inv:food:p1` | `inv:food:p2` | Reconstruction           |
| ------ | ---------- | ------------- | ------------- | ------------------------ |
| 42     | 42         | (not emitted) | (not emitted) | 42                       |
| 1234   | 210        | 4             | (not emitted) | 210 + 4 \* 256 = 1234    |
| 65535  | 255        | 255           | (not emitted) | 255 + 255 \* 256 = 65535 |

**Examples with token_value_base=100:**

| Amount | `inv:food` | `inv:food:p1` | `inv:food:p2` | Reconstruction              |
| ------ | ---------- | ------------- | ------------- | --------------------------- |
| 42     | 42         | (not emitted) | (not emitted) | 42                          |
| 1234   | 34         | 12            | (not emitted) | 34 + 12 \* 100 = 1234       |
| 54321  | 21         | 43            | 5             | 21 + 43 \* 100 + 5 \* 10000 |

The actual maximum is limited by the underlying `InventoryQuantity` type (uint16_t, max 65535).
