# Token-Based Observation System

## Overview

MettaGrid implements a **token-based observation system** that provides a compact, efficient way for agents to perceive their environment. Unlike traditional dense grid representations, this system uses sparse token encoding to represent only relevant environmental information, making it highly memory-efficient while maintaining rich semantic information.

## Core Concepts

### Observation Tokens
The fundamental unit of observation is the **ObservationToken**, a compact 3-byte structure:

```cpp
struct alignas(1) ObservationToken {
  ObservationType location = EmptyTokenByte;    // 8 bits: Packed coordinate (y,x)
  ObservationType feature_id = EmptyTokenByte;  // 8 bits: Feature type identifier
  ObservationType value = EmptyTokenByte;      // 8 bits: Feature value
};
```

**Total size**: 3 bytes per token (guaranteed by `static_assert`)

### Packed Coordinate System
Coordinates are efficiently packed into a single byte using 4 bits each for row and column:

```cpp
// Upper 4 bits: row (y-coordinate)
// Lower 4 bits: column (x-coordinate)
// Maximum coordinate: 14 (4 bits = 0-14 range)
// Special value: 0xFF = empty/invalid coordinate
```

**Example packing**:
- Position (5, 3) → `0x53` (0101 0011 in binary)
- Empty position → `0xFF`

## Token Generation Process

### 1. Grid Object Features
Every `GridObject` (agents, walls, resources, etc.) implements `obs_features()`:

```cpp
virtual std::vector<PartialObservationToken> obs_features() const {
    return {};  // Default: no observable features
}
```

### 2. Agent Token Generation
Agents generate rich feature tokens including:

```cpp
std::vector<PartialObservationToken> Agent::obs_features() const {
    // Core agent features
    features.push_back({ObservationFeature::TypeId, type_id});
    features.push_back({ObservationFeature::Group, group});
    features.push_back({ObservationFeature::Frozen, frozen != 0});
    features.push_back({ObservationFeature::Orientation, orientation});
    features.push_back({ObservationFeature::Color, color});

    // Optional glyph feature
    if (glyph != 0) {
        features.push_back({ObservationFeature::Glyph, glyph});
    }

    // Inventory items (dynamic count)
    for (const auto& [item, amount] : inventory) {
        auto feature_id = InventoryFeatureOffset + item;
        features.push_back({feature_id, amount});
    }

    return features;
}
```

### 3. Observation Encoding
The `ObservationEncoder` combines object features with spatial information:

```cpp
size_t encode_tokens(const GridObject* obj, ObservationTokens tokens, ObservationType location) {
    auto features = obj->obs_features();
    return append_tokens_if_room_available(tokens, features, location);
}
```

## Feature Type System

### Core Observation Features
The system defines standardized feature types in the `ObservationFeature` namespace:

```cpp
namespace ObservationFeature {
    constexpr ObservationType TypeId = 0;                    // Object type identifier
    constexpr ObservationType Group = 1;                     // Agent group affiliation
    constexpr ObservationType Hp = 2;                        // Health points
    constexpr ObservationType Frozen = 3;                    // Freeze status (0/1)
    constexpr ObservationType Orientation = 4;               // Facing direction
    constexpr ObservationType Color = 5;                     // Visual color identifier
    constexpr ObservationType ConvertingOrCoolingDown = 6;   // Processing status
    constexpr ObservationType Swappable = 7;                 // Movement capability
    constexpr ObservationType EpisodeCompletionPct = 8;      // Progress indicator
    constexpr ObservationType LastAction = 9;                // Previous action taken
    constexpr ObservationType LastActionArg = 10;            // Action parameter
    constexpr ObservationType LastReward = 11;               // Previous reward received
    constexpr ObservationType Glyph = 12;                    // Visual glyph identifier
    constexpr ObservationType ResourceRewards = 13;          // Resource reward values
    constexpr ObservationType VisitationCounts = 14;         // Location visit frequency

    constexpr ObservationType ObservationFeatureCount = 15;  // Total core features
}
```

### Dynamic Inventory Features
Inventory items are mapped to feature IDs starting after core features:

```cpp
const ObservationType InventoryFeatureOffset = ObservationFeature::ObservationFeatureCount;

// Example: Wood (item 0) → feature ID 15
// Example: Stone (item 1) → feature ID 16
auto item_feature = InventoryFeatureOffset + item_id;
```

### Recipe Features (Optional)
When recipe observation is enabled, additional features are added:

```cpp
// Input recipe features: what items are required
const ObservationType input_recipe_offset = InventoryFeatureOffset + inventory_item_count;

// Output recipe features: what items are produced
const ObservationType output_recipe_offset = input_recipe_offset + inventory_item_count;
```

## Feature Normalization

Each feature type has an associated normalization factor for scaling:

```cpp
inline const std::map<ObservationType, float>& GetFeatureNormalizations() {
    static const std::map<ObservationType, float> feature_normalizations = {
        {ObservationFeature::LastAction, 10.0},           // Max action ID
        {ObservationFeature::LastActionArg, 10.0},        // Max action parameter
        {ObservationFeature::EpisodeCompletionPct, 255.0}, // Progress percentage
        {ObservationFeature::LastReward, 100.0},          // Reward range
        {ObservationFeature::TypeId, 1.0},                // Usually 0-1
        {ObservationFeature::Group, 10.0},                // Group identifier range
        {ObservationFeature::Hp, 30.0},                   // Health point range
        {ObservationFeature::Frozen, 1.0},                // Binary (0/1)
        {ObservationFeature::Orientation, 1.0},           // Usually 0-3
        {ObservationFeature::Color, 255.0},               // Color palette size
        {ObservationFeature::ConvertingOrCoolingDown, 1.0}, // Binary status
        {ObservationFeature::Swappable, 1.0},             // Binary capability
        {ObservationFeature::Glyph, 255.0},               // Glyph identifier range
        {ObservationFeature::ResourceRewards, 255.0},     // Reward value range
        {ObservationFeature::VisitationCounts, 1000.0},   // Visit count maximum
    };
    return feature_normalizations;
}
```

## Python Interface

### Token Types Constants
Python provides access to token type constants:

```python
@dataclass
class TokenTypes:
    # Core observation features
    TYPE_ID_FEATURE: int = 0
    GROUP: int = 1
    HP: int = 2
    FROZEN: int = 3
    ORIENTATION: int = 4
    COLOR: int = 5
    CONVERTING_OR_COOLING_DOWN: int = 6
    SWAPPABLE: int = 7
    EPISODE_COMPLETION_PCT: int = 8
    LAST_ACTION: int = 9
    LAST_ACTION_ARG: int = 10
    LAST_REWARD: int = 11
    GLYPH: int = 12
    RESOURCE_REWARDS: int = 13
    VISITATION_COUNTS: int = 14

    # Object type identifiers
    WALL_TYPE_ID: int = 1
    ALTAR_TYPE_ID: int = 10

    # Special values
    EMPTY_TOKEN = [0xFF, 0xFF, 0xFF]  # 3 bytes
    OBS_TOKEN_SIZE = 3
```

### Observation Helper Utilities
Python provides helper functions for working with token observations:

```python
class ObservationHelper:
    @staticmethod
    def find_tokens_at_location(obs: np.ndarray, x: int, y: int) -> np.ndarray:
        """Find all tokens at a specific location."""

    @staticmethod
    def find_tokens_by_type(obs: np.ndarray, type_id: int) -> np.ndarray:
        """Find all tokens of a specific type."""

    @staticmethod
    def find_feature_at_location(obs: np.ndarray, x: int, y: int, feature_type_id: int) -> np.ndarray:
        """Find specific feature type at location."""

    @staticmethod
    def get_wall_positions(obs: np.ndarray) -> list[tuple[int, int]]:
        """Extract all wall positions from observation."""

    @staticmethod
    def has_wall_at(obs: np.ndarray, x: int, y: int) -> bool:
        """Check if wall exists at position."""
```

## Data Flow Architecture

### Token Generation Pipeline
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Grid Objects  │───▶│  Feature Tokens  │───▶│   Observation   │
│   (Agents,      │    │  (PartialObs)    │    │   Tokens        │
│    Walls, etc.) │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Object State  │    │   Spatial Info   │    │   Agent Input   │
│   (Health, Inv) │    │   (Coordinates)  │    │   (Training)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Agent Perception Process
1. **Environment Scan**: Agent scans visible grid cells
2. **Object Query**: For each occupied cell, query object's `obs_features()`
3. **Token Assembly**: Combine features with packed coordinates
4. **Normalization**: Apply feature-specific normalization
5. **Agent Input**: Provide normalized token array to agent

## Advantages of Token-Based System

### Memory Efficiency
- **Sparse Representation**: Only encode occupied/non-empty cells
- **Compact Storage**: 3 bytes per feature vs dense grid overhead
- **Scalable**: Memory usage grows with complexity, not grid size

### Semantic Richness
- **Feature Diversity**: Different object types can have different features
- **Dynamic Features**: Inventory items create features dynamically
- **Contextual Information**: Rich metadata beyond just presence/absence

### Flexibility
- **Extensible**: Easy to add new feature types without changing core structure
- **Configurable**: Feature normalization can be adjusted per environment
- **Backward Compatible**: Core features remain stable while new ones can be added

## Example Token Observation

Consider an agent at position (5, 3) with the following properties:
- Type ID: 2 (independent agent)
- Group: 1
- Health: 25/30
- Frozen: false
- Orientation: North (0)
- Color: 128
- Inventory: 5 wood, 3 stone

**Generated tokens**:
```
| Location | Feature ID | Value | Description              |
| -------- | ---------- | ----- | ------------------------ |
| 0x53     | 0          | 2     | Type ID at (5,3)         |
| 0x53     | 1          | 1     | Group at (5,3)           |
| 0x53     | 2          | 25    | Health at (5,3)          |
| 0x53     | 3          | 0     | Frozen status at (5,3)   |
| 0x53     | 4          | 0     | Orientation at (5,3)     |
| 0x53     | 5          | 128   | Color at (5,3)           |
| 0x53     | 15         | 5     | Wood inventory at (5,3)  |
| 0x53     | 16         | 3     | Stone inventory at (5,3) |
```

## Integration with Agent Training

### PyTorch Processing
```python
def process_token_observation(tokens: np.ndarray) -> torch.Tensor:
    """Convert token observation to agent input format."""

    # Extract different feature types
    agent_features = extract_features_by_type(tokens, TokenTypes.TYPE_ID_FEATURE)
    health_features = extract_features_by_type(tokens, TokenTypes.HP)
    inventory_features = extract_inventory_features(tokens)

    # Normalize features
    normalized = apply_normalization(agent_features, health_features, inventory_features)

    return normalized
```

### Training Pipeline Integration
1. **Raw Tokens**: Environment provides token observation
2. **Feature Extraction**: Parse tokens by type and location
3. **Normalization**: Apply feature-specific scaling
4. **Embedding**: Convert to agent input format
5. **Training**: Feed into neural network

## Performance Characteristics

### Memory Usage Comparison
- **Token System**: ~3 bytes × number of features × number of objects
- **Dense Grid**: 4 bytes × grid_width × grid_height × num_channels
- **Typical Ratio**: 10-100x memory reduction for sparse environments

### Query Performance
- **Spatial Queries**: O(n) where n = number of tokens (fast for sparse data)
- **Feature Queries**: O(n) with efficient filtering
- **Location Lookup**: O(1) with packed coordinate indexing

### Scalability
- **Large Grids**: Memory usage independent of grid size
- **Complex Objects**: Feature count scales with object complexity
- **Multi-Agent**: Each agent gets localized token observations

## Limitations and Trade-offs

### Sparse Nature
- **No Empty Cell Information**: Cannot distinguish unexplored from empty cells
- **Limited Spatial Context**: No information about unoccupied areas
- **Distance Calculations**: Requires coordinate unpacking for spatial reasoning

### Feature Complexity
- **Parsing Overhead**: Token parsing more complex than direct array access
- **Feature Coordination**: Need to maintain feature ID consistency across versions
- **Normalization Management**: Each feature type needs appropriate scaling

## Future Extensions

### Enhanced Spatial Information
- **Distance Encoding**: Add relative distance features
- **Visibility Masks**: Include line-of-sight information
- **Terrain Analysis**: Add local terrain features

### Temporal Features
- **Change Detection**: Track changes between observations
- **Memory Integration**: Add temporal decay to token values
- **Prediction Features**: Include predicted future states

### Multi-Modal Integration
- **Visual Features**: Integrate with image-based observations
- **Audio Features**: Add sound-based token types
- **Tactile Features**: Include touch/proximity information

## Conclusion

The token-based observation system provides a **highly efficient and semantically rich** way to represent environmental information for agent training. Its sparse encoding, extensible feature system, and compact memory footprint make it particularly well-suited for:

- **Complex Environments**: Rich feature sets without dense grid overhead
- **Scalable Training**: Memory usage grows with complexity, not grid size
- **Research Flexibility**: Easy to add new observation types and features
- **Multi-Agent Scenarios**: Each agent can receive customized observations

The system's design balances **memory efficiency**, **semantic richness**, and **extensibility**, making it an excellent foundation for advanced agent training scenarios in complex, dynamic environments.
