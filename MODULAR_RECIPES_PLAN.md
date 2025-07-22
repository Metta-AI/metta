# Modular Recipes Implementation Plan for Converter Objects

## Executive Summary

This document outlines a plan to extend the current converter system in Metta to support modular, time-based recipe sequences. Instead of having a single fixed recipe, converters will be able to cycle through multiple recipe states, each with its own duration, inputs, and outputs.

## Current System Analysis

### Architecture Overview

The current converter system consists of:

1. **Converter Class** (`mettagrid/src/metta/mettagrid/objects/converter.hpp`)
   - Inherits from `HasInventory`
   - Fixed `input_resources` and `output_resources` maps
   - Two boolean states: `converting` and `cooling_down`
   - Single recipe configuration

2. **Event System** (`mettagrid/src/metta/mettagrid/event.hpp`)
   - `EventManager` schedules and processes events
   - Two converter events: `FinishConverting` and `CoolDown`
   - Events processed each timestep

3. **Configuration** (`mettagrid/src/metta/mettagrid/mettagrid_config.py`)
   - Static YAML configuration
   - One recipe per converter type
   - Fixed timing parameters

### Current Flow

1. Agent puts items → `maybe_start_converting()` checks conditions
2. If valid, consumes inputs → schedules `FinishConverting` event
3. After `conversion_ticks`, produces outputs → schedules `CoolDown` event
4. After `cooldown` ticks, ready to convert again

### Key Dependencies

- **Actions**: `PutRecipeItems`, `GetOutput`
- **Event Handlers**: `ProductionHandler`, `CoolDownHandler`
- **Configuration**: `ConverterConfig`, YAML parsing
- **Core Systems**: Grid, inventory management, observation encoding

## Proposed Design

### Recipe State System

```cpp
struct RecipeState {
    std::map<InventoryItem, InventoryQuantity> input_resources;
    std::map<InventoryItem, InventoryQuantity> output_resources;
    unsigned short duration_ticks;
    std::string state_name;  // For debugging/logging
};

struct RecipeSequence {
    std::vector<RecipeState> states;
    bool loop;  // Whether to restart at state 0 after completing sequence
};
```

### Enhanced Converter Class

```cpp
class ModularConverter : public HasInventory {
private:
    RecipeSequence recipe_sequence;
    size_t current_state_index;
    unsigned short ticks_in_current_state;
    bool is_active;  // Replaces converting/cooling_down booleans

    void advance_to_next_state();
    void process_current_state();
    bool can_execute_current_recipe();
```

### Configuration Schema

```yaml
converter_example:
  type_id: 10
  recipe_sequence:
    loop: true
    states:
      - name: "produce_heart"
        duration_ticks: 20
        input_resources:
          ore_red: 3
        output_resources:
          heart: 1
      - name: "rest"
        duration_ticks: 20
        input_resources: {}
        output_resources:
          heart: 1  # Passive generation
      - name: "produce_battery"
        duration_ticks: 40
        input_resources:
          ore_red: 3
        output_resources:
          battery_red: 2
```

## Implementation Plan

### Phase 1: Core Infrastructure (Week 1)

1. **Create Recipe State Classes**
   - Define `RecipeState` and `RecipeSequence` structures
   - Add serialization/deserialization support
   - Update Python bindings

2. **Extend Configuration System**
   - Update `PyConverterConfig` to support sequences
   - Modify YAML parser to handle new schema
   - Maintain backward compatibility

3. **Refactor Event System**
   - Create new event type: `StateTransition`
   - Update event handlers for state-based processing

### Phase 2: Converter Logic (Week 2)

1. **Implement ModularConverter**
   - Inherit existing converter functionality
   - Add state management logic
   - Handle state transitions

2. **Update State Machine**
   - Replace `maybe_start_converting()` with state-aware logic
   - Implement `advance_to_next_state()`
   - Handle edge cases (empty inventory, invalid states)

3. **Modify Actions**
   - Update `PutRecipeItems` to check current state's requirements
   - Ensure `GetOutput` works with dynamic outputs

### Phase 3: Integration & Testing (Week 3)

1. **Observation System**
   - Update observations to show current state
   - Add state index/name to observation tokens
   - Update `recipe_details_obs` for current state

2. **Backward Compatibility**
   - Create adapter for old configs
   - Ensure existing maps work unchanged
   - Add migration utilities

3. **Testing Suite**
   - Unit tests for state transitions
   - Integration tests with agents
   - Performance benchmarks

### Phase 4: Advanced Features (Week 4)

1. **Input Modifiers System**
   - Catalyst items that affect conversion speed
   - Weight modifiers for probabilistic recipe selection
   - Optional vs required inputs

2. **Decay System**
   - Item decay timers and lifecycle management
   - Decay events and cascading effects
   - Visual indicators for expiring items

3. **State Persistence & Visualization**
   - Save/load converter states and decay timers
   - Replay compatibility with decay events
   - Update renderer for state display and decay warnings

## Design Decisions & Trade-offs

### Option 1: Extend Existing Converter (Recommended)

**Pros:**
- Maintains backward compatibility
- Reuses existing infrastructure
- Minimal changes to actions/observations

**Cons:**
- More complex converter class
- Potential performance overhead
- Legacy code constraints

### Option 2: New Converter Types

**Pros:**
- Clean implementation
- Type-specific optimizations
- Clear separation of concerns

**Cons:**
- Code duplication
- Migration challenges
- Increased maintenance burden

### Option 3: Plugin-Based Recipe System

**Pros:**
- Maximum flexibility
- Runtime recipe loading
- Easy experimentation

**Cons:**
- Complex architecture
- Performance concerns
- Debugging difficulty

## Implementation Challenges

1. **State Synchronization**
   - Ensuring deterministic behavior
   - Handling mid-state saves/loads
   - Network synchronization
   - Replay compatibility (grid object state tracking)

2. **Performance**
   - State checks every tick
   - Memory overhead per converter
   - Observation token limits
   - Benchmark target: < 5% overhead on step performance
   - Current baseline: ~20,000 agent steps/second (4 agents)

3. **User Experience**
   - Clear error messages
   - Intuitive configuration
   - Migration tools

4. **Replay System Integration**
   - Grid objects tracked via `_seq_key_merge` in replay writer
   - State changes need to be observable in replays
   - Current state index must be included in grid object data
   - Recipe transitions visible in MettaScope visualization
   - Decay timers must be included in replay data

5. **Risk Mitigation**
   - **Complexity Creep**: Start with simple features, add complexity only where it adds clear value
   - **Balance Issues**: Extensive playtesting needed for decay timers and modifier values
   - **Learning Curve**: Ensure basic converters still work simply; complexity is opt-in
   - **Debugging**: Comprehensive logging for state transitions and decay events

## Alternative Approaches

### 1. Time-Based Recipe Selection

Instead of sequential states, use time-based functions:

```yaml
converter:
  recipes:
    - time_range: [0, 100]
      recipe: {input: {ore: 1}, output: {heart: 1}}
    - time_range: [100, 200]
      recipe: {input: {}, output: {battery: 1}}
```

### 2. Probabilistic Recipes

Random recipe selection each cycle:

```yaml
converter:
  recipes:
    - weight: 0.7
      recipe: {input: {ore: 3}, output: {heart: 1}}
    - weight: 0.3
      recipe: {input: {ore: 1}, output: {battery: 3}}
```

### 3. Input-Dependent Recipes

Recipe selection based on available inputs:

```yaml
converter:
  recipes:
    - condition: "has_ore >= 3"
      recipe: {input: {ore: 3}, output: {heart: 1}}
    - condition: "has_battery >= 1"
      recipe: {input: {battery: 1}, output: {laser: 1}}
```

## Advanced Concepts: Input Modifiers & Timed Destruction

### 1. Input Items as Production Modifiers

Input items can influence converter behavior beyond simple consumption:

#### Speed Modifiers (Catalysts)
```yaml
converter:
  base_conversion_ticks: 40
  speed_modifiers:
    battery_red: 0.5  # Halves production time
    ore_blue: 0.75    # 25% faster
  recipe:
    input: {ore_red: 3}
    output: {heart: 1}
    optional_catalyst: [battery_red, ore_blue]  # Not consumed
```

#### Recipe Selection Probability
```yaml
converter:
  recipe_selection: probabilistic
  recipes:
    - base_weight: 0.5
      weight_modifiers:
        ore_blue: +0.3    # Increases chance when ore_blue present
        battery_red: -0.2  # Decreases chance
      recipe: {input: {ore_red: 3}, output: {heart: 1}}
    - base_weight: 0.5
      weight_modifiers:
        battery_red: +0.4
      recipe: {input: {ore_red: 1}, output: {battery_red: 2}}
```

**Implementation Benefits:**
- Creates strategic decisions about resource usage
- Enables "tech tree" progression (better catalysts = better efficiency)
- Adds emergent complexity without UI complexity

### 2. Timed Item Destruction (Decay System)

Items can have limited lifespans, creating timing-based gameplay:

```yaml
converter:
  type_id: 15
  recipe_sequence:
    states:
      - name: "produce_volatile_heart"
        duration_ticks: 20
        input_resources: {ore_red: 5}
        output_resources: {heart: 1}
        output_properties:
          heart:
            decay_ticks: 100  # Heart disappears after 100 ticks
            decay_warning: 20   # Visual warning at 20 ticks remaining
```

#### On/Off Timing Components
```yaml
timing_converter:
  recipe_sequence:
    states:
      - name: "charge"
        duration_ticks: 50
        input_resources: {battery_red: 1}
        output_resources: {timing_crystal: 1}
        output_properties:
          timing_crystal:
            decay_ticks: 200
            on_decay_event: "pulse"  # Triggers nearby converters
      - name: "wait"
        duration_ticks: 150
        input_resources: {}
        output_resources: {}
```

**Use Cases:**
1. **Synchronization Puzzles**: Multiple converters must be activated in sequence
2. **Resource Pressure**: Forces immediate use of perishable items
3. **Timing Gates**: Items that exist only during specific windows
4. **Cascading Systems**: Decay events trigger other converters

### 3. Combined System Example

```yaml
advanced_converter:
  type_id: 20
  base_conversion_ticks: 60

  # Speed modifiers from inventory
  speed_modifiers:
    catalyst_blue: 0.5     # Halves time
    catalyst_green: 0.75   # 25% faster

  # Recipe states with modifiers
  recipe_sequence:
    loop: true
    selection_mode: "weighted"  # or "sequential", "conditional"

    states:
      - name: "volatile_production"
        base_duration_ticks: 40
        base_weight: 1.0
        weight_modifiers:
          ore_blue: +0.5
        input_resources: {ore_red: 3}
        output_resources: {heart: 1}
        output_properties:
          heart:
            decay_ticks: 150
            decay_into: {ore_red: 1}  # Returns some resources

      - name: "stable_production"
        base_duration_ticks: 80
        base_weight: 1.0
        weight_modifiers:
          battery_red: +1.0
        input_resources: {ore_red: 5, battery_red: 1}
        output_resources: {heart: 2}
        # No decay - permanent hearts
```

### Implementation Considerations

1. **Inventory Extension**
   ```cpp
   struct InventoryItemState {
       InventoryQuantity quantity;
       std::vector<uint16_t> decay_timers;  // Per-item decay tracking
       ItemProperties properties;
   };
   ```

2. **Decay Event System**
   - New event type: `ItemDecay`
   - Track items with decay timers in grid
   - Process decay before regular events each tick
   - Global decay registry for efficient processing

3. **Visual Feedback**
   - Decaying items show countdown/progress bar
   - Different colors for decay stages
   - Warning effects near expiration
   - Observation tokens include decay state

4. **Performance Impact**
   - Decay checking: O(n) where n = items with decay
   - Can optimize with priority queue for next decay event
   - Negligible impact if decay items are rare
   - Batch process all decay events per tick

5. **Grid Integration**
   ```cpp
   class GridDecayManager {
       // Track all items with decay across all inventories
       std::priority_queue<DecayEvent> decay_queue;

       void register_item(GridObjectId owner, InventoryItem item,
                         uint16_t decay_time, uint32_t current_tick);
       void process_decay_events(uint32_t current_tick);
   };
   ```

## Metrics for Success

1. **Functionality**
   - All example scenarios work
   - No regression in existing maps
   - Deterministic behavior

2. **Performance**
   - < 5% overhead vs current system
   - Memory usage within limits
   - Smooth gameplay at scale

3. **Usability**
   - Clear documentation
   - Intuitive configuration
   - Helpful error messages

## Key Code Locations for Implementation

### Core Files to Modify

1. **Converter Definition**
   - `mettagrid/src/metta/mettagrid/objects/converter.hpp` - Main converter class
   - `mettagrid/src/metta/mettagrid/objects/converter.cpp` (create if needed)

2. **Configuration System**
   - `mettagrid/src/metta/mettagrid/mettagrid_config.py` - Python config classes
   - `mettagrid/src/metta/mettagrid/mettagrid_c_config.py` - Config conversion
   - `mettagrid/src/metta/mettagrid/mettagrid_c.pyi` - Type stubs

3. **Event System**
   - `mettagrid/src/metta/mettagrid/objects/constants.hpp` - Add StateTransition event
   - `mettagrid/src/metta/mettagrid/objects/production_handler.hpp` - Modify handlers

4. **Actions**
   - `mettagrid/src/metta/mettagrid/actions/put_recipe_items.hpp` - State-aware input checking
   - `mettagrid/src/metta/mettagrid/actions/get_output.hpp` - Handle dynamic outputs

5. **Python Bindings**
   - `mettagrid/src/metta/mettagrid/mettagrid_c.cpp` - Update bindings (lines 921-955)

6. **Observation System**
   - `mettagrid/src/metta/mettagrid/observation_encoder.hpp` - Add state tokens
   - `mettagrid/src/metta/mettagrid/objects/constants.hpp` - New observation features

### Testing Files

1. **Unit Tests**
   - `mettagrid/tests/test_converter.py` - Extend for state transitions
   - `mettagrid/tests/test_mettagrid.cpp` - C++ converter tests

2. **Integration Tests**
   - Create `mettagrid/tests/test_modular_recipes.py`

3. **Performance Tests**
   - `mettagrid/benchmarks/test_mettagrid_env_benchmark.py` - Add converter benchmarks

### Configuration Examples

1. **YAML Configs**
   - `configs/env/mettagrid/game/objects/modular_converters.yaml` (create)
   - Update existing converter configs for backward compatibility

## Strategic Implications of Advanced Features

### Input Modifiers Create Depth
- **Resource Management**: Players must decide whether to use items for immediate conversion or save them as catalysts
- **Exploration Rewards**: Finding rare catalysts becomes valuable for optimizing production chains
- **Emergent Strategies**: Different catalyst combinations lead to different optimal paths

### Decay System Enables New Mechanics
- **Time Pressure**: Perishable resources force quick decision-making
- **Synchronization Puzzles**: Coordinating multiple decaying items creates complex challenges
- **Risk/Reward**: Volatile high-value items vs stable low-value items
- **Living Systems**: Converters that must be "fed" regularly to maintain production

### Combined Impact
The combination of modular recipes, input modifiers, and decay creates a rich ecosystem where:
1. **Planning Matters**: Understanding timing and decay helps optimize resource flow
2. **Adaptation Required**: Changing converter states force dynamic strategies
3. **Cooperation Enhanced**: Team coordination becomes crucial for complex timing
4. **Skill Expression**: Mastery involves understanding all system interactions

### Example Scenario: The Synchronized Factory

Consider a puzzle where players must produce exactly 5 hearts within a narrow time window:

1. **Timing Converter**: Produces a `timing_crystal` that decays in 200 ticks
2. **Fast Converter**: When `timing_crystal` is present, produces hearts in 10 ticks (vs normal 50)
3. **Volatile Converter**: Produces `volatile_hearts` that decay in 100 ticks
4. **Stabilizer**: Converts `volatile_hearts` + `battery` → permanent `hearts` (30 ticks)

**Solution requires**:
- Activate Timing Converter first
- Use the timing window to rapidly produce volatile hearts
- Coordinate battery delivery to stabilize hearts before decay
- All while managing limited inventory space

This creates a multi-step puzzle that emerges from simple, modular components.

## Conclusion

The modular recipe system with input modifiers and decay mechanics will transform converters from simple input→output machines into dynamic, strategic gameplay elements. The phased implementation approach allows for iterative development and testing, ensuring stability throughout the process.

The recommended approach (Option 1: Extend Existing Converter) provides the best balance of functionality, compatibility, and implementation complexity. These enhancements enable rich, emergent gameplay while preserving the existing ecosystem of maps and configurations.

The addition of input modifiers and decay systems specifically addresses the need for:
- More strategic depth without UI complexity
- Timing-based challenges and synchronization puzzles
- Resource management decisions beyond simple collection
- Emergent gameplay from simple, composable mechanics
