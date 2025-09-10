# Stochastic Resource Expiry Implementation

This document describes the stochastic resource expiry functionality that has been implemented for both Agent and
Converter classes in the MettaGrid environment.

## Overview

The stochastic resource expiry system allows resources to be lost over time based on configurable probabilities. Each
resource instance has a unique ID and is tracked individually, with loss events scheduled using an exponential
distribution.

## Key Features

### 1. Individual Resource Tracking

- Each resource instance has a unique ID (`uint64_t`)
- Resources are tracked with creation timestep
- Efficient lookup using `std::map` and `std::vector` data structures

### 2. Stochastic Loss

- Loss probability is configurable per resource type (`resource_loss_prob`)
- Uses exponential distribution to determine resource lifetime
- If `resource_loss_prob` is 0, no loss events are scheduled

### 3. Random Removal

- When resources are removed for any reason, they're removed at random
- Uses proper RNG (`std::mt19937`) for random selection
- Maintains consistency between inventory counts and resource instances

### 4. Event-Driven System

- Uses existing `StochasticResourceLoss` event type
- Events are scheduled when resources are created
- Handlers efficiently process loss events using `GridObjectId`

## Implementation Details

### Agent Class Changes

**New Members:**

```cpp
std::mt19937* rng;  // Random number generator
std::map<uint64_t, ResourceInstance> resource_instances;
std::map<InventoryItem, std::vector<uint64_t>> item_to_resources;
uint64_t next_resource_id;
```

**New Methods:**

```cpp
void set_rng(std::mt19937* rng_ptr);
uint64_t create_resource_instance(InventoryItem item_type, unsigned int current_timestep);
void remove_resource_instance(uint64_t resource_id);
uint64_t get_random_resource_id(InventoryItem item_type);
void create_and_schedule_resources(InventoryItem item_type, int count, unsigned int current_timestep);
```

**Modified Methods:**

- `populate_initial_inventory()`: Now creates resource instances for initial inventory
- `update_inventory()`: Handles stochastic resource loss for added/removed items

### Converter Class Changes

**New Members:**

```cpp
std::mt19937* rng;  // Random number generator
std::map<uint64_t, ResourceInstance> resource_instances;
std::map<InventoryItem, std::vector<uint64_t>> item_to_resources;
uint64_t next_resource_id;
```

**New Methods:**

```cpp
void set_rng(std::mt19937* rng_ptr);
uint64_t create_resource_instance(InventoryItem item_type, unsigned int current_timestep);
void remove_resource_instance(uint64_t resource_id);
uint64_t get_random_resource_id(InventoryItem item_type);
void create_and_schedule_resources(InventoryItem item_type, int count, unsigned int current_timestep);
```

**Modified Methods:**

- Constructor: Initializes resource tracking members
- `set_event_manager()`: Creates resource instances for existing inventory
- `update_inventory()`: Handles stochastic resource loss for added/removed items

### MettaGrid Integration

**Changes:**

- Added `converter->set_rng(&_rng);` call in converter creation
- RNG is already set for agents via `add_agent()` method

## Configuration

### Agent Configuration

```python
agent_config = {
    "type_id": 1,
    "type_name": "agent.player",
    "initial_inventory": {"wood": 5, "stone": 3, "food": 2},
    "resource_loss_prob": {"wood": 0.1, "stone": 0.05, "food": 0.2},
    # ... other agent config fields
}
```

### Converter Configuration

```python
converter_config = {
    "type_id": 2,
    "type_name": "mine",
    "input_resources": {},
    "output_resources": {"stone": 1},
    "max_output": 10,
    "max_conversions": -1,
    "conversion_ticks": 2,
    "cooldown": 1,
    "initial_resource_count": 3,
    "resource_loss_prob": {"stone": 0.15},
    # ... other converter config fields
}
```

## Usage Examples

### Setting Resource Loss Probabilities

**High Loss Rate (for testing):**

```python
"resource_loss_prob": {"wood": 0.5, "stone": 0.3, "food": 0.4}
```

**Moderate Loss Rate:**

```python
"resource_loss_prob": {"wood": 0.1, "stone": 0.05, "food": 0.15}
```

**No Loss:**

```python
"resource_loss_prob": {}  # or omit the field entirely
```

### Expected Behavior

1. **Initial Inventory**: When an agent or converter is created with initial inventory, resource instances are created
   and loss events are scheduled.

2. **Adding Resources**: When resources are added to inventory (e.g., through production or collection), new resource
   instances are created and loss events are scheduled.

3. **Removing Resources**: When resources are removed from inventory (e.g., through consumption or trading), random
   resource instances are selected and removed.

4. **Loss Events**: At scheduled timesteps, resources are lost based on the exponential distribution. The loss is
   tracked in statistics.

## Performance Considerations

- **Memory**: Each resource instance uses minimal memory (ID + type + timestep)
- **Lookup**: O(1) average case for resource instance lookups
- **Random Selection**: O(1) for random resource removal
- **Event Scheduling**: O(log n) for event insertion (using existing event system)

## Statistics Tracking

The system tracks resource loss in the existing statistics system:

- `{resource_name}.lost`: Number of resources lost to stochastic expiry
- `{resource_name}.gained`: Number of resources gained
- `{resource_name}.consumed`: Number of resources consumed

## Testing

The implementation has been tested by:

1. Building the C++ code successfully
2. Verifying that resource instances are created and tracked
3. Confirming that loss events are scheduled correctly
4. Ensuring that random removal works as expected

## Future Enhancements

Potential improvements could include:

- Configurable loss distribution (currently exponential)
- Resource quality/durability system
- Batch loss events for performance
- Resource aging effects
- Conditional loss based on environmental factors

## Troubleshooting

**Resources not expiring:**

- Check that `resource_loss_prob` is set and > 0
- Verify that the event manager and RNG are properly initialized
- Ensure the simulation runs for enough timesteps

**Unexpected resource loss:**

- Check that `resource_loss_prob` values are reasonable (0.0 to 1.0)
- Verify that loss events are being scheduled correctly
- Check statistics to see actual loss rates

**Performance issues:**

- Monitor the number of resource instances over time
- Consider reducing loss probabilities if too many events are scheduled
- Check that resources are being removed when consumed (not just lost)
