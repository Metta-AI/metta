# Cyclical Converter with Auto-Emptying: Implementation Requirements

## Objective

Implement a converter that cycles through three states:
1. **Converting** (N ticks): Producing a heart
2. **Holding** (M ticks): Heart sits in inventory
3. **Emptying**: Heart is destroyed, cycle restarts

This creates an on/off timing component where hearts are only available during specific windows.

## Current System Analysis

### Existing Converter States
- `converting`: Currently producing output
- `cooling_down`: Waiting before next conversion
- Inventory management: Items stay until taken by agents

### Event System
```cpp
enum EventType {
  FinishConverting = 0,  // Triggers output production
  CoolDown = 1,          // Triggers cooldown completion
  EventTypeCount
};
```

### Current Flow
1. `maybe_start_converting()` → schedules `FinishConverting`
2. `finish_converting()` → adds output, schedules `CoolDown`
3. `finish_cooldown()` → calls `maybe_start_converting()`

## Simplest Possible Implementation

### Ultra-Minimal Approach: State Bit Reuse

Since `cooling_down` already tracks a binary state, we can add meaning to the cooldown phase:

```cpp
// In converter.hpp - modify finish_cooldown()
void finish_cooldown() {
  // NEW: Check if this is an auto-emptying converter
  if (this->type_name.find("cyclical") != std::string::npos) {
    // Empty all output items
    for (const auto& [item, _] : this->output_resources) {
      if (this->inventory.count(item) > 0) {
        this->inventory[item] = 0;
        this->inventory.erase(item);
      }
    }
  }

  // Original behavior continues
  this->cooling_down = false;
  stats.incr("cooldown.completed");
  this->maybe_start_converting();
}
```

**That's it!** Just 10 lines added to one function.

### Even Better: Use Special Type ID Range

Instead of string matching, reserve a type ID range for cyclical converters:

```cpp
// In converter.hpp - modify finish_cooldown()
void finish_cooldown() {
  // NEW: Type IDs 100-199 are cyclical converters
  if (this->type_id >= 100 && this->type_id < 200) {
    // Empty all output items
    this->inventory.clear();  // Even simpler!
  }

  // Original behavior continues
  this->cooling_down = false;
  stats.incr("cooldown.completed");
  this->maybe_start_converting();
}
```

**Only 5 lines!** And faster than string comparison.

**Configuration**:
```yaml
cyclical_altar:
  type_id: 100  # 100-199 = cyclical converters
  input_resources: {}
  output_resources: {heart: 1}
  conversion_ticks: 20  # N ticks to produce heart
  cooldown: 30         # M ticks before auto-empty
  max_output: 1
```

## Minimal Implementation Approach

### Option 1: Reuse Existing Events (Recommended)

**Concept**: Add a small flag to distinguish auto-emptying converters, reuse CoolDown event for emptying.

**Note**: Cannot use negative cooldown as signal since it already means "never convert again".

```cpp
class CyclicalConverter : public Converter {
  bool auto_empty_mode;  // Track if we're in auto-empty cycle

  void finish_converting() override {
    if (auto_empty_mode && cooldown < 0) {
      // Schedule cooldown event that will empty inventory
      this->cooling_down = true;
      this->event_manager->schedule_event(
        EventType::CoolDown,
        -cooldown,  // Use absolute value
        this->id,
        0
      );
    } else {
      // Normal behavior
      Converter::finish_converting();
    }
  }

  void finish_cooldown() override {
    if (auto_empty_mode) {
      // Empty the inventory
      for (auto& [item, _] : output_resources) {
        if (inventory.count(item) > 0) {
          inventory.erase(item);
          stats.incr("items.auto_emptied");
        }
      }
    }
    // Resume converting
    this->cooling_down = false;
    this->maybe_start_converting();
  }
};
```

**Configuration**:
```yaml
cyclical_converter:
  type_id: 20
  input_resources: {}  # No inputs needed
  output_resources: {heart: 1}
  conversion_ticks: 20  # N ticks to produce
  cooldown: -30        # Negative = auto-empty after 30 ticks
  max_output: 1
  initial_resource_count: 0
```

### Option 2: Add New Event Type

**Add to constants.hpp**:
```cpp
enum EventType {
  FinishConverting = 0,
  CoolDown = 1,
  AutoEmpty = 2,  // NEW
  EventTypeCount
};
```

**Handler**:
```cpp
class AutoEmptyHandler : public EventHandler {
  void handle_event(GridObjectId obj_id, EventArg /*arg*/) override {
    Converter* converter = static_cast<Converter*>(
      this->event_manager->grid->object(obj_id)
    );
    if (!converter) return;

    // Empty inventory and restart cycle
    converter->empty_and_restart();
  }
};
```

## Implementation Steps

### Phase 1: Core Functionality (2-3 days)

1. **Extend Converter Class**
   - Add `auto_empty_enabled` flag
   - Override `finish_converting()` to schedule emptying
   - Implement inventory clearing logic

2. **Event Handling**
   - Either reuse CoolDown with special logic
   - Or add new AutoEmpty event type

3. **Configuration Support**
   - Add `auto_empty_ticks` to ConverterConfig
   - Use negative cooldown as signal (Option 1)
   - Or add explicit flag (Option 2)

### Phase 2: Testing & Polish (1-2 days)

1. **Unit Tests**
   - Verify cycle timing
   - Test item removal
   - Ensure stats tracking

2. **Integration Tests**
   - Agents can take hearts during window
   - Hearts disappear after timeout
   - Cycle repeats correctly

3. **Observations**
   - Consider adding "TimeUntilEmpty" token
   - Or reuse existing ConvertingOrCoolingDown

## Code Changes Required

### Minimal Changes (Option 1)

1. **converter.hpp** (~50 lines)
   - Add `bool is_auto_empty_cycle`
   - Override `finish_converting()` and `finish_cooldown()`
   - Add emptying logic

2. **mettagrid_c.cpp** (~10 lines)
   - Check for negative cooldown in config
   - Set auto-empty flag during construction

3. **Python Config** (~5 lines)
   - Allow negative cooldown values
   - Document special behavior

### Total: ~65 lines of code changes

## Performance Impact

- **Memory**: 1 additional bool per converter (negligible)
- **CPU**: One additional branch in event handlers
- **Events**: Same number of events, just different handling
- **Observations**: No new tokens needed

**Expected overhead**: < 0.1% (essentially zero)

## Configuration Examples

### Basic Cyclical Heart Generator
```yaml
heart_timer:
  type_id: 20
  input_resources: {}
  output_resources: {heart: 1}
  conversion_ticks: 20   # 20 ticks to create heart
  cooldown: -30          # Heart exists for 30 ticks then vanishes
  max_output: 1
  initial_resource_count: 0
```

### Synchronized Pulse Generator
```yaml
pulse_generator:
  type_id: 21
  input_resources: {battery: 1}  # Requires battery to start
  output_resources: {timing_crystal: 1}
  conversion_ticks: 50   # 50 ticks to charge
  cooldown: -100         # Crystal lasts 100 ticks
  max_output: 1
```

## Advantages of This Approach

1. **Minimal Code Changes**: Works within existing architecture
2. **Backward Compatible**: Old converters unaffected
3. **No New Systems**: Reuses event infrastructure
4. **Deterministic**: Simple timing, no randomness
5. **Observable**: Agents can see ConvertingOrCoolingDown state

## Limitations

1. **Fixed Timing**: Can't dynamically change cycle times
2. **Single Item Type**: Empties all output types
3. **No Cascades**: Can't trigger other converters
4. **Binary State**: Either all items exist or none

## Future Extensions

If successful, could extend to:
- Multiple cycle states (produce A → empty → produce B)
- Conditional emptying (only if inventory full)
- Partial emptying (remove some, not all)
- Event cascades (emptying triggers nearby converters)

## Conclusion

The ultra-minimal implementation provides the requested cyclical behavior with auto-emptying while requiring only **10 lines of code** added to a single function. By simply checking the converter's type name for "cyclical" and emptying inventory during cooldown completion, we achieve the desired behavior with:

- **Zero new classes or event types**
- **Zero configuration changes**
- **Zero memory overhead**
- **Negligible performance impact** (one string search)
- **Complete backward compatibility**

This demonstrates that clever reuse of existing mechanics can achieve complex behaviors without architectural changes. The entire feature can be implemented in under an hour and tested within a day.
