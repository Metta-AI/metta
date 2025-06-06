# Debug Maps Smoke Test Infrastructure

## Overview

This document describes the smoke testing infrastructure created for the new debug maps in `configs/env/mettagrid/maps/debug/`. The smoke tests are designed to check basic agent performance on these maps to ensure they load correctly and agents can navigate and complete basic tasks.

## Created Files

### 1. Debug Map Environment Configurations

Created environment configuration files for each debug map in the debug evaluation structure:

- `configs/env/mettagrid/debug/evals/defaults.yaml` - Default configuration for debug evaluations
- `configs/env/mettagrid/debug/evals/debug_mixed_objects.yaml`
- `configs/env/mettagrid/debug/evals/debug_resource_collection.yaml`
- `configs/env/mettagrid/debug/evals/debug_simple_obstacles.yaml`
- `configs/env/mettagrid/debug/evals/debug_tiny_two_altars.yaml`

These configurations follow the standard pattern used by other evaluation environments, specifying:
- Map builder using `mettagrid.room.ascii.Ascii`
- URI pointing to the corresponding `.map` file
- Appropriate `max_steps` values based on map complexity
- Proper inheritance from debug-specific defaults

### 2. Smoke Test Simulation Suite

Created `configs/sim/debug_maps_smoke_test.yaml` which defines a simulation suite that tests all debug maps:

```yaml
name: debug_maps_smoke_test
max_time_s: 30
num_episodes: 3
simulations:
  debug_mixed_objects:
    env: env/mettagrid/debug/evals/debug_mixed_objects
  debug_resource_collection:
    env: env/mettagrid/debug/evals/debug_resource_collection
  debug_simple_obstacles:
    env: env/mettagrid/debug/evals/debug_simple_obstacles
  debug_tiny_two_altars:
    env: env/mettagrid/debug/evals/debug_tiny_two_altars
```

### 3. Comprehensive Smoke Test Script

Created `tools/debug_maps_smoke_test.py` - a standalone script that:

- Tests each debug map individually
- Measures performance metrics (reward, steps, completion rate, execution time)
- Validates that maps load without errors
- Provides detailed reporting of test results
- Supports both full and quick test modes

#### Key Features:

- **Performance Validation**: Checks average reward, step count, completion rate
- **Error Detection**: Catches and reports runtime errors
- **Configurable Criteria**: Adjustable success thresholds
- **Detailed Reporting**: Comprehensive test result summary
- **CLI Interface**: Command-line arguments for flexibility

### 4. Unit Tests (Parallel Structure)

Created comprehensive test coverage following the parallel directory structure:

- `tests/env/mettagrid/debug/test_debug_maps.py` - Tests for debug configurations and maps
- `tests/tools/test_debug_maps_smoke_test.py` - Tests for the smoke test script itself

#### Test Coverage:

- Map file existence and structure validation
- Environment configuration validation
- Content uniqueness verification
- Required game element presence checks
- Smoke test script functionality validation
- Integration with existing ASCII map validation infrastructure

## Directory Structure

```
configs/env/mettagrid/debug/
├── evals/
│   ├── defaults.yaml
│   ├── debug_mixed_objects.yaml
│   ├── debug_resource_collection.yaml
│   ├── debug_simple_obstacles.yaml
│   └── debug_tiny_two_altars.yaml

tests/env/mettagrid/debug/
└── test_debug_maps.py

tests/tools/
└── test_debug_maps_smoke_test.py

tools/
└── debug_maps_smoke_test.py

configs/sim/
└── debug_maps_smoke_test.yaml
```

## Usage

### Running with the Simulation Tool

Use the standard simulation tool with the debug maps smoke test configuration:

```bash
python -m tools.sim sim=debug_maps_smoke_test policy_uri=<POLICY_URI> run=<RUN_NAME>
```

Where `<POLICY_URI>` is a valid trained policy (e.g., from W&B or local path).

### Running the Standalone Smoke Test Script

```bash
# Full test with default policy
python tools/debug_maps_smoke_test.py

# Quick test (1 episode per map)
python tools/debug_maps_smoke_test.py --quick

# Test with specific policy
python tools/debug_maps_smoke_test.py --policy-uri "wandb://run/your-policy-uri"
```

### Running Unit Tests

```bash
# Test debug map structure and configuration (parallel structure)
python -m pytest tests/env/mettagrid/debug/test_debug_maps.py -v

# Test smoke test script functionality
python -m pytest tests/tools/test_debug_maps_smoke_test.py -v

# Test all ASCII maps (includes debug maps)
python -m pytest tests/map/test_validate_all_ascii_maps.py -v
```

## Success Criteria

The smoke tests validate:

1. **Map Loading**: Maps load without instantiation errors
2. **Basic Navigation**: Agents can move and interact with environments
3. **Performance Thresholds**:
   - Minimum average reward: 0.1
   - Maximum average steps: 200 (efficiency check)
   - Minimum completion rate: 0% (at least some episodes should complete tasks)
   - Maximum execution time: 60 seconds per map

4. **Error-Free Execution**: No runtime exceptions during simulation

## Debug Map Characteristics

The debug maps test different aspects of agent performance:

- **`tiny_two_altars.map`**: Minimal 6x4 map for basic functionality testing
- **`simple_obstacles.map`**: 12x11 map with basic wall obstacles and navigation challenges
- **`mixed_objects.map`**: 16x13 map with diverse object types (logs, moss, stone, altars)
- **`resource_collection.map`**: 14x12 map focused on resource gathering scenarios

Each map contains:
- Agents (`a`) and altars (`A`)
- Walls (`W`) for boundaries and obstacles
- Various objects like logs (`L`), moss (`m`), stone (`s`)
- Strategic layout for different navigation and interaction patterns

## Integration with Existing Infrastructure

The smoke test infrastructure integrates seamlessly with:

- **Hydra Configuration System**: All configs follow standard patterns with proper debug namespace
- **Existing Test Suite**: Leverages ASCII map validation infrastructure
- **Parallel Test Structure**: Follows the repository's parallel testing conventions
- **W&B Integration**: Compatible with existing policy storage and metrics
- **CI/CD Pipeline**: Unit tests can be integrated into automated testing

## Troubleshooting

### Common Issues:

1. **Policy Not Found**: Ensure the policy URI points to a valid trained model
2. **Environment Loading Errors**: Check that all map files exist and are properly formatted
3. **Low Performance**: May indicate issues with map design or agent training
4. **Configuration Path Errors**: Ensure using `env/mettagrid/debug/evals/` paths

### Validation Steps:

1. Run ASCII map validation: `python -m pytest tests/map/test_validate_all_ascii_maps.py`
2. Check debug configurations: `python -m pytest tests/env/mettagrid/debug/test_debug_maps.py`
3. Validate smoke test script: `python -m pytest tests/tools/test_debug_maps_smoke_test.py`
4. Test with known working policy from existing evaluations

## Future Enhancements

Potential improvements to the smoke test infrastructure:

1. **Automated Performance Baselines**: Track performance over time
2. **Visual Map Validation**: Render maps to verify layout correctness
3. **Multi-Agent Testing**: Test maps with multiple agents
4. **Custom Metrics**: Task-specific performance measurements
5. **Regression Detection**: Alert on significant performance degradation
6. **Integration with CI/CD**: Automated smoke tests on map changes
