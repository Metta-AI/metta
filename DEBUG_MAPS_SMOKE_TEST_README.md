# Debug Maps Smoke Test Infrastructure

## Overview

This document describes the smoke testing infrastructure created for the new debug maps in `configs/env/mettagrid/maps/debug/`. The smoke tests are designed to check basic agent performance on these maps to ensure they load correctly and agents can navigate and complete basic tasks.

The infrastructure leverages existing general-purpose tools (`tools/sim.py`) and pytest-based testing rather than creating map-specific commands.

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

### 3. Comprehensive Test Suite (Parallel Structure)

Created comprehensive test coverage following the parallel directory structure:

- `tests/env/mettagrid/debug/test_debug_maps.py` - Tests for debug configurations and maps

#### Test Coverage:

- **Map Structure Validation**: File existence, line consistency, valid symbols
- **Environment Configuration Testing**: Hydra config loading, proper inheritance
- **Content Validation**: Uniqueness verification, required game elements
- **Smoke Tests**: Map loading into ASCII builders, environment instantiation
- **Integration Testing**: Compatibility with existing infrastructure

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

configs/sim/
└── debug_maps_smoke_test.yaml
```

## Usage

### Running Smoke Tests with the General Simulation Tool

Use the existing `tools/sim.py` with the debug maps smoke test configuration:

```bash
# Basic smoke test with existing simulation infrastructure
python -m tools.sim sim=debug_maps_smoke_test policy_uri=<POLICY_URI> run=<RUN_NAME>

# Examples with different policies
python -m tools.sim sim=debug_maps_smoke_test policy_uri=training_regular_envset run=debug_test
python -m tools.sim sim=debug_maps_smoke_test policy_uri="wandb://run/your-policy" run=debug_smoke_test
```

### Running Unit Tests and Pytest-based Smoke Tests

```bash
# Run all debug map tests (includes smoke tests)
python -m pytest tests/env/mettagrid/debug/test_debug_maps.py -v

# Run only structure validation tests
python -m pytest tests/env/mettagrid/debug/test_debug_maps.py::TestDebugMaps -v

# Run only smoke tests (no trained policy required)
python -m pytest tests/env/mettagrid/debug/test_debug_maps.py::TestDebugMapsSmoke -v

# Run specific test
python -m pytest tests/env/mettagrid/debug/test_debug_maps.py::TestDebugMapsSmoke::test_debug_map_loading_smoke_test -v

# Test all ASCII maps (includes debug maps)
python -m pytest tests/map/test_validate_all_ascii_maps.py -v
```

### Testing Individual Debug Environments

```bash
# Test loading a specific debug environment
python -c "
import hydra
with hydra.initialize(config_path='configs'):
    cfg = hydra.compose('env/mettagrid/debug/evals/debug_tiny_two_altars')
    print('✅ Successfully loaded debug environment')
    print(f'Map: {cfg.game.map_builder.uri}')
    print(f'Max steps: {cfg.game.max_steps}')
"
```

## Success Criteria

The smoke tests validate:

1. **Map Loading**: Maps load without instantiation errors
2. **Configuration Validity**: Hydra can load all debug environment configs
3. **Basic Compatibility**: Maps work with ASCII room builders
4. **Structure Integrity**: Consistent formatting and valid symbols

For performance testing with trained policies, the simulation tool validates:
- **Basic Navigation**: Agents can move and interact with environments
- **Performance Thresholds**: Configurable success criteria
- **Error-Free Execution**: No runtime exceptions during simulation

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

- **General-Purpose Tools**: Uses existing `tools/sim.py` for performance testing
- **Hydra Configuration System**: All configs follow standard patterns with proper debug namespace
- **Existing Test Suite**: Leverages ASCII map validation infrastructure
- **Parallel Test Structure**: Follows the repository's parallel testing conventions
- **W&B Integration**: Compatible with existing policy storage and metrics
- **CI/CD Pipeline**: Unit tests can be integrated into automated testing

## Troubleshooting

### Common Issues:

1. **Policy Not Found**: Ensure the policy URI points to a valid trained model
2. **Environment Loading Errors**: Check that all map files exist and are properly formatted
3. **Configuration Path Errors**: Ensure using `env/mettagrid/debug/evals/` paths
4. **Hydra Initialization Errors**: Make sure config paths are correct

### Validation Steps:

1. **ASCII Map Validation**: `python -m pytest tests/map/test_validate_all_ascii_maps.py`
2. **Debug Configuration Testing**: `python -m pytest tests/env/mettagrid/debug/test_debug_maps.py -v`
3. **Smoke Tests (No Policy Required)**: `python -m pytest tests/env/mettagrid/debug/test_debug_maps.py::TestDebugMapsSmoke -v`
4. **Performance Testing**: `python -m tools.sim sim=debug_maps_smoke_test policy_uri=<POLICY> run=test`

### Quick Validation Without Trained Policy

```bash
# These tests don't require a trained policy
python -m pytest tests/env/mettagrid/debug/test_debug_maps.py::TestDebugMapsSmoke::test_debug_map_loading_smoke_test -v
python -m pytest tests/env/mettagrid/debug/test_debug_maps.py::TestDebugMaps::test_debug_environment_configs_loadable -v
```

## Benefits of This Approach

1. **Leverages Existing Infrastructure**: No new tools commands needed
2. **Flexible Testing**: Both unit tests and performance testing available
3. **Policy-Independent Validation**: Core functionality can be tested without trained models
4. **Standard Integration**: Follows repository conventions for tools and testing
5. **Comprehensive Coverage**: From basic map validation to full agent performance testing

## Future Enhancements

Potential improvements to the smoke test infrastructure:

1. **Automated Performance Baselines**: Track performance over time
2. **Visual Map Validation**: Render maps to verify layout correctness
3. **Multi-Agent Testing**: Test maps with multiple agents
4. **Custom Metrics**: Task-specific performance measurements
5. **Regression Detection**: Alert on significant performance degradation
6. **CI Integration**: Automated smoke tests on map changes
7. **Performance Benchmarking**: Compare debug maps against standard evaluation maps
