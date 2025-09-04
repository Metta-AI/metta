# MettaScope 2 Tests

This directory contains test files for the MettaScope 2 game visualization.

## Test Files

### Consolidated Test Suites
These are the main test files that group related tests together using the unittest framework:

- `test_villages_consolidated.nim` - Village structures, terrain generation, and placement
- `test_combat_consolidated.nim` - Combat mechanics and agent interactions (legacy)
- `test_resources_consolidated.nim` - Resource gathering and conversion system
- `test_clippies_consolidated.nim` - Clippy AI behavior and movement patterns
- `test_mettascope_integration.nim` - Integration tests for the full environment
- `test_full_simulation.nim` - Full simulation step tests

### Individual Test Files
Additional focused test files:

- `test_nim_config.nim` - Basic configuration and environment setup test
- `test_utils.nim` - Utility functions for testing
- `test_clippy_simple.nim` - Simple Clippy behavior tests
- `test_clippy_wandering.nim` - Clippy wandering pattern visualization

### Legacy Test Files (from original implementation)
These files test older features that may be deprecated:
- Various combat and resource gathering tests from the original implementation

## Running Tests

### Run all tests:
```bash
./run_all_tests.sh
```

### Run individual test:
```bash
nim c -r test_nim_config.nim
```

## Writing New Tests

When creating new tests:
1. Name them `test_*.nim`
2. Import modules using relative paths: `import ../src/mettascope/module`
3. Add a description comment at the top of the file
4. Keep tests focused on specific functionality