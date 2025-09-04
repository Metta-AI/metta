# MettaScope 2 Tests

This directory contains test files for the MettaScope 2 game visualization.

## Test Files

### Environment & Configuration
- `test_nim_config.nim` - Basic configuration and environment setup test
- `simple_test.nim` - Simple environment test
- `test_simple_step.nim` - Basic step execution test

### Terrain & Map Generation
- `test_terrain_clusters.nim` - Tests terrain generation (rivers, wheat, trees)
- `test_water_spawn.nim` - Tests that structures avoid spawning on water
- `test_clearing.nim` - Tests terrain clearing functionality
- `test_clearings_visual.nim` - Visual test for terrain clearing around structures

### Structures & Spawning
- `test_village_spawn.nim` - Tests village/house spawning logic
- `test_village_details.nim` - Detailed village structure tests
- `test_corner_placement.nim` - Tests agent placement at house corners
- `test_temples.nim` - Tests temple spawning and Clippy generation

### Gameplay & Combat
- `test_combat.nim` - Tests combat mechanics between agents
- `test_combat_simple.nim` - Simple combat scenario tests
- `test_combat_forced.nim` - Forced combat interaction tests
- `test_respawn.nim` - Tests agent respawn mechanics at altars
- `test_resource_gathering.nim` - Tests resource gathering mechanics

### Debugging & Visualization
- `test_visual.nim` - Visual debugging and display tests
- `debug_map.nim` - Map debugging utilities
- `debug_positions.nim` - Position debugging utilities

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