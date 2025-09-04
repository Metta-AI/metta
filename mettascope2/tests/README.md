# MettaScope 2 Tests

This directory contains test files for the MettaScope 2 game visualization.

## Test Files

- `test_nim_config.nim` - Basic configuration and environment setup test
- `test_terrain_clusters.nim` - Tests terrain generation (rivers, wheat, trees)
- `test_water_spawn.nim` - Tests that structures avoid spawning on water
- `test_village_spawn.nim` - Tests village/house spawning logic
- `test_village_details.nim` - Detailed village structure tests
- `test_corner_placement.nim` - Tests agent placement at house corners
- `test_temples.nim` - Tests temple spawning and Clippy generation
- `test_clearings_visual.nim` - Visual test for terrain clearing around structures

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