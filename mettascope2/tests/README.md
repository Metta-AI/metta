# MettaScope 2 Tests

This directory contains test files for the MettaScope 2 game visualization.

## Test Files

### Main Test Suites
The test suite has been consolidated into three focused test files:

- `test_core_systems.nim` - Core gameplay mechanics
  - Directional movement (N/S/E/W)
  - Resource collection from mines
  - Ore to battery conversion at converters  
  - Altar deposits (batteries → hearts)
  - Map generation consistency

- `test_ai_behavior.nim` - AI and agent behaviors
  - Agent controller decision making
  - Clippy wandering patterns (concentric circles)
  - Village agent grouping
  - Full simulation with resource flow

- `test_diagonal_movement_fix.nim` - Specific edge case test
  - Ensures agents move to cardinal positions before using objects
  - Tests the diagonal adjacency fix for mines/converters

## Running Tests

### Run all tests:
```bash
./run_tests.sh
```

### Run individual test:
```bash
nim c -r tests/test_core_systems.nim
nim c -r tests/test_ai_behavior.nim
nim c -r tests/test_diagonal_movement_fix.nim
```

## Test Output

Tests provide detailed output showing:
- Test name and description
- Step-by-step progress
- Success/failure indicators (✓/✗)
- Summary statistics

## Writing New Tests

When creating new tests:
1. Name them `test_*.nim`
2. Import modules using relative paths: `import ../src/mettascope/module`
3. Add a description comment at the top of the file
4. Keep tests focused on specific functionality
5. Consider adding to existing consolidated tests when appropriate