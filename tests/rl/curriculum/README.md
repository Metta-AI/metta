# Curriculum Tests Organization

This directory contains tests for the curriculum learning system in Metta. The tests have been reorganized to eliminate duplicates and improve clarity.

## Test Files

### Core Tests

- **`conftest.py`** - Shared fixtures, mock classes, and test utilities:
  - `MockCurriculum` - Simple mock for basic functionality testing
  - `StatefulCurriculum` - Mock with comprehensive state tracking and stats
  - Score generators (MonotonicLinearScores, ZeroScores, RandomScores, etc.)
  - Test utilities (run_curriculum_simulation, create_mock_curricula, etc.)
  - `free_port` fixture - Provides available ports for server testing

- **`test_curriculum_core.py`** - Tests for the core curriculum interface:
  - Basic curriculum operations (get_task, complete_task)
  - Stats methods functionality
  - Task configuration variation
  - Learning adaptation based on performance
  - Task interface validation

- **`test_curriculum_algorithms.py`** - Tests for specific curriculum algorithm implementations:
  - SingleTaskCurriculum
  - RandomCurriculum
  - PrioritizeRegressedCurriculum
  - SamplingCurriculum
  - ProgressiveCurriculum
  - BucketedCurriculum
  - MultiTaskCurriculum
  - SampledTaskCurriculum

### Scenario Tests

- **`test_curriculum_progressive_scenarios.py`** - Progressive curriculum behavior scenarios:
  - Monotonic linear signal progression
  - Zero signal behavior (should stay on first task)
  - Random signal progression patterns

- **`test_curriculum_learning_progress_scenarios.py`** - Learning progress curriculum scenarios:
  - Conditional linear scores testing
  - Threshold-dependent task progression

- **`test_curriculum_prioritize_regressed_scenarios.py`** - Prioritize regressed curriculum scenarios:
  - Task weight adjustment based on performance regression
  - Independent linear progression for multiple tasks

### Server/Client Tests

- **`test_server_client.py`** - Consolidated server/client tests:
  - Basic communication tests
  - Batch prefetching
  - Error handling
  - Client returns empty stats
  - Concurrent client access (sequential and threaded)
  - Server lifecycle management (restart, shutdown)
  - Complex curriculum integration (BucketedCurriculum, RandomCurriculum)
  - Empty batch handling scenarios

### Integration Tests

- **`test_trainer_integration.py`** - Comprehensive trainer integration tests:
  - Trainer expected methods validation
  - Stats collection and processing with curriculum
  - Training simulation with learning progress
  - Server/client workflow from trainer perspective
  - Batch exhaustion and prefetching behavior

### Validation Tests

- **`test_validate_all_curriculums.py`** - Validates all curriculum configurations can be instantiated

## Test Organization Summary

The reorganization successfully:
1. Consolidated duplicate tests across multiple files
2. Created a comprehensive conftest.py with all shared utilities
3. Organized tests by functionality (core, algorithms, scenarios, integration)
4. Improved test naming and documentation
5. Maintained clear separation between unit and integration tests

## Running Tests

Run all curriculum tests:
```bash
pytest tests/rl/curriculum/ -v
```

Run specific test categories:
```bash
# Algorithm tests
pytest tests/rl/curriculum/test_curriculum_algorithms.py -v

# Server/Client tests
pytest tests/rl/curriculum/test_server_client.py -v

# Trainer integration tests
pytest tests/rl/curriculum/test_trainer_integration.py -v

# Core curriculum tests
pytest tests/rl/curriculum/test_curriculum_core.py -v

# Scenario tests
pytest tests/rl/curriculum/test_curriculum_*_scenarios.py -v
```