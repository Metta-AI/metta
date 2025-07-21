# Curriculum Tests Organization

This directory contains tests for the curriculum learning system in Metta. The tests have been reorganized to eliminate duplicates and improve clarity.

## Final Organization

### Test Files (66 tests total)

1. **`conftest.py`** - Shared fixtures, mock classes, and test utilities:
   - `MockCurriculum` - Simple mock for basic functionality testing
   - `StatefulCurriculum` - Mock with comprehensive state tracking and stats
   - Score generators for testing curriculum behavior patterns
   - Test utilities (run_curriculum_simulation, create_mock_curricula, etc.)
   - `free_port` fixture - Provides available ports for server testing

2. **`test_curriculum_core.py`** (5 tests) - Core curriculum interface tests:
   - Basic curriculum operations (get_task, complete_task)
   - Stats methods functionality
   - Task configuration variation
   - Learning adaptation based on performance
   - Task interface validation

3. **`test_curriculum_algorithms.py`** (9 tests) - Algorithm implementation tests:
   - SingleTaskCurriculum
   - RandomCurriculum
   - PrioritizeRegressedCurriculum
   - SamplingCurriculum
   - ProgressiveCurriculum
   - BucketedCurriculum
   - MultiTaskCurriculum
   - SampledTaskCurriculum
   - Helper function tests

4. **`test_server_client.py`** (11 tests) - Server/client communication tests:
   - Basic communication
   - Batch prefetching
   - Error handling
   - Client returns empty stats
   - Concurrent client access (sequential and threaded)
   - Server lifecycle management (restart, shutdown)
   - Complex curriculum integration (BucketedCurriculum, RandomCurriculum)
   - Empty batch handling

5. **`test_trainer_integration.py`** (5 tests) - Trainer integration tests:
   - Trainer expected methods validation
   - Stats collection and processing with curriculum
   - Training simulation with learning progress
   - Server/client workflow from trainer perspective
   - Batch exhaustion and prefetching behavior

6. **`test_validate_all_curriculums.py`** (36 tests) - Configuration validation:
   - Tests all curriculum YAML configurations can be instantiated
   - Covers all environments and task configurations

## Consolidation Summary

The reorganization successfully eliminated numerous duplicate tests:

### Removed Duplicates:
- `test_curriculum_stats_collection` (was in 3 files)
- `test_curriculum_stats_methods` (was in 2 files)
- Multiple client tests (was in 3 files)
- Server restart/shutdown tests (was in 2 files)
- Trainer integration tests (was scattered across 4 files)

### Key Improvements:
1. **Reduced file count** from ~15 test files to 6 focused test files
2. **Consolidated utilities** into a comprehensive conftest.py
3. **Clear separation** between unit tests and integration tests
4. **Improved naming** and organization by functionality
5. **Maintained coverage** while eliminating redundancy

## Running Tests

Run all curriculum tests:
```bash
pytest tests/rl/curriculum/ -v
```

Run specific test categories:
```bash
# Core functionality
pytest tests/rl/curriculum/test_curriculum_core.py -v

# Algorithm implementations
pytest tests/rl/curriculum/test_curriculum_algorithms.py -v

# Server/Client communication
pytest tests/rl/curriculum/test_server_client.py -v

# Trainer integration
pytest tests/rl/curriculum/test_trainer_integration.py -v

# Configuration validation
pytest tests/rl/curriculum/test_validate_all_curriculums.py -v
```