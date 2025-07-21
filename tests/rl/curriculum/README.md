# Curriculum Tests Organization

This directory contains tests for the curriculum learning system in Metta. The tests have been reorganized to eliminate duplicates and improve clarity.

## Test Files

### Core Tests

- **`conftest.py`** - Shared fixtures and mock curriculum classes used across tests:
  - `MockCurriculum` - Simple mock for basic functionality testing
  - `StatefulCurriculum` - Mock with comprehensive state tracking and stats
  - `free_port` fixture - Provides available ports for server testing

- **`test_curriculum_core.py`** - Tests for the core curriculum interface

- **`test_curriculum_algorithms.py`** - Tests for specific curriculum algorithm implementations:
  - SingleTaskCurriculum
  - RandomCurriculum
  - PrioritizeRegressedCurriculum
  - SamplingCurriculum
  - ProgressiveCurriculum
  - BucketedCurriculum
  - MultiTaskCurriculum
  - SampledTaskCurriculum

### Integration Tests

- **`test_curriculum_trainer_integration.py`** - Consolidated trainer integration tests:
  - Trainer expected methods validation
  - Stats collection and processing
  - Training simulation with learning progress
  - Server/client communication
  - Batch exhaustion testing
  - Concurrent client access

### Validation Tests

- **`test_validate_all_curriculums.py`** - Validates all curriculum configurations can be instantiated

## Test Consolidation Summary

The following duplicate tests were removed during consolidation:

1. **Removed Files:**
   - `test_curriculum_server.py` - Tests moved to trainer integration
   - `test_curriculum_server_client.py` - Tests moved to trainer integration
   - `test_trainer_curriculum_stats.py` - Tests moved to trainer integration
   - `test_curriculum_integration.py` - Tests consolidated
   - `test_curriculum_server_integration.py` - Tests consolidated
   - `test_trainer_curriculum_integration.py` - Tests consolidated

2. **Consolidated Duplicate Tests:**
   - Basic server-client communication (appeared in 3 files)
   - Concurrent client access (appeared in 4 files)
   - Stats collection (appeared in 3 files)
   - Error handling (appeared in 2 files)
   - Server lifecycle tests (appeared in 2 files)

3. **Key Improvements:**
   - Created shared mock classes in conftest.py
   - Grouped related tests into logical test classes
   - Removed redundant test implementations
   - Improved test documentation and naming

## Running Tests

Run all curriculum tests:
```bash
pytest tests/rl/curriculum/ -v
```

Run specific test categories:
```bash
# Algorithm tests
pytest tests/rl/curriculum/test_curriculum_algorithms.py -v

# Trainer integration tests
pytest tests/rl/curriculum/test_curriculum_trainer_integration.py -v

# Core curriculum tests
pytest tests/rl/curriculum/test_curriculum_core.py -v
```

## Test Coverage

The consolidated tests cover:
- Core curriculum interface and task management
- All curriculum algorithm implementations
- Server/client communication patterns
- Stats collection and reporting
- Trainer integration points
- Error handling and edge cases
- Configuration validation