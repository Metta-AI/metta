# CI/CD Testing Schedule Analysis

## Executive Summary

The Metta project has a comprehensive, well-implemented testing infrastructure with scheduled test runs already in place. The system uses a tiered approach with hourly critical tests, daily comprehensive tests, and weekly distributed training validation.

## Current Implementation Status

### 1. Scheduled Test Execution

**File**: `.github/workflows/scheduled-tests.yml`

The workflow implements:
- **Hourly Tests**: Run at :15 past each hour (e.g., 1:15, 2:15)
  - Focus: Critical path validation
  - Timeout: 30 minutes
  - Marker: `@pytest.mark.hourly`
  - Current count: 12 tests

- **Daily Tests**: Run at 2:30 AM UTC (6:30 PM PST / 7:30 PM PDT)
  - Focus: Comprehensive validation including integration tests
  - Timeout: 90 minutes
  - Markers: `@pytest.mark.daily` or tests without `hourly` marker
  - Current count: 14 tests
  - Includes coverage reporting to Codecov

### 2. Test Categorization

**Configuration**: `pyproject.toml`

```toml
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "hourly: marks tests for hourly execution (critical tests that should run frequently)",
    "daily: marks tests for daily execution (comprehensive tests including integration)",
    "integration: marks tests that require external services or are more expensive to run",
]
```

**Current Distribution**:
| Marker | Count | Purpose |
|--------|-------|---------|
| hourly | 12 | Critical tests run every hour |
| daily | 14 | Comprehensive tests run once per day |
| integration | 8 | Tests requiring external services |
| slow | 4 | Long-running tests |

### 3. Failure Monitoring & Alerts

The scheduled tests workflow includes automatic failure detection:

1. **GitHub Issue Creation**: Automatically creates issues for test failures
2. **Labels**: `test-failure`, `automated`, `high-priority`
3. **Deduplication**: Updates existing issues instead of creating duplicates
4. **Escalation Path**: Documented in `docs/testing_schedule.md`

### 4. Other Scheduled Workflows

| Workflow | Schedule | Purpose |
|----------|----------|---------|
| `dist-smoketest.yml` | Weekly (Sundays 3 AM UTC) | Distributed training validation |
| `build-image.yml` | Daily (7 AM UTC) | Docker image builds |
| `stale.yml` | Daily (midnight) | Cleanup stale issues/PRs |

### 5. Manual Triggers

All scheduled workflows support `workflow_dispatch` for on-demand execution with configurable parameters.

## Test Coverage by Category

### Hourly Tests (Critical Path)
- Core agent initialization
- Policy metadata and state operations
- Basic training loop functionality
- Configuration parsing and validation
- MetaGrid environment initialization checks

### Daily Tests (Comprehensive)
- End-to-end training scenarios
- WandB integration tests
- S3 checkpoint storage
- GPU training tests
- Comprehensive configuration validation
- Long-running evaluations

### Integration Tests
- External service dependencies (AWS S3, WandB)
- Cloud storage operations
- API integrations
- Database connections

## Strengths of Current Implementation

1. **Well-Structured Tiering**: Clear separation between critical (hourly) and comprehensive (daily) tests
2. **Automatic Failure Detection**: GitHub issues created automatically for tracking
3. **Flexible Scheduling**: Cron-based with manual override options
4. **Good Documentation**: Comprehensive guides in `docs/testing_schedule.md`
5. **Test Analysis Tools**: `tools/test_schedule_summary.py` for monitoring test distribution
6. **Time Zone Optimization**: Scheduled during off-peak hours (US West Coast)

## Potential Improvements

### 1. Increase Test Coverage Markers
Currently only 12 hourly and 14 daily tests are marked. Consider:
- Review critical paths and mark more tests as `@pytest.mark.hourly`
- Add markers to integration tests that should run daily
- Create a script to identify unmarked tests that should be categorized

### 2. Performance Metrics Tracking
Add test execution time tracking:
```yaml
- name: Upload test metrics
  uses: actions/upload-artifact@v4
  with:
    name: test-execution-metrics
    path: |
      pytest-times.json
      test-durations.csv
```

### 3. Parallel Test Execution
The daily tests could benefit from matrix strategy for faster execution:
```yaml
strategy:
  matrix:
    test-group: [unit, integration, slow]
```

### 4. Test Flakiness Detection
Implement retry logic for known flaky tests:
```yaml
- name: Run tests with retry
  uses: nick-invision/retry@v2
  with:
    timeout_minutes: 30
    max_attempts: 3
    command: pytest -m "hourly" -v
```

### 5. Enhanced Monitoring Dashboard
Consider integrating with tools like:
- Grafana for test execution visualization
- Datadog for advanced alerting
- Custom dashboard showing test health over time

## Recommended Next Steps

1. **Audit Test Markers**: Run a campaign to ensure all critical tests have appropriate markers
2. **Implement Test Metrics**: Add execution time and memory usage tracking
3. **Create Test Health Dashboard**: Visualize test pass rates and execution times
4. **Document Test Ownership**: Assign owners to test categories for faster resolution
5. **Quarterly Review Process**: Schedule regular reviews of test categorization

## Test Marker Analysis Results

Running the `tools/analyze_test_markers.py` script reveals significant opportunities for improvement:

- **Total tests analyzed**: 643
- **Already marked**: 29 (4.5%)
  - Hourly: 12
  - Daily: 14
  - Integration: 8
  - Slow: 4
- **Recommended for hourly**: 77 additional tests
- **Recommended for daily**: 34 additional tests
- **Unclassified**: 503 tests

This shows that only 4.5% of tests have been categorized for scheduled execution, leaving significant room for improvement.

## Conclusion

The Metta project has a mature, well-implemented CI/CD testing infrastructure. The scheduled test execution is already operational with:

✅ **Infrastructure**: Hourly and daily scheduled workflows fully implemented
✅ **Monitoring**: Automatic failure detection and GitHub issue creation
✅ **Documentation**: Comprehensive guides and escalation procedures
✅ **Tooling**: Analysis scripts for test categorization

⚠️ **Main Gap**: Only 29 out of 643 tests (4.5%) are marked for scheduled execution

The infrastructure is ready and waiting - the primary action needed is to run a test categorization campaign to mark the ~111 tests identified as candidates for hourly or daily execution. This would increase scheduled test coverage from 4.5% to over 20%.
