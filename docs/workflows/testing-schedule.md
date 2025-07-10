# Testing Schedule Implementation

## Quick Reference

This document provides a quick reference for the testing schedule implementation added to the Metta project.

### New Files Added

1. **`.github/workflows/scheduled-tests.yml`**
   - Implements hourly and daily scheduled test runs
   - Includes automatic failure notifications via GitHub issues

2. **`docs/testing_schedule.md`**
   - Comprehensive documentation of the testing cadence
   - Includes escalation procedures and best practices

3. **`tests/test_marker_examples.py`**
   - Example test file demonstrating marker usage
   - Shows how to categorize tests properly

### Configuration Changes

**`pyproject.toml`** - Added new pytest markers:
```toml
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "hourly: marks tests for hourly execution (critical tests that should run frequently)",
    "daily: marks tests for daily execution (comprehensive tests including integration)",
    "integration: marks tests that require external services or are more expensive to run",
]
```

### Test Schedule Summary

| Schedule | Frequency | Tests Run | Purpose |
|----------|-----------|-----------|----------|
| CI | On PR/Push | All unmarked tests | Immediate feedback |
| Hourly | :15 past each hour | `@pytest.mark.hourly` | Critical path validation |
| Daily | 2:30 AM UTC | `@pytest.mark.daily` + integration | Comprehensive validation |
| Weekly | Sunday 3 AM UTC | Distributed training | Multi-GPU validation |

### Usage Examples

```python
# Mark a critical test to run hourly
@pytest.mark.hourly
def test_critical_feature():
    pass

# Mark a comprehensive test to run daily
@pytest.mark.daily
def test_end_to_end_scenario():
    pass

# Mark an integration test
@pytest.mark.integration
def test_external_service():
    pass
```

### Manual Execution

Trigger scheduled tests manually:
1. Go to Actions → "Scheduled Tests"
2. Run workflow with desired category (hourly/daily/all)

### Monitoring

- Failed scheduled tests create GitHub issues automatically
- Issues are labeled: `test-failure`, `automated`, `high-priority`
- Existing issues get comments for subsequent failures
