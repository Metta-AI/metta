# Metta CI/CD Testing Schedule

## Overview

Metta implements a tiered testing strategy to balance comprehensive coverage with efficient resource utilization:

- **Quick tests** run on every PR and push (standard CI)
- **Hourly tests** run critical path validations
- **Daily tests** run comprehensive test suites
- **Weekly tests** run distributed training smoke tests

## Test Execution Schedule

### Continuous Integration (Every PR/Push)
- **Trigger**: Pull requests, pushes to main, merge group events
- **Workflow**: `.github/workflows/checks.yml`
- **Duration**: ~10 minutes
- **Tests**: All unit tests without special markers
- **Purpose**: Catch regressions immediately

### Hourly Critical Tests
- **Schedule**: `:15 past every hour` (e.g., 1:15, 2:15, 3:15)
- **Workflow**: `.github/workflows/scheduled-tests.yml`
- **Duration**: ~30 minutes maximum
- **Marker**: `@pytest.mark.hourly`
- **Tests**:
  - Core agent initialization
  - Critical training loop functionality
  - Multi-agent scenarios
  - Essential API endpoints
- **Purpose**: Early detection of critical failures

### Daily Comprehensive Tests
- **Schedule**: `2:30 AM UTC` (6:30 PM PST / 7:30 PM PDT)
- **Workflow**: `.github/workflows/scheduled-tests.yml`
- **Duration**: ~90 minutes maximum
- **Markers**: `@pytest.mark.daily` or tests without `hourly` marker
- **Tests**:
  - Full test suite including slow tests
  - Integration tests with external services
  - End-to-end training scenarios
  - Memory-intensive operations
  - S3/cloud storage integrations
- **Purpose**: Comprehensive validation including expensive tests

### Weekly Distributed Training Tests
- **Schedule**: `Sundays at 3:00 AM UTC`
- **Workflow**: `.github/workflows/dist-smoketest.yml`
- **Duration**: ~3 hours
- **Tests**:
  - 1-GPU training runs
  - 4-GPU distributed training
  - Full training pipeline validation
- **Purpose**: Validate distributed training infrastructure

### Daily Docker Image Build
- **Schedule**: `7:00 AM UTC` (11 PM PST / 12 AM PDT)
- **Workflow**: `.github/workflows/build-image.yml`
- **Purpose**: Ensure Docker images build successfully

## Test Categorization with Pytest Markers

### Available Markers

```python
# Critical tests that must pass - run hourly
@pytest.mark.hourly

# Comprehensive tests - run daily
@pytest.mark.daily

# Tests requiring external services (AWS, WandB, etc.)
@pytest.mark.integration

# Long-running tests
@pytest.mark.slow
```

### Marker Usage Guidelines

1. **@pytest.mark.hourly**
   - Use for critical path tests
   - Must complete in under 1 minute
   - No external dependencies
   - Examples: Core initialization, basic training loop

2. **@pytest.mark.daily**
   - Use for comprehensive scenarios
   - Can take up to 5 minutes
   - May include integration tests
   - Examples: End-to-end workflows, performance tests

3. **@pytest.mark.integration**
   - Use for tests requiring external services
   - Often combined with `@pytest.mark.daily`
   - Examples: S3 operations, WandB logging, API calls

4. **@pytest.mark.slow**
   - Use for tests taking >30 seconds
   - Automatically included in daily runs
   - Examples: Large model training, memory stress tests

### Combining Markers

```python
@pytest.mark.integration
@pytest.mark.daily
def test_s3_checkpoint_upload():
    """Runs daily as part of integration test suite"""
    pass
```

## Expected Execution Times

| Test Category | Expected Duration | Timeout | Frequency |
|--------------|-------------------|---------|-----------|
| PR Tests | 5-10 min | 10 min | Every PR |
| Hourly Tests | 15-30 min | 30 min | Every hour |
| Daily Tests | 60-90 min | 90 min | Once daily |
| Weekly Distributed | 2-3 hours | 3 hours | Weekly |

## Monitoring and Alerts

### Automatic Failure Detection

The scheduled-tests workflow automatically:
1. Detects test failures
2. Creates GitHub issues for failures
3. Tags issues with `test-failure`, `automated`, `high-priority`
4. Updates existing issues if tests remain broken

### Manual Monitoring

- **GitHub Actions**: Check [Actions tab](../../../actions/workflows/scheduled-tests.yml)
- **Test Results**: Available as artifacts for 7 days (hourly) or 30 days (daily)
- **Coverage Reports**: Uploaded to Codecov for daily runs

## Escalation Procedures

### Immediate Response Required
1. **Hourly test failures**: Fix within 4 hours
   - Affects: Core functionality
   - Action: Revert breaking changes if fix not immediate

2. **Multiple consecutive failures**: Escalate immediately
   - Pattern: Same test failing 3+ times
   - Action: Page on-call engineer

### Standard Response
1. **Daily test failures**: Fix within 24 hours
   - Affects: Integration and comprehensive tests
   - Action: Create fix PR, assign to test owner

2. **Weekly distributed test failures**: Fix within 48 hours
   - Affects: Multi-GPU training
   - Action: Investigate infrastructure issues

### Escalation Chain
1. **Hour 0-4**: Test author or last committer
2. **Hour 4-8**: Team lead notification
3. **Hour 8-24**: On-call engineer engagement
4. **Hour 24+**: Engineering manager escalation

## Running Tests Locally

### Run specific test categories:
```bash
# Run only hourly tests
pytest -m "hourly" -v

# Run daily tests (excluding hourly)
pytest -m "daily and not hourly" -v

# Run all integration tests
pytest -m "integration" -v

# Run everything except slow tests
pytest -m "not slow" -v
```

### Simulate scheduled runs:
```bash
# Trigger scheduled tests manually
gh workflow run scheduled-tests.yml -f test_category=hourly
gh workflow run scheduled-tests.yml -f test_category=daily
gh workflow run scheduled-tests.yml -f test_category=all
```

## Best Practices

1. **New Features**: Add `@pytest.mark.hourly` to critical path tests
2. **Integration Points**: Always mark with `@pytest.mark.integration`
3. **Performance Tests**: Use `@pytest.mark.slow` for tests >30 seconds
4. **Flaky Tests**: Fix immediately or mark with `@pytest.mark.skip` with issue link
5. **Resource Usage**: Respect timeouts to avoid blocking other scheduled runs

## Maintenance

### Adding New Test Categories

1. Update `pyproject.toml` to add new marker:
   ```toml
   markers = [
       "new_category: description",
   ]
   ```

2. Update `.github/workflows/scheduled-tests.yml` to add new schedule

3. Document in this file

### Adjusting Schedules

Schedules use cron syntax in workflow files:
- Hourly: `15 * * * *` (15 minutes past every hour)
- Daily: `30 2 * * *` (2:30 AM UTC daily)
- Weekly: `0 3 * * 0` (3:00 AM UTC on Sundays)

Consider timezone impacts when adjusting schedules. Current times are optimized for US West Coast working hours.
