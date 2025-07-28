# Stable Tag Advancement Workflow

This document describes the automated stable tag advancement system that ensures only thoroughly tested commits are marked as stable releases.

## Overview

The stable tag advancement workflow (`advance-stable-tag.yml`) automatically runs comprehensive tests after merges to the main branch and advances a "stable" tag when all tests pass. This provides a reliable reference point for deployments and ensures quality.

## Workflow Triggers

The workflow is triggered by:
- **Pushes to main branch**: Automatically runs after PR merges
- **Manual dispatch**: Can be triggered manually with custom parameters

## Comprehensive Test Suite

The workflow runs the following test categories:

### 1. Unit Tests
- Runs all unit tests across all packages
- Uses parallel test execution for speed
- Fails fast on first 5 failures to save time

### 2. Integration Tests
- `test_integration_sweep_pipeline.py`: Tests the sweep pipeline integration
- `test_docker_integration.py`: Validates Docker container integration
- `test_eval_task_orchestrator_integration.py`: Tests evaluation orchestration

### 3. Configuration Validation
- Validates all environment configurations
- Validates all curriculum configurations
- Validates all simulation configurations
- Tests typed configuration parsing and validation

### 4. Training Script Validation
- Verifies demo training scripts can be imported
- Runs a quick training test to ensure the training loop works

### 5. Performance Benchmarks
- Runs Python performance benchmarks
- Checks for performance regressions compared to previous stable tag
- Currently logs results; future versions will implement regression detection

## Grace Period

After all tests pass, the workflow enters a configurable grace period (default: 30 minutes) before advancing the tag. This allows for:
- Manual cancellation if issues are discovered
- Time to verify the build in staging environments
- Review of test results before marking as stable

The grace period uses GitHub Environments, allowing authorized users to:
- Cancel the deployment
- Approve early completion
- Review logs and artifacts

## Tag Naming Convention

Stable tags follow the format: `stable-YYYYMMDD-HHMMSS`

Example: `stable-20240115-143022`

Each tag includes an annotated message with:
- Commit SHA
- Test status summary
- Original commit message
- Timestamp

## Stability Dashboard

The workflow generates a comprehensive stability dashboard that tracks:

### Metrics
- Total number of stable tags
- Latest stable tag information
- Average time between stable releases
- Test success rate over time

### Historical Data
- Complete list of all stable tags
- Time intervals between releases
- Commit information for each tag
- Links to GitHub releases

### Generation
To generate the dashboard manually:
```bash
python tools/generate_stability_dashboard.py -o stability-dashboard
```

With workflow run data (requires GitHub CLI):
```bash
python tools/generate_stability_dashboard.py --include-workflow-runs
```

## Notifications

When a new stable tag is created, the workflow:

1. **Creates a GitHub Release**: Includes changelog with all changes since previous stable tag
2. **Logs notification details**: Currently logs what would be sent to Slack/Discord
3. **Future integrations**: Can be extended to send notifications to:
   - Slack channels
   - Discord webhooks
   - Email lists
   - Custom webhooks

## Manual Controls

### Skip Tests (Use with Caution!)
For emergency situations, tests can be skipped:
```bash
gh workflow run advance-stable-tag.yml -f skip_tests=true
```

### Custom Grace Period
Adjust the grace period duration:
```bash
gh workflow run advance-stable-tag.yml -f grace_period_minutes=60
```

## Failure Handling

If tests fail or performance regressions are detected:
- No stable tag is created
- Failure notifications are logged
- The workflow reports which tests failed
- Developers must fix issues before the next merge to main

## Integration with Other Workflows

The stable tags can be used by:
- Deployment workflows: Deploy only from stable tags
- Docker builds: Tag images with stable versions
- Release workflows: Create official releases from stable tags

## Best Practices

1. **Monitor the dashboard regularly**: Check for patterns in test failures
2. **Investigate gaps**: Long gaps between stable tags may indicate quality issues
3. **Use stable tags for deployments**: Always deploy from stable tags in production
4. **Review grace period cancellations**: Track why deployments were cancelled

## Troubleshooting

### No stable tag created after merge
- Check the workflow run logs for test failures
- Verify all integration tests are passing
- Check for performance regressions

### Dashboard not updating
- Ensure git tags are fetched: `git fetch --tags`
- Check permissions for GitHub CLI if using workflow data
- Verify the dashboard generation script has proper permissions

### Grace period issues
- Check GitHub Environment protection rules
- Verify authorized approvers are configured
- Review environment timeout settings