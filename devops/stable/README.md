# Release Qualification Guide

This document outlines the manual process for qualifying a release before tagging and pushing to the stable branch.

## Release Qualification Checklist

### 1. Notify the team

- Post intent to mark a new stable branch to Discord.

### 2. Code Quality Checks

- Run pytest suite from repo root and verify all tests pass

  ```bash
  make pytest
  ```

- Run code formatting check and verify no formatting changes are needed
  ```bash
  make format
  ```

### 4. Documentation Verification

- **Main Repository README**
  - Manually execute all steps in the main repo README
  - Document any issues or unclear instructions
  - Verify all examples work as expected

- **Mettagrid README**
  - Manually execute all steps in the mettagrid README
  - Document any issues or unclear instructions
  - Verify all examples work as expected

### 4 Workflow Tests

#### 4.1 Launch Training Runs

- **Cluster Test**

  ```bash
  devops/skypilot/tests/cluster_test launch
  devops/skypilot/tests/cluster_test check
  ```

- **Recipe Test**
  ```bash
  tests/experiments/recipes/recipe_test launch
  tests/experiments/recipes/recipe_test check
  ```

#### 4.2 Check Train Workflow

- Pick any Job ID from 4.1
- Verify job appears in W&B
- Verify training metrics
- Verify model checkpoints are saved correctly

#### 4.3 Check Eval Workflow

- Pick any Job ID from 4.1
- Verify job appears in W&B
- Verify evaluation results
- Verify evaluation reports are generated
- Verify replay and replay link are generated

#### 4.4 Play Workflow

- Launch play environment from VS code
- Verify interactions work correctly by navigating an agent to collect a heart

#### 4.5 Sweep Workflow

- Ask Axel via discord to verify that sweeps are working to his satisfaction.

#### 4.6 CI Workflow

- Verify CI pipeline has completed successful on the release commit push to main.

### 5 Bug Status Check

- Open Asana project for bug tracking
- Verify no active/open bugs marked as blockers
- Document any known issues that are acceptable for release
- Update bug statuses as needed in consultation with bug owners

### 6 Performance Benchmarks

- Compare github test run against previous tagged release in W&B
- Document any performance regressions or improvements

### 6 Final Step

- Post release process completion to Discord.

## Release Tagging Process

Once all checks pass:

1. Note the current commit hash

   ```bash
   git rev-parse HEAD
   ```

2. Create release tag

   ```bash
   git tag -a v<VERSION> -m "Release version <VERSION>"
   ```

3. Push to stable branch
   ```bash
   git checkout stable
   git merge <commit-hash>
   git push origin stable
   git push origin v<VERSION>
   ```

## Release Notes Template

```markdown
## Version <VERSION>

### Release Date: <DATE>

### Commit Hash: <HASH>

### Key Changes

- Feature 1
- Feature 2
- Bug fixes

### Performance Metrics

- Training speed: X
- Evaluation throughput: Y

### Known Issues

- Issue 1 (non-blocking)
- Issue 2 (workaround available)

### W&B Run Links

- Training: <link to examined job id>
- Evaluation: <link to examined job id>
```

## Future Automation

This process will be progressively automated. Planned automation includes:

- Selenium or Playwright for UI testing
- Automated W&B metrics extraction
- CI/CD integration for automatic checks
- Automated performance regression detection

## Troubleshooting

### Common Issues

1. **Cluster test fails**
   - Check cloud credentials
   - Verify resource quotas
   - Review cluster logs

2. **W&B metrics missing**
   - Verify W&B API key
   - Check network connectivity
   - Review W&B project settings

3. **Documentation steps fail**
   - Note exact step that failed
   - Check for environment differences
   - Update documentation if needed

## Contact

For questions about this release process, contact:

- Release Manager: @Robb
- Technical Lead: @Jack Heart
- Bug Triage: @Richard Higgins
