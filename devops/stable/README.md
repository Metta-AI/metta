# Release Qualification Guide

This document outlines the manual process for qualifying a release before tagging and pushing to the stable branch.

For questions about this release process, contact:

- Release Manager: @Robb
- Technical Lead: @Jack Heart
- Bug Triage: @Nishad Singh

## Release Qualification Checklist

### 1. Prepare Release Branch

- Create a temporary release qualification branch off of the commit you intend to tag

  ```bash
  git checkout -b release-qual/<VERSION>
  ```

- Push the branch to trigger CI

  ```bash
  git push -u origin release-qual/<VERSION>
  ```

- Review the triggered CI run and confirm it passes before proceeding.

### 2. Bug Status Check

- Open Asana project for bug tracking
- Verify no active/open bugs marked as blockers
- Update bug statuses as needed in consultation with bug owners
- Note any known issues that are acceptable for release

### 3. Workflow Tests

#### 3.1 Launch Training Runs

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

#### 3.2 Check Train Workflow

- Pick any Job ID from 3.1
- Verify job appears in W&B
- Verify training metrics (need to define further)
- Verify performance metrics: that SPS is near 40k and does not dip
- Verify model checkpoints are saved correctly

#### 3.3 Check Eval Workflow

- Pick any Job ID from 3.1
- Verify job appears in W&B
- Verify evaluation results
- Verify evaluation reports are generated
- Verify replay and replay link are generated

#### 3.4 Play Workflow

- Launch play environment from VS code
- Verify interactions work correctly by navigating an agent to collect a heart

#### 3.5 Sweep Workflow

- Ask Axel via discord to verify that sweeps are working to his satisfaction.

#### 3.6 Observatory Workflow

- Ask Pasha via discord to verify that observatory is working to his satisfaction.

#### 3.6 CI Workflow

- Verify CI pipeline has completed successful on the release commit push to main.

## 4. Release

#### 4.1 Prepare Release PR

1. Open a review PR from `release-qual/<VERSION>` into `stable` so the team can approve the release candidate.
   - Store the release notes as `devops/stable/release-notes/v<VERSION>.md` on the qualification branch and link to the
     file in the PR description.
   - Follow this template for the PR description:

```markdown
## Version <VERSION>

### Known Issues

<Notes from step 1>

### W&B Run Links

- Training: <link to job id examined in step 3.2>
- Evaluation: <link to job id examined step 3.3>
```

#### 4.2 Merge Release PR and update stable tag

1. After the PR is approved, create an annotated release tag on the qualification branch tip

   ```bash
   git tag -a v<VERSION> -m "Release version <VERSION>"
   ```

2. Click **Merge** on the PR.

3. Push the annotated tag and remove the qualification branch if it still exists on origin

   ```bash
   git push origin v<VERSION>
   git push origin --delete release-qual/<VERSION>
   ```

### 5. Announce

- Post release process completion to Discord in #eng-process

## Future Work

This process will be progressively automated. Planned automation includes:

- A `metta` script that walks through this process
- Selenium or Playwright for UI testing
- Automated W&B metrics extraction
- CI/CD integration for automatic checks
- Automated performance regression detection
