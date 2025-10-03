# Release Qualification Guide

This document outlines the manual process for qualifying a release before tagging and pushing to the **stable** branch.

For questions about this release process, contact:

- Release Manager: @Robb
- Technical Lead: @Jack Heart
- Bug Triage: @Nishad Singh

---

## Release Qualification Checklist

### 1. Prepare Staging Branch

Create a versioned staging branch at the commit you intend to tag:

```bash
git checkout main
git pull
git checkout -B staging/v<VERSION>-rc1 <COMMIT_SHA>
git push -u origin staging/v<VERSION>-rc1
```

You may tag release candidates directly on this branch for tracking:

```bash
git tag -a v<VERSION>-rc1 -m "Release candidate 1 for v<VERSION>"
git push origin v<VERSION>-rc1
```

> Example: `staging/v1.2.0-rc1`

---

### 2. Bug Status Check

- Open the Asana project for bug tracking.
- Verify no active/open bugs marked as blockers.
- Update bug statuses as needed in consultation with bug owners.
- Note any known issues that are acceptable for release.

---

### 3. Workflow Tests

#### 3.1 Cluster Tests

```bash
devops/skypilot/tests/cluster_test launch
devops/skypilot/tests/cluster_test check
```

- Verify all jobs run to completion in Skypilot logs.
- Confirm no unexpected terminations.

#### 3.2 Recipe Tests

```bash
tests/experiments/recipes/recipe_test launch
tests/experiments/recipes/recipe_test check
```

- Verify jobs run through to completion in Skypilot.
- Confirm W&B logs and artifacts are generated.

#### 3.3 Train Workflow

- Find the `experiments/recipes/arena_basic_easy_shaped` Job ID launched in the recipe tests:

```bash
tests/experiments/recipes/recipe_test check
```

- Verify data from the job appears correctly in W&B.
- Record key metrics for release notes (see template below):
  - **Final loss convergence**
  - **Final reward convergence**
  - **Elapsed training time**
  - **GPU and CPU usage**
  - **GPU and CPU memory usage**
  - **Training throughput (steps/sec)**

- Verify model checkpoints are saved and accessible in the artifact store.

#### 3.4 Eval Workflow

- Find the `experiments/recipes/arena_basic_easy_shaped` Job ID launched in the recipe tests:

```bash
tests/experiments/recipes/recipe_test check
```

- Verify job appears in W&B.
- Confirm evaluation reports are generated and artifacts are saved.
- Verify replay is functional and replay link is accessible.

#### 3.5 Play Workflow

- Launch play environment from VS Code.
- Verify interactions work correctly by navigating an agent to collect a heart.

#### 3.6 Sweep Workflow

- Ask Axel (via Discord) to confirm sweep jobs are working to his satisfaction.

#### 3.7 Observatory Workflow

- Ask Pasha (via Discord) to confirm observatory is working to his satisfaction.

#### 3.8 CI Workflow

- Verify CI pipeline has completed successfully on the **staging/v<VERSION>-rcX** branch (current release candidate).

---

### 4. Release

#### 4.1 Prepare Release PR

1. Open a review PR from `staging/v<VERSION>-rcX` → `stable` so the team can approve the release candidate.
   - Store the release notes as `devops/stable/release-notes/v<VERSION>.md` on the staging branch and link to the file
     in the PR description.
   - Use this PR template:

```markdown
## Version <VERSION>

### Known Issues

<Notes from step 2>

### W&B Run Links

- Training: <link to job id examined in step 3.3>
- Evaluation: <link to job id examined in step 3.4>

### Key Metrics

- Final loss: <value>
- Final reward: <value>
- Elapsed training time: <value>
- GPU usage: <value>
- CPU usage: <value>
- GPU memory usage: <value>
- CPU memory usage: <value>
- Training throughput (steps/sec): <value>
```

#### 4.2 Merge and Tag

1. After the PR is approved, merge into `stable`.
2. Checkout `stable` and create the final annotated release tag:

```bash
git checkout stable
git pull
git tag -a v<VERSION> -m "Release version <VERSION>"
git push origin v<VERSION>
```

---

### 5. Announce

- Post completion notice in Discord `#eng-process` with:
  - A link to the PR from `staging/v<VERSION>-rcX` → `stable`.
  - A link to the release notes file.
  - Any significant performance changes observed.

---

## Future Work

This process will be progressively automated. Planned automation includes:

- A `metta stable-release` command (unifying tagging and version bumps).
- Selenium or Playwright for UI testing.
- Automated W&B metrics extraction.
- CI/CD integration for automatic checks.
- Automated performance regression detection.
