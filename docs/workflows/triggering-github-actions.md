# Triggering GitHub Actions Without Merging a PR

This guide explains various methods to trigger GitHub Actions workflows without merging a pull request. This is particularly useful for testing Docker builds, running CI/CD pipelines, or executing other automated tasks during development.

## Table of Contents

1. [Manual Workflow Dispatch](#manual-workflow-dispatch)
2. [Push to Branch](#push-to-branch)
3. [Pull Request Events](#pull-request-events)
4. [GitHub API](#github-api)
5. [Local Testing with Act](#local-testing-with-act)
6. [Best Practices](#best-practices)

## Manual Workflow Dispatch

The most straightforward way to trigger a workflow manually is using the `workflow_dispatch` event.

### Step 1: Add workflow_dispatch to your workflow

Edit your `.github/workflows/your-workflow.yml` file:

```yaml
name: Build and Push Docker Image

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  workflow_dispatch:  # This enables manual triggering
    inputs:
      image_tag:
        description: 'Docker image tag'
        required: false
        default: 'latest'
      build_args:
        description: 'Additional build arguments'
        required: false
```

### Step 2: Trigger from GitHub UI

1. Go to your repository on GitHub
2. Click on the **Actions** tab
3. Select your workflow from the left sidebar
4. Click **Run workflow** button
5. Select the branch to run the workflow on
6. Fill in any inputs (if configured)
7. Click **Run workflow**

### Step 3: Monitor the run

The workflow will appear in the Actions tab with a 🟡 yellow dot while running.

## Push to Branch

You can trigger workflows by pushing to your feature branch if the workflow is configured to run on push events.

### Configure push triggers

```yaml
on:
  push:
    branches:
      - main
      - 'feature/**'  # Triggers on any feature/* branch
      - '**'          # Triggers on all branches
```

### Trigger by pushing

```bash
# Make your changes
git add .
git commit -m "Trigger workflow"
git push origin your-branch-name
```

## Pull Request Events

Workflows can be triggered on various PR events without merging:

```yaml
on:
  pull_request:
    types: [opened, synchronize, reopened, ready_for_review]
    branches: [main]
```

### Common PR triggers:

- **opened**: When PR is created
- **synchronize**: When new commits are pushed to the PR
- **reopened**: When a closed PR is reopened
- **ready_for_review**: When draft PR is marked ready

### Label-based triggers

```yaml
on:
  pull_request:
    types: [labeled]

jobs:
  build:
    if: contains(github.event.label.name, 'build-docker')
    runs-on: ubuntu-latest
    steps:
      # Your build steps
```

Add the label "build-docker" to your PR to trigger the workflow.

## GitHub API

Trigger workflows programmatically using the GitHub API:

### Using curl

```bash
# Set your variables
OWNER="your-github-username-or-org"
REPO="your-repo-name"
WORKFLOW_ID="your-workflow.yml"  # or workflow ID
GITHUB_TOKEN="your-personal-access-token"

# Trigger the workflow
curl -X POST \
  -H "Accept: application/vnd.github+json" \
  -H "Authorization: Bearer $GITHUB_TOKEN" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  https://api.github.com/repos/$OWNER/$REPO/actions/workflows/$WORKFLOW_ID/dispatches \
  -d '{
    "ref": "your-branch-name",
    "inputs": {
      "image_tag": "test-build",
      "build_args": "--no-cache"
    }
  }'
```

### Using GitHub CLI

```bash
# Install GitHub CLI if not already installed
# https://cli.github.com/

# Trigger workflow
gh workflow run "your-workflow.yml" \
  --ref your-branch-name \
  -f image_tag="test-build" \
  -f build_args="--no-cache"
```

## Local Testing with Act

[Act](https://github.com/nektos/act) allows you to run GitHub Actions locally:

### Installation

```bash
# macOS
brew install act

# Linux
curl https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash

# Or using Go
go install github.com/nektos/act@latest
```

### Basic usage

```bash
# List available workflows
act -l

# Run a specific workflow
act -W .github/workflows/docker-build.yml

# Run with specific event
act pull_request -W .github/workflows/docker-build.yml

# Run with secrets
act -s GITHUB_TOKEN="your-token" -s DOCKER_PASSWORD="your-password"
```

### Act configuration file

Create `.actrc` in your project root:

```
-P ubuntu-latest=catthehacker/ubuntu:act-latest
-P ubuntu-22.04=catthehacker/ubuntu:act-22.04
--container-architecture linux/amd64
```

## Best Practices

### 1. Use specific triggers

Instead of triggering on all events, be specific:

```yaml
on:
  push:
    branches: [main, develop]
    paths:
      - 'src/**'
      - 'Dockerfile'
      - '.github/workflows/docker-build.yml'
```

### 2. Implement concurrency controls

Prevent multiple runs of the same workflow:

```yaml
concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true
```

### 3. Use environment-specific builds

```yaml
jobs:
  build:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        environment: [dev, staging, prod]
    steps:
      - name: Build Docker image
        run: |
          docker build \
            --build-arg ENV=${{ matrix.environment }} \
            -t myapp:${{ matrix.environment }}-${{ github.sha }} \
            .
```

### 4. Conditional workflow execution

```yaml
jobs:
  docker-build:
    if: |
      github.event_name == 'workflow_dispatch' ||
      (github.event_name == 'pull_request' && contains(github.event.pull_request.labels.*.name, 'build-docker')) ||
      (github.event_name == 'push' && github.ref == 'refs/heads/main')
    runs-on: ubuntu-latest
    steps:
      # Your build steps
```

### 5. Use reusable workflows

Create a reusable workflow in `.github/workflows/reusable-docker-build.yml`:

```yaml
name: Reusable Docker Build

on:
  workflow_call:
    inputs:
      image_name:
        required: true
        type: string
      dockerfile_path:
        required: false
        type: string
        default: './Dockerfile'

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build Docker image
        run: |
          docker build -f ${{ inputs.dockerfile_path }} -t ${{ inputs.image_name }} .
```

Then call it from other workflows:

```yaml
jobs:
  build-app:
    uses: ./.github/workflows/reusable-docker-build.yml
    with:
      image_name: myapp:latest
      dockerfile_path: ./docker/Dockerfile
```

## Example: Docker Build and Push Workflow

Here's a complete example that combines several techniques:

```yaml
name: Docker Build and Push

on:
  push:
    branches: [main]
  pull_request:
    types: [opened, synchronize, labeled]
  workflow_dispatch:
    inputs:
      tag:
        description: 'Image tag'
        required: true
        default: 'latest'
      push_to_registry:
        description: 'Push to registry'
        required: true
        type: boolean
        default: false

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  build:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Log in to the Container registry
        if: github.event.inputs.push_to_registry == 'true' || github.event_name == 'push'
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: |
            type=ref,event=branch
            type=ref,event=pr
            type=sha,prefix={{branch}}-
            type=raw,value=${{ github.event.inputs.tag || 'latest' }}

      - name: Build and push Docker image
        uses: docker/build-push-action@v5
        with:
          context: .
          push: ${{ github.event.inputs.push_to_registry == 'true' || github.event_name == 'push' }}
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

This workflow:
- Builds on push to main (and pushes to registry)
- Builds on PR creation/update (without pushing)
- Can be manually triggered with custom tags
- Uses GitHub Container Registry
- Implements caching for faster builds

## Troubleshooting

### Common issues:

1. **Workflow not appearing in Actions tab**: Ensure the workflow file is in the default branch or the branch you're working on.

2. **Permission denied errors**: Check that your GitHub token has the necessary permissions (workflow, write:packages, etc.).

3. **Act not finding Docker**: Ensure Docker is running locally when using Act.

4. **Workflow dispatch not working**: The workflow_dispatch trigger must exist in the default branch for it to appear in the UI.

## Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Events that trigger workflows](https://docs.github.com/en/actions/using-workflows/events-that-trigger-workflows)
- [GitHub Actions API](https://docs.github.com/en/rest/actions/workflows)
- [Act - Run GitHub Actions Locally](https://github.com/nektos/act)
- [GitHub CLI](https://cli.github.com/)