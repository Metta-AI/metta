# Gitta - Git Utilities Library

A comprehensive Python library for Git operations, GitHub API interactions, and advanced repository management.

## Features

- **Core Git Operations**: Run git commands with consistent error handling
- **Git Utilities**: Branch management, commit tracking, diff analysis
- **GitHub API Integration**: Create PRs, post commit statuses, query PR information
- **Authenticated GitHub API**: Multiple authentication methods (token, custom headers)
- **GitHub Actions**: Query workflow runs and job statuses
- **Commit Management**: List and filter commits with pagination support
- **Repository Filtering**: Extract specific paths using git subtree split

## Installation

We recommend using [uv](https://docs.astral.sh/uv/) to manage virtual environments and dependencies:

```bash
uv venv
source .venv/bin/activate
uv pip install gitta
```

## Usage

### Basic Git Operations

```python
import gitta

# Get current branch
branch = gitta.get_current_branch()

# Get current commit
commit = gitta.get_current_commit()

# Check for unstaged changes
has_changes, status = gitta.has_unstaged_changes()

# Run any git command
output = gitta.run_git("log", "--oneline", "-5")
```

### GitHub API

#### Authentication

Gitta supports multiple authentication methods for GitHub API requests:

```python
# Using GITHUB_TOKEN environment variable (automatic)
commits = gitta.get_commits(repo="owner/repo", branch="main")

# Passing token explicitly
commits = gitta.get_commits(
    repo="owner/repo",
    branch="main",
    token="ghp_your_token_here"
)

# Custom authentication headers (e.g., Basic auth)
commits = gitta.get_commits(
    repo="owner/repo",
    branch="main",
    Authorization="Basic your_base64_token"
)

# Using the github_client context manager for custom requests
with gitta.github_client(repo="owner/repo", token="ghp_token") as client:
    response = client.get("/issues")
    issues = response.json()
```

#### Working with Commits

```python
# Get commits from a repository
commits = gitta.get_commits(
    repo="owner/repo",
    branch="main",
    since="2024-01-01T00:00:00Z",  # ISO 8601 timestamp
    per_page=100
)

for commit in commits:
    print(f"{commit['sha'][:7]} - {commit['commit']['message']}")
```

#### GitHub Actions Workflows

```python
# Get workflow runs
runs = gitta.get_workflow_runs(
    repo="owner/repo",
    workflow_filename="checks.yml",
    branch="main",
    status="completed",  # or "in_progress", "queued"
    per_page=10
)

# Get jobs for a specific workflow run
if runs:
    run_id = runs[0]["id"]
    jobs = gitta.get_workflow_run_jobs(
        repo="owner/repo",
        run_id=run_id,
        per_page=100
    )

    for job in jobs:
        print(f"{job['name']}: {job['status']} - {job['conclusion']}")
```

#### Pull Requests and Commit Statuses

```python
# Create a pull request
pr = gitta.create_pr(
    repo="owner/repo",
    title="My PR",
    body="Description",
    head="feature-branch",
    base="main",
    token="ghp_your_token"  # Optional
)

# Post commit status
gitta.post_commit_status(
    commit_sha="abc123",
    state="success",
    repo="owner/repo",
    description="Tests passed",
    token="ghp_your_token"  # Optional
)

# Check if commit is part of an open PR
pr_info = gitta.get_matched_pr(
    commit_hash="abc123",
    repo="owner/repo"
)
if pr_info:
    pr_number, pr_title = pr_info
    print(f"Part of PR #{pr_number}: {pr_title}")
```

## Advanced Options

```python
# Custom error handling
try:
    gitta.run_git("push", "origin", "main")
except gitta.GitError as e:
    print(f"Git error: {e}")

# Check if commit is pushed
if gitta.is_commit_pushed("abc123"):
    print("Commit is on remote!")

# Get repository root
repo_root = gitta.find_root(Path.cwd())

# Validate commit state before remote execution
commit_hash = gitta.validate_commit_state(
    require_clean=True,      # Require no uncommitted changes
    require_pushed=True,     # Require commit is pushed to remote
    target_repo="owner/repo", # Optional: verify we're in correct repo
    allow_untracked=False    # Whether to allow untracked files
)
```

## Environment Variables

- `GITHUB_TOKEN`: GitHub personal access token for API operations
- `GITTA_AUTO_ADD_SAFE_DIRECTORY`: Set to "1" to auto-handle git safe directory issues

## API Reference

### GitHub API Functions

#### `github_client(repo, token=None, base_url=None, timeout=30.0, **headers)`

Context manager that creates an authenticated httpx.Client for GitHub API requests.

**Parameters:**
- `repo` (str): Repository in format "owner/repo"
- `token` (str, optional): GitHub token. Uses GITHUB_TOKEN env var if not provided
- `base_url` (str, optional): Custom base URL for API
- `timeout` (float): Request timeout in seconds (default: 30.0)
- `**headers`: Additional headers (can override Authorization for custom auth)

**Returns:** httpx.Client configured for GitHub API

**Example:**
```python
with gitta.github_client(repo="owner/repo", token="ghp_token") as client:
    response = client.get("/issues")
    issues = response.json()
```

#### `get_commits(repo, branch="main", since=None, per_page=100, token=None, **headers)`

Get list of commits from a repository with automatic pagination.

**Parameters:**
- `repo` (str): Repository in format "owner/repo"
- `branch` (str): Branch name (default: "main")
- `since` (str, optional): ISO 8601 timestamp to filter commits
- `per_page` (int): Number of commits per page, max 100 (default: 100)
- `token` (str, optional): GitHub token for authentication
- `**headers`: Additional headers (can override Authorization)

**Returns:** List of commit objects from GitHub API

**Raises:** GitError if API request fails

#### `get_workflow_runs(repo, workflow_filename, branch=None, status=None, per_page=1, token=None, **headers)`

Get workflow runs for a specific GitHub Actions workflow.

**Parameters:**
- `repo` (str): Repository in format "owner/repo"
- `workflow_filename` (str): Workflow filename (e.g., "checks.yml")
- `branch` (str, optional): Filter by branch name
- `status` (str, optional): Filter by status ("completed", "in_progress", "queued")
- `per_page` (int): Number of runs per page (default: 1)
- `token` (str, optional): GitHub token for authentication
- `**headers`: Additional headers (can override Authorization)

**Returns:** List of workflow run objects

**Raises:** GitError if API request fails

#### `get_workflow_run_jobs(repo, run_id, per_page=100, token=None, **headers)`

Get jobs for a specific workflow run.

**Parameters:**
- `repo` (str): Repository in format "owner/repo"
- `run_id` (int): Workflow run ID
- `per_page` (int): Number of jobs per page (default: 100)
- `token` (str, optional): GitHub token for authentication
- `**headers`: Additional headers (can override Authorization)

**Returns:** List of job objects

**Raises:** GitError if API request fails

## Module Structure

```
gitta/
├── __init__.py      # Public package surface, re-exporting submodules
├── core.py          # Low-level git command runners and shared exceptions
├── git.py           # High-level git operations built on top of core helpers
├── github.py        # GitHub REST/CLI integrations
└── filter.py        # Repository filtering utilities
```
