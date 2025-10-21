# Gitta

A Python library for Git operations, GitHub API interactions, and AI-powered PR splitting.

## Features

- **Core Git Operations**: Run git commands with consistent error handling and type safety
- **Git Utilities**: Branch management, commit tracking, diff analysis, repository validation
- **GitHub API Integration**: Create PRs, post commit statuses, query PR information
- **Authenticated GitHub API**: Multiple authentication methods (token, custom headers)
- **GitHub Actions**: Query workflow runs and job statuses
- **Commit Management**: List and filter commits with pagination support
- **Repository Filtering**: Extract specific paths using git subtree split
- **AI-Powered PR Splitting**: Automatically split large PRs into smaller, logical units

## Installation

### From Monorepo (Current)

```bash
# Install the metta workspace which includes gitta
uv sync
```

### Standalone (Future)

```bash
# Will be available after export to standalone repo
uv pip install gitta
```

## Quick Start

### Basic Git Operations

```python
import gitta

# Get current branch and commit
branch = gitta.get_current_branch()
commit = gitta.get_current_commit()

# Check for changes
has_changes, status = gitta.has_unstaged_changes()

# Run git commands safely
try:
    output = gitta.run_git("log", "--oneline", "-5")
except gitta.GitError as e:
    print(f"Git error: {e}")
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
import gitta

# Create a pull request (requires GITHUB_TOKEN)
pr = gitta.create_pr(
    repo="owner/repo",
    title="feat: Add new feature",
    body="## Summary\n\nDetailed description...",
    head="feature-branch",
    base="main",
    token="ghp_your_token"  # Optional
)
print(f"Created PR: {pr['html_url']}")

# Post commit status
gitta.post_commit_status(
    commit_sha="abc123def456",
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

### Repository Validation

```python
from pathlib import Path
import gitta

# Find git repository root
repo_root = gitta.find_root(Path.cwd())

# Validate repository state before running operations
commit_hash = gitta.validate_commit_state(
    require_clean=True,           # No uncommitted changes
    require_pushed=True,           # Commit exists on remote
    target_repo="owner/repo",      # Optional: verify correct repo
    allow_untracked=False          # No untracked files
)

# Check if specific commit is pushed
if gitta.is_commit_pushed("abc123"):
    print("Commit is on remote")
```

## PR Splitting

Split large pull requests into smaller, logically isolated PRs using AI analysis.

### Prerequisites

```bash
# Required: Anthropic API key for AI analysis
export ANTHROPIC_API_KEY="your-anthropic-api-key"

# Optional: GitHub token for automatic PR creation
export GITHUB_TOKEN="your-github-token"
```

### Basic Usage

```bash
# From your feature branch with uncommitted changes
python -m gitta.split
```

The splitter will:
1. Analyze your changes using Claude AI
2. Determine logical groupings (e.g., refactors vs features, frontend vs backend)
3. Create two new branches with split changes
4. Verify all changes are preserved
5. Push branches to origin
6. Create GitHub PRs (if token provided)

### Advanced Options

```bash
# Use command-line arguments
python -m gitta.split \
  --model claude-sonnet-4-5 \
  --skip-hooks \
  --commit-timeout 600

# Or set environment variables
export GITTA_SPLIT_MODEL="claude-sonnet-4-5"
export GITTA_SKIP_HOOKS="1"
export GITTA_COMMIT_TIMEOUT="600"
python -m gitta.split
```

### Programmatic API

```python
from gitta.split import split_pr

split_pr(
    anthropic_api_key="your-key",
    github_token="your-token",
    model="claude-sonnet-4-5",
    skip_hooks=True,
    commit_timeout=600
)
```

## Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GITHUB_TOKEN` | No | - | GitHub personal access token for API operations |
| `ANTHROPIC_API_KEY` | Yes (for splitting) | - | Anthropic API key for AI-powered PR analysis |
| `GITTA_SPLIT_MODEL` | No | `claude-sonnet-4-5` | Anthropic model name to use for PR splitting |
| `GITTA_SKIP_HOOKS` | No | `false` | Set to `1` to skip git hooks (use `--no-verify`) when committing |
| `GITTA_COMMIT_TIMEOUT` | No | `300` | Git commit timeout in seconds |

### Getting API Keys

**GitHub Token:**
1. Go to GitHub Settings → Developer settings → Personal access tokens
2. Generate a token with `repo` scope
3. Set as `GITHUB_TOKEN` environment variable

**Anthropic API Key:**
1. Sign up at [console.anthropic.com](https://console.anthropic.com)
2. Generate an API key from your account settings
3. Set as `ANTHROPIC_API_KEY` environment variable

## Error Handling

Gitta provides specific exception types for different error conditions:

```python
import gitta

try:
    gitta.run_git("push", "origin", "main")
except gitta.GitNotInstalledError:
    print("Git is not installed")
except gitta.NotAGitRepoError:
    print("Not in a git repository")
except gitta.DubiousOwnershipError as e:
    print(f"Repository ownership issue: {e}")
    # Follow the fix instructions in the error message
except gitta.GitError as e:
    print(f"Git command failed: {e}")
```

## Troubleshooting

### "dubious ownership" Error

If you encounter a "detected dubious ownership in repository" error:

```bash
# Fix by adding the repository to git's safe directories
git config --global --add safe.directory /path/to/repo
```

This typically occurs in Docker containers or when repository ownership differs from the current user.

### PR Splitting Issues

**"Need at least 2 files to split"**
- PR splitting requires changes in at least 2 files to create meaningful splits
- Consider making your PR smaller or keeping it as-is

**"Working tree has uncommitted changes"**
- Commit or stash your changes before running the splitter
- The splitter requires a clean working tree to safely create branches

**API Rate Limits**
- GitHub API has rate limits (5000/hour authenticated, 60/hour unauthenticated)
- Anthropic API has usage limits based on your plan
- Use tokens to get higher rate limits

## Development

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=gitta --cov-report=term-missing

# Run specific test file
uv run pytest tests/test_split.py -v
```

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
├── src/gitta/
│   ├── __init__.py      # Main exports
│   ├── core.py          # Core git command runner
│   ├── git.py           # Git operations
│   ├── github.py        # GitHub API functionality
│   ├── filter.py        # Repository filtering
│   └── split.py         # PR splitting logic and CLI
├── tests/
│   ├── test_core.py     # Core functionality tests
│   ├── test_git.py      # Git operations tests
│   ├── test_github.py   # GitHub API tests
│   ├── test_filter.py   # Filtering tests
│   └── test_split.py    # PR splitting tests
├── pyproject.toml       # Package configuration
└── README.md            # This file
```

## Examples

### Safe Remote Operations

```python
import gitta

# Validate state before pushing
try:
    commit = gitta.validate_commit_state(
        require_clean=True,
        require_pushed=False
    )
    gitta.run_git("push", "origin", "main")
except gitta.GitError as e:
    print(f"Cannot push: {e}")
```

### Branch Management

```python
import gitta

# Get current branch
current = gitta.get_current_branch()
print(f"On branch: {current}")

# Check if commit is on remote
commit = gitta.get_current_commit()
if gitta.is_commit_pushed(commit):
    print("Current commit is pushed")
```

### Repository Filtering

```python
import gitta
from pathlib import Path

# Extract subdirectory to new repo
gitta.filter_repo(
    source_repo=Path("/path/to/source"),
    target_repo=Path("/path/to/target"),
    paths_to_keep=["packages/mylib"]
)
```

## License

Part of the Metta AI project. See main repository for license information.

## Contributing

This package is currently maintained as part of the Metta monorepo. When exported as a standalone package, contribution guidelines will be added.

For now, follow the main Metta repository's contribution process.
