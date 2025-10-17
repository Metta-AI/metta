# Gitta

A Python library for Git operations, GitHub API interactions, and AI-powered PR splitting.

## Features

- **Core Git Operations** - Run git commands with consistent error handling and type safety
- **Git Utilities** - Branch management, commit tracking, diff analysis, repository validation
- **GitHub API** - Create PRs, post commit statuses, query PR information
- **Repository Filtering** - Extract specific paths using git-filter-repo
- **AI-Powered PR Splitting** - Automatically split large PRs into smaller, logical units

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

```python
import gitta

# Create a pull request (requires GITHUB_TOKEN)
pr = gitta.create_pr(
    repo="owner/repo",
    title="feat: Add new feature",
    body="## Summary\n\nDetailed description...",
    head="feature-branch",
    base="main"
)
print(f"Created PR: {pr['html_url']}")

# Post commit status
gitta.post_commit_status(
    commit_sha="abc123def456",
    state="success",
    repo="owner/repo",
    description="All tests passed",
    context="ci/tests"
)
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
| `ANTHROPIC_API_KEY` | Yes (for splitting) | - | Anthropic API key for AI-powered PR analysis |
| `GITHUB_TOKEN` | No | - | GitHub personal access token for API operations |
| `GITTA_SPLIT_MODEL` | No | `claude-sonnet-4-5` | Anthropic model name to use for PR splitting |
| `GITTA_SKIP_HOOKS` | No | `false` | Set to `1` to skip git hooks (use `--no-verify`) when committing |
| `GITTA_COMMIT_TIMEOUT` | No | `300` | Git commit timeout in seconds |

### Getting API Keys

**Anthropic API Key:**
1. Sign up at [console.anthropic.com](https://console.anthropic.com)
2. Generate an API key from your account settings
3. Set as `ANTHROPIC_API_KEY` environment variable

**GitHub Token:**
1. Go to GitHub Settings → Developer settings → Personal access tokens
2. Generate a token with `repo` scope
3. Set as `GITHUB_TOKEN` environment variable

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

### Project Structure

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
