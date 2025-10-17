# Gitta - Git Utilities Library

A comprehensive Python library for Git operations, GitHub API interactions, and advanced repository management.

## Features

- **Core Git Operations**: Run git commands with consistent error handling
- **Git Utilities**: Branch management, commit tracking, diff analysis
- **GitHub API Integration**: Create PRs, post commit statuses, query PR information
- **Repository Filtering**: Extract specific paths using git-filter-repo
- **PR Splitting**: Automatically split large PRs into smaller, logical units using AI

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

```python
# Create a pull request
pr = gitta.create_pr(
    repo="owner/repo",
    title="My PR",
    body="Description",
    head="feature-branch",
    base="main"
)

# Post commit status
gitta.post_commit_status(
    commit_sha="abc123",
    state="success",
    repo="owner/repo",
    description="Tests passed"
)
```

## PR Splitting

Split large PRs into smaller, logically isolated ones:

```bash
# Set environment variables
export ANTHROPIC_API_KEY="your-key"
export GITHUB_TOKEN="your-token"      # Optional, for auto-creating PRs
# Optional overrides
export GITTA_SPLIT_MODEL="claude-sonnet-4-5"
export GITTA_SKIP_HOOKS="1"
export GITTA_COMMIT_TIMEOUT="600"

# Run the splitter
python -m gitta.split
# Inline overrides
python -m gitta.split --model claude-sonnet-4-5 --skip-hooks --commit-timeout 600
```

The PR splitter will:

1. Analyze your changes using AI to determine logical groupings
2. Create two new branches with the split changes
3. Verify no changes are lost
4. Push the branches to origin
5. Create pull requests (if GitHub token provided)

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
- `ANTHROPIC_API_KEY`: API key for AI-powered PR splitting
- `GITTA_SPLIT_MODEL`: Anthropic model name (defaults to `claude-sonnet-4-5`)
- `GITTA_SKIP_HOOKS`: Set to `1` to append `--no-verify` when committing split branches
- `GITTA_COMMIT_TIMEOUT`: Override the git commit timeout (seconds; defaults to 300)

## Module Structure

```
gitta/
├── __init__.py      # Main exports
├── core.py          # Core git command runner
├── git.py           # Git operations
├── github.py        # GitHub API functionality
├── filter.py        # Repository filtering
└── split.py         # PR splitting logic and CLI
```
