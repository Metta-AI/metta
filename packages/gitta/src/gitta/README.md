# Gitta - Git Utilities Library

A comprehensive Python library for Git operations, GitHub API interactions, and advanced repository management.

## Features

- **Core Git Operations**: Run git commands with consistent error handling
- **Git Utilities**: Branch management, commit tracking, diff analysis
- **GitHub API Integration**: Create PRs, post commit statuses, query PR information
- **Repository Filtering**: Extract specific paths using git-filter-repo

## Installation

```bash
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
```

## Environment Variables

- `GITHUB_TOKEN`: GitHub personal access token for API operations
- `GITTA_AUTO_ADD_SAFE_DIRECTORY`: Set to "1" to auto-handle git safe directory issues

## Module Structure

```
gitta/
├── __init__.py      # Main exports
├── core.py          # Core git command runner
├── git.py           # Git operations
├── github.py        # GitHub API functionality
└── filter.py        # Repository filtering
```
