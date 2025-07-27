# Git Filter-Repo Tools

Simple tools for extracting and publishing child repositories from a monorepo while preserving commit history.

## What This Does

This system allows you to extract specific directories (like `mettagrid/` and `mettascope/`) from your monorepo and push them to a separate repository. The key feature is that it preserves the complete Git history for just those directories.

## Prerequisites

Install git-filter-repo using metta:
```bash
metta install filter-repo
```

Or install manually:
```bash
curl -O https://raw.githubusercontent.com/newren/git-filter-repo/main/git-filter-repo
chmod +x git-filter-repo
sudo mv git-filter-repo /usr/local/bin/
```

## Using with Metta

You can also run all filter-repo commands through metta:

```bash
# Filter repository
metta run filter-repo filter . mettagrid/ mettascope/

# Inspect filtered result
metta run filter-repo inspect /tmp/filtered-repo-xyz/filtered

# Push to production
metta run filter-repo push /tmp/filtered-repo-xyz/filtered git@github.com:yourorg/filter_repo_test.git
```

## Quick Start: One-Liner for Production

For production use, you can run all three steps in one command:

```bash
# Extract mettagrid/ and mettascope/ and push to filter_repo_test
./devops/git/sync_package.py filter . mettagrid/ mettascope/ && \
./devops/git/sync_package.py push $(ls -td /tmp/filtered-repo-*/filtered | head -1) git@github.com:yourorg/filter_repo_test.git -y
```

⚠️ **Warning**: This force-pushes immediately! Only use after you've tested the three-step process.

## Three-Step Process (Recommended)

### Step 1: Filter - Extract Directories

```bash
./devops/git/sync_package.py filter . mettagrid/ mettascope/
```

**What happens:**
1. Creates a fresh clone of your repository in a temporary directory
2. Uses `git-filter-repo` to remove everything EXCEPT the specified paths
3. Rewrites Git history to only include commits that touched those paths
4. Returns the path to the filtered repository (e.g., `/tmp/filtered-repo-abc123/filtered`)

**Important notes:**
- Your original repository is never modified
- The filtered repo is a complete Git repository with full history
- All commit SHAs will change (this is expected)

### Step 2: Inspect - Verify the Result

```bash
./devops/git/sync_package.py inspect /tmp/filtered-repo-abc123/filtered
```

**What you'll see:**
```
📊 Repository Statistics:
   Files: 127
   Commits: 534
   Location: /tmp/filtered-repo-abc123/filtered

📁 Files by directory:
   mettagrid/           89 files
   mettascope/          38 files

📄 Sample files:
   mettagrid/__init__.py
   mettagrid/core.py
   mettascope/app.py
   ... and 124 more
```

**Why this matters:**
- Confirms the filter worked correctly
- Shows you exactly what will be pushed
- Lets you verify no sensitive files were included

### Step 3: Push - Send to Production

Always do a dry run first:
```bash
./devops/git/sync_package.py push /tmp/filtered-repo-abc123/filtered git@github.com:yourorg/filter_repo_test.git --dry-run
```

Then push for real:
```bash
./devops/git/sync_package.py push /tmp/filtered-repo-abc123/filtered git@github.com:yourorg/filter_repo_test.git
```

**What happens:**
1. Adds the target repository as a remote named "production"
2. Force pushes the filtered history to the main branch
3. Completely replaces whatever was in the target repository

**⚠️ Critical warnings:**
- This FORCE PUSHES and overwrites the target repository
- All existing history in the target will be lost
- Make sure the target URL is correct!

## Complete Example Workflow

```bash
# 1. Filter the repository
$ ./devops/git/sync_package.py filter . mettagrid/ mettascope/
🔧 Filtering repository: /Users/you/monorepo
📁 Paths to extract: mettagrid/, mettascope/
Cloning for filtering...
Filtering to: mettagrid/, mettascope/
✅ Filtered: 127 files, 534 commits

✅ Success! Filtered repository at:
   /tmp/filtered-repo-abc123/filtered

Next steps:
   1. Inspect: ./devops/git/sync_package.py inspect /tmp/filtered-repo-abc123/filtered
   2. Push:    ./devops/git/sync_package.py push /tmp/filtered-repo-abc123/filtered <remote-url>

# 2. Inspect what we created
$ ./devops/git/sync_package.py inspect /tmp/filtered-repo-abc123/filtered
[... see output above ...]

# 3. Dry run to see what would happen
$ ./devops/git/sync_package.py push /tmp/filtered-repo-abc123/filtered git@github.com:yourorg/filter_repo_test.git --dry-run
📤 Pushing to: git@github.com:yourorg/filter_repo_test.git
   From: /tmp/filtered-repo-abc123/filtered
   Mode: DRY RUN

🔔 DRY RUN: Pushing...
To git@github.com:yourorg/filter_repo_test.git
 * [new branch]      HEAD -> main

✅ Dry run completed successfully!

# 4. Actually push
$ ./devops/git/sync_package.py push /tmp/filtered-repo-abc123/filtered git@github.com:yourorg/filter_repo_test.git
📤 Pushing to: git@github.com:yourorg/filter_repo_test.git
   From: /tmp/filtered-repo-abc123/filtered
   Mode: LIVE

⚠️  This will FORCE push. Continue? [y/N]: y

Pushing...
✅ Push completed successfully!
```

## How It Works Under the Hood

1. **Filtering Process**:
   - `git-filter-repo` walks through every commit in your repository
   - For each commit, it checks if any files match your specified paths
   - If a commit only touched files outside your paths, it's removed entirely
   - If a commit touched both included and excluded paths, it's rewritten to only include changes to your paths
   - The result is a clean history as if your repository only ever contained those directories

2. **Why Force Push**:
   - The filtered repository has completely different commit SHAs
   - Git won't allow a normal push because the histories have diverged
   - Force push replaces the entire remote repository with your filtered version

3. **Safety**:
   - Everything happens on temporary clones
   - Your source repository is never modified
   - You can inspect before pushing
   - Dry run shows exactly what will happen

## Common Issues

1. **"git-filter-repo not found"**: Install it using the command at the top
2. **"Not a git repository"**: Run from the root of your monorepo
3. **"Path not found"**: Check that your paths exist (typos in directory names)
4. **"Permission denied" on push**: Ensure you have write access to the target repository

## Architecture

- `common/src/metta/common/util/git.py`: Core `GitRepo` class with `filter_repo()` method
- `devops/git/sync_package.py`: CLI tool with filter/inspect/push commands
- `metta/setup/components/filter_repo.py`: Metta component for installing git-filter-repo
- Uses temporary directories for all operations (safe)
- Force pushes to target (destructive for target repo)