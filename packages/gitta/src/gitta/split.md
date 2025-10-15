# Quick Start: PR Splitting with Gitta

This guide will help you get started with splitting large PRs using gitta.

## Prerequisites

1. **Install gitta:**

   ```bash
   pip install gitta
   ```

2. **(For AI-driven splitting) Get an Anthropic API key:**
   - Sign up at https://console.anthropic.com/
   - Create an API key
   - Set it as an environment variable:
     ```bash
     export ANTHROPIC_API_KEY="sk-ant-..."
     ```

3. **Optional: Set up GitHub token for auto PR creation:**
   - Go to https://github.com/settings/tokens
   - Create a token with `repo` scope
   - Set it as an environment variable:
     ```bash
     export GITHUB_TOKEN="ghp_..."
     ```

## Basic Usage

### Command Line

The easiest way to split a PR is from the command line:

```bash
# Navigate to your repository
cd my-project

# Make sure you're on the branch you want to split
git checkout feature/big-refactor

# Run the splitter
python -m gitta.split
```

## What Happens

When you run the splitter, it will:

1. **Analyze your changes** - Get the diff between your branch and the base branch
2. **Use AI to decide the split** - Claude analyzes the files and suggests logical groupings
3. **Show the split plan** - You'll see which files go into each PR
4. **Create two new branches** - Named `<your-branch>-part1` and `<your-branch>-part2`
5. **Apply the changes** - Each branch gets its subset of changes
6. **Push to origin** - Both branches are pushed
7. **Create PRs** - If you have a GitHub token, PRs are created automatically

## Example Output

```
🔄 Starting PR split process...
📍 Current branch: feature/big-refactor
📍 Base branch: origin/main
📥 Getting diff...
📊 Found 8 changed files
🤖 Analyzing with AI to determine split strategy...

📋 Split Decision:
Group 1 (4 files): Refactor authentication system
  Files: auth.py, login.py, session.py, test_auth.py
  Description: Updates core authentication logic and session handling

Group 2 (4 files): Add user profile features
  Files: profile.py, views.py, templates/profile.html, test_profile.py
  Description: Implements user profile viewing and editing

✂️  Creating patches...
✅ Verifying split integrity...
  Verification passed!
🌿 Creating branch: feature/big-refactor-part1
🌿 Creating branch: feature/big-refactor-part2
📤 Pushing branches...
🔧 Creating pull requests...
Created PR: https://github.com/owner/repo/pull/123
Created PR: https://github.com/owner/repo/pull/124
✨ PR split complete!
```

> **Tip:** You can still use `PRSplitter` helpers (such as `parse_diff` or `create_patch_file`) without an Anthropic key, which is handy for scripting or testing scenarios.

## Tips

1. **Review before pushing** - The tool shows the split plan before creating branches
2. **Check the PRs** - Always review the created PRs before merging
3. **Keep your original branch** - It's preserved as a backup
4. **Works best with 2-20 files** - Very large PRs might need manual adjustment
5. **Logical grouping** - The AI tries to keep related changes together
6. **Rerun safely** - If you rerun the splitter, existing `*-part{n}` branches are recreated from the base branch automatically and remote branches are refreshed with `--force-with-lease` when needed (pass `--no-force-push` if you prefer the command to fail instead).

## Troubleshooting

### "Not in a git repository"

Make sure you're in a git repository with commits.

### "Could not determine base branch"

The tool looks for `main`, `master`, or `develop`. If you use a different base:

```python
splitter = PRSplitter()
splitter.base_branch = "origin/your-base-branch"
splitter.split()
```

### "No changes detected"

Make sure you have uncommitted changes or commits not in the base branch.

### API Key Issues

Verify your API key:

```bash
echo $ANTHROPIC_API_KEY  # Should show your key
```

### GitHub PR Creation Failed

- Check your GitHub token has `repo` scope
- Make sure the remote is set to a GitHub repository
- Verify you have push access to the repository

## Next Steps

- Read the full documentation in the README
- Check out the examples directory
- Customize the AI prompts for your project's needs
- Integrate into your CI/CD workflow
