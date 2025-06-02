# Claude Review System Documentation

## Overview

The Claude Review System is a suite of AI-powered code review workflows that automatically analyze pull requests for specific improvements. Built on Anthropic's Claude AI, these workflows provide targeted, actionable feedback without the noise of traditional linters.

**Key Philosophy**: Only comment when there are genuine improvements to suggest. If everything looks good, stay silent.

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  GitHub Event   │────▶│ Review Workflow  │────▶│ Claude Analysis │
│  (PR opened)    │     │ (Specialized)    │     │ (via API)       │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                │                           │
                                ▼                           ▼
                        ┌──────────────────┐     ┌─────────────────┐
                        │ File Filtering   │     │ JSON Output     │
                        │ & Setup          │     │ (if issues)     │
                        └──────────────────┘     └─────────────────┘
                                                           │
                                                           ▼
                                                  ┌─────────────────┐
                                                  │ GitHub Review   │
                                                  │ with Suggestions│
                                                  └─────────────────┘
```

## Review Types

### 1. 📝 README Accuracy Review

**Purpose**: Ensures documentation stays accurate when code changes.

**Triggers**:
- Pull request opened or reopened
- Manual workflow dispatch

**What it checks**:
- Commands or CLI usage that no longer work
- Installation instructions that are now incorrect
- Removed dependencies still documented
- API endpoints or functions that were removed/renamed
- File paths that no longer exist
- Examples that would throw errors

**What it ignores**:
- Missing documentation (doesn't suggest additions)
- Style or formatting issues
- Opportunities for better documentation

### 2. 💬 Code Comments Review

**Purpose**: Identifies and removes unnecessary comments that clutter code.

**Triggers**:
- Pull request opened or reopened
- Manual workflow dispatch

**Comments flagged for removal**:
- Restating obvious code (e.g., `# increment counter` before `counter += 1`)
- Explaining self-evident operations
- Outdated comments that don't match implementation
- Duplicating information from variable/function names
- Stating obvious commands

**Comments preserved**:
- Explaining WHY something is done
- Important context or warnings
- TODO/FIXME comments (unless obsolete)
- Complex algorithm explanations
- License headers

### 3. 🔄 Einops Suggestions Review

**Purpose**: Suggests using `einops.rearrange` for complex tensor operations.

**Triggers**:
- Pull request opened or reopened (Python files only)
- Manual workflow dispatch

**Good candidates for einops**:
```python
# Complex and unclear:
x = x.permute(0, 2, 1).reshape(batch_size, -1)
# Suggested:
x = rearrange(x, 'b h w -> b (w h)')
```

**Ignored operations**:
- Simple, clear operations (`x.transpose(0, 1)`)
- When existing code is already readable
- Performance-critical sections
- Codebases not using einops

### 4. 🏷️ Type Annotations Review

**Purpose**: Identifies missing type annotations that would improve code quality.

**Triggers**:
- Pull request opened or reopened (Python files only)
- Manual workflow dispatch

**Priority levels**:
- **High**: Missing parameter types, public API return types, Optional returns
- **Medium**: Complex functions, non-obvious returns, empty collections
- **Ignored**: Private methods, obvious getters, simple functions, clear inference

**Modern syntax preferred**:
- `list[str]` over `List[str]`
- `type | None` over `Optional[type]`
- `dict[str, int]` over `Dict[str, int]`

## Base Workflow System

All reviews share a common base workflow (`claude-review-base.yml`) that provides:

### Two-Stage Process

**Stage 1: Claude Analysis**
- Runs specialized prompt for the review type
- Creates `claude-review-analysis.json` only if issues found
- No output means no issues (expected behavior)

**Stage 2: GitHub Review Creation**
- Only runs if Stage 1 found issues
- Creates PR review with inline suggestions
- Falls back to regular comment if review API fails

### JSON Output Format

When issues are found, Claude creates:

```json
{
  "review_summary": "Brief overall assessment",
  "review_status": "COMMENT|CHANGES_REQUESTED",
  "suggestions": [
    {
      "file": "path/to/file.py",
      "start_line": 23,
      "end_line": 24,
      "side": "RIGHT",
      "severity": "minor|major|blocking|nitpick",
      "reason": "Why this change improves the code",
      "original_code": "current code",
      "suggested_code": "improved code"
    }
  ],
  "compliments": [
    {
      "file": "path/to/file.py",
      "line": 45,
      "comment": "Positive feedback"
    }
  ],
  "tldr": [
    "Quick summary of changes"
  ]
}
```

### Features

- **File pattern matching**: Filter files to review
- **Python setup**: Optional Python environment configuration
- **Package installation**: Install dependencies for analysis
- **Tool access**: Configurable Claude tools (Edit, Replace, Bash, Write)
- **Manual trigger**: Review specific PRs via workflow dispatch
- **Detailed summaries**: GitHub Actions summary with results

## Configuration

### Required Secrets

```yaml
ANTHROPIC_API_KEY  # Claude API access key
```

### Workflow Inputs

Each specialized workflow can configure:

```yaml
review_name: "Display name for the review"
file_pattern: "Regex pattern for files to review"
setup_python: boolean
install_packages: "Space-separated package list"
tools: "Comma-separated Claude tools"
prompt: "Specialized review instructions"
pr_number: "Manual PR number (workflow_dispatch)"
```

## Usage Examples

### Manual Trigger
```bash
# Using GitHub CLI
gh workflow run "Claude Review: README" -f pr_number=123

# Using GitHub UI
# Navigate to Actions → Select workflow → Run workflow → Enter PR number
```

### Workflow Configuration
```yaml
name: "Claude Review: Custom"
on:
  pull_request:
    types: [opened, reopened]
    paths: ["**/*.rs"]  # Only Rust files

jobs:
  review:
    uses: ./.github/workflows/claude-review-base.yml
    secrets:
      anthropic_api_key: ${{ secrets.ANTHROPIC_API_KEY }}
    with:
      review_name: "Rust Safety"
      file_pattern: "\\.rs$"
      tools: "Edit,Replace,Bash(cargo check)"
      prompt: |
        Review Rust code for memory safety issues...
```

## Best Practices

### 1. Writing Review Prompts

**DO:**
- Start with clear "no issues found" instructions
- Define specific criteria for flagging issues
- Provide examples of good vs bad patterns
- Focus on actionable improvements

**DON'T:**
- Ask for general "improvements"
- Flag style preferences
- Suggest additions (only fixes)
- Create reviews for minor issues

### 2. Choosing Review Triggers

- Use `paths` filters to limit file types
- Consider `pull_request` event types carefully
- Always include `workflow_dispatch` for manual runs
- Avoid running on every commit (use PR events)

### 3. Tool Selection

**Common tool combinations:**
- `Edit,Replace,Bash(git diff HEAD~1)` - For code analysis
- `Edit,Replace,Bash(find . -name "*.py")` - For file discovery
- `Edit,Replace,Bash(mypy --version)` - For tool verification

## Performance Considerations

1. **File Filtering**: Use `file_pattern` to reduce analysis scope
2. **Conditional Execution**: Reviews only run when relevant files change
3. **Early Exit**: Claude stops immediately if no issues found
4. **Parallel Reviews**: Multiple review types can run simultaneously

## Troubleshooting

### Review Not Appearing
1. Check if relevant files were changed
2. Verify Claude found issues (check logs)
3. Ensure PR has appropriate permissions
4. Check for API rate limits

### Suggestions Not Inline
- File might not be in PR diff
- Line numbers might be incorrect
- GitHub API validation failure (check logs)

### Claude Not Finding Expected Issues
- Review prompt specificity
- Check file pattern matching
- Verify tool permissions
- Examine Claude's reasoning in logs

## Future Enhancements

Potential improvements to the system:

1. **Caching**: Cache reviews for unchanged files
2. **Incremental Reviews**: Only analyze changed sections
3. **Custom Severity**: Configure what triggers a review
4. **Team Preferences**: Per-repository configuration files
5. **Metrics**: Track fix rates and review effectiveness

## Creating New Review Types

To add a new specialized review:

1. Create workflow file: `.github/workflows/claude-review-[type].yml`
2. Use the base workflow
3. Define specific prompt with clear criteria
4. Test with manual dispatch
5. Document in this guide

Example template:
```yaml
name: "Claude Review: [Type]"
on:
  pull_request:
    types: [opened, reopened]
    paths: ["**/*.[ext]"]
  workflow_dispatch:
    inputs:
      pr_number:
        description: "Pull Request number to review"
        required: true
        type: string

permissions:
  contents: read
  pull-requests: write
  id-token: write

jobs:
  review:
    uses: ./.github/workflows/claude-review-base.yml
    secrets:
      anthropic_api_key: ${{ secrets.ANTHROPIC_API_KEY }}
    with:
      review_name: "[Review Type]"
      file_pattern: "\\.[ext]$"
      tools: "Edit,Replace,Bash(git diff HEAD~1)"
      pr_number: ${{ inputs.pr_number || '' }}
      prompt: |
        [Specific review instructions following the pattern:
        1. When to stay silent
        2. What to flag
        3. What to ignore
        4. Examples]
```
