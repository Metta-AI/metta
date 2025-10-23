# Claude Review System Documentation

## Overview

The Claude Review System is a suite of AI-powered code review workflows that automatically analyze pull requests for
specific improvements. Built on Anthropic's Claude AI, these workflows provide targeted, actionable feedback without the
noise of traditional linters.

**Key Philosophy**: Only comment when there are genuine improvements to suggest. If everything looks good, stay silent.

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  PR Event or    │────▶│   Orchestrator   │────▶│ Validate PR     │
│  Manual Trigger │     │    Workflow      │     │ Number          │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                │                           │
                                ▼                           ▼
                    ┌───────────────────────────┐   ┌─────────────────┐
                    │  Parallel Review Jobs     │   │ Individual      │
                    │  - README Accuracy        │──▶│ Review Base     │
                    │  - Code Comments          │   │ Workflow        │
                    │  - Type Annotations       │   └─────────────────┘
                    │  - Einops Suggestions     │           │
                    └───────────────────────────┘           ▼
                                │                   ┌─────────────────┐
                                ▼                   │ Claude Analysis │
                    ┌───────────────────────────┐   │ (Conditional)   │
                    │  Consolidation Script     │   └─────────────────┘
                    │  - Download artifacts     │           │
                    │  - Merge suggestions      │           ▼
                    │  - Create GitHub review   │   ┌─────────────────┐
                    └───────────────────────────┘   │ Artifact Upload │
                                │                   │ (if issues)     │
                                ▼                   └─────────────────┘
                        ┌─────────────────┐
                        │ Unified GitHub  │
                        │ PR Review       │
                        └─────────────────┘
```

## System Components

### 1. Orchestrator Workflow (`claude-review-orchestrator.yml`)

The main entry point that:

- Validates PR context (fails fast if no valid PR number)
- Runs all review types in parallel
- Consolidates results into a single GitHub review
- Provides summary in GitHub Actions

### 2. Base Workflow (`claude-review-base.yml`)

Shared foundation for all review types:

- Handles PR checkout and file filtering
- Sets up optional Python environment
- Runs Claude analysis with specialized prompts
- Uploads artifacts only when issues are found

### 3. Individual Review Workflows

Four specialized review types, each a thin wrapper around the base:

- `claude-review-readme.yml` - Documentation accuracy
- `claude-review-comments.yml` - Comment cleanup
- `claude-review-einops.yml` - Tensor operation improvements
- `claude-review-types.yml` - Type annotation coverage

### 4. Consolidation Script (`claude_review.py`)

Single Python script that:

- Downloads all review artifacts
- Merges suggestions
- Creates unified GitHub review with inline comments
- Falls back to PR comment if review API fails

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

## How It Works

### 1. PR Validation

The orchestrator first validates that a PR number exists:

```yaml
validate-pr:
  steps:
    - name: Validate PR Number
      run: |
        PR_NUMBER="${{ inputs.pr_number || github.event.pull_request.number }}"
        if [ -z "$PR_NUMBER" ] || [ "$PR_NUMBER" = "0" ]; then
          echo "❌ Error: No valid PR number found"
          exit 1
        fi
```

### 2. Parallel Review Execution

All review types run simultaneously for efficiency:

- Each review only processes relevant files (based on file patterns)
- Reviews that find no issues create no artifacts
- Claude is instructed to respond with "No issues found" when appropriate

### 3. Artifact-Based Communication

Reviews communicate through artifacts:

- Only created when issues are found
- Contains `claude-review-analysis.json` with suggestions
- Artifacts are downloaded and consolidated by the Python script

### 4. Unified Review Creation

The consolidation script:

1. Downloads all artifacts from the workflow run
2. Merges suggestions from all review types
3. Creates a single GitHub review with inline comments
4. Groups feedback by review type in the summary

## JSON Output Format

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
  "tldr": ["Quick summary of changes"]
}
```

## Configuration

### Required Secrets

```yaml
ANTHROPIC_API_KEY # Claude API access key
```

### Required Python Dependencies

```txt
PyGithub>=2.1.1
requests>=2.31.0
```

### Base Workflow Inputs

```yaml
review_name: string # Display name for the review
review_type: string # Unique identifier for artifact naming
file_pattern: string # Regex for file filtering (default: ".*")
setup_python: boolean # Whether to setup Python (default: false)
tools: string # Comma-separated Claude tools
prompt: string # Review-specific instructions
pr_number: string # PR number to review
```

## Usage Examples

### Manual Trigger

```bash
# Using GitHub CLI
gh workflow run "Claude Review: Orchestrator" -f pr_number=123

# Using GitHub UI
# Navigate to Actions → "Claude Review: Orchestrator" → Run workflow → Enter PR number
```

### Direct Review Type Trigger

```bash
# Run only the type annotations review
gh workflow run "Claude Review: Types" -f pr_number=123
```

### Creating a Custom Review

1. Create workflow file: `.github/workflows/claude-review-security.yml`

```yaml
name: 'Claude Review: Security'
on:
  workflow_call:
    inputs:
      pr_number:
        required: true
        type: string

jobs:
  review:
    uses: ./.github/workflows/claude-review-base.yml
    with:
      review_name: 'Security Analysis'
      review_type: 'security'
      file_pattern: "\\.(py|js|ts)$"
      tools: 'Edit,Replace,Bash(git diff HEAD~1)'
      pr_number: ${{ inputs.pr_number }}
      prompt: |
        Review code for security vulnerabilities.

        **CRITICAL INSTRUCTIONS**:
        1. If you find NO security issues:
           - Simply respond with "No issues found."
           - DO NOT create any review
        2. ONLY create a review if you find actual vulnerabilities

        **Security issues to flag**:
        - SQL injection vulnerabilities
        - XSS possibilities
        - Hardcoded credentials
        - Insecure random number generation
        - Path traversal risks

        **What to ignore**:
        - Best practices that aren't vulnerabilities
        - Performance issues
        - Code style
```

2. Add to orchestrator's review list
3. Update the `review_types` array in `claude_review.py`

## Best Practices

### 1. Writing Review Prompts

**DO:**

- Start with clear "no issues found" instructions
- Define specific criteria for flagging issues
- Provide concrete examples of patterns to find
- Focus on actionable improvements
- Use severity levels appropriately

**DON'T:**

- Ask for general "improvements"
- Flag style preferences
- Suggest additions (only fixes)
- Create reviews for minor issues
- Be vague about what to look for

### 2. File Pattern Strategy

Use specific patterns to reduce analysis time:

```yaml
file_pattern: "\\.py$"           # Python files only
file_pattern: "\\.(js|ts)$"      # JavaScript and TypeScript
file_pattern: "^src/.*\\.py$"    # Python files in src/ directory
```

### 3. Tool Selection

**Common tool combinations:**

```yaml
# Basic code review
tools: "Edit,Replace,Bash(git diff HEAD~1)"

# With file discovery
tools: "Edit,Replace,Bash(git diff HEAD~1),Bash(find . -name '*.py')"

# With tool verification
tools: "Edit,Replace,Bash(git diff HEAD~1),Bash(python -m mypy --version)"
```

## Performance Considerations

1. **Parallel Execution**: All reviews run simultaneously
2. **File Filtering**: Use patterns to reduce scope
3. **Early Exit**: No artifact creation when no issues found
4. **Single API Call**: One GitHub review instead of multiple
5. **Conditional Steps**: Python setup only when needed

## Troubleshooting

### No Review Appearing

1. **Check PR validation**:

   - Ensure PR number is valid
   - Check orchestrator logs for validation errors

2. **Verify issues were found**:

   - Check individual review job logs
   - Look for "No issues found" responses
   - Verify artifacts were created

3. **Review consolidation script logs**:
   - Check for artifact download failures
   - Verify PR permissions
   - Look for API rate limit errors

### Suggestions Not Inline

Common causes:

- File not in PR diff
- Incorrect line numbers
- GitHub API validation failure

Debug by checking:

- Skipped suggestions in review body
- Script output for specific errors
- PR file list vs suggestion files

### Performance Issues

1. **Reduce file scope**: Use specific file patterns
2. **Check for large diffs**: Very large PRs may timeout
3. **Monitor API limits**: Both Claude and GitHub have rate limits

## System Limitations

1. **Artifact size**: Large PRs with many issues may hit limits
2. **API rate limits**: Both Claude and GitHub endpoints
3. **Concurrent reviews**: Maximum 4 parallel reviews currently
4. **File patterns**: Regex complexity affects performance

## Future Enhancements

Potential improvements:

1. **Incremental reviews**: Only analyze changed lines
2. **Custom severity thresholds**: Configure what triggers reviews
3. **Team preferences**: `.claude-review.yml` configuration files
4. **Review metrics**: Track suggestion acceptance rates
5. **Caching layer**: Reuse reviews for unchanged files
6. **More review types**: Security, performance, accessibility
7. **IDE integration**: Direct integration with VS Code/IntelliJ

## Maintenance

### Adding Review Types

1. Create workflow following the pattern
2. Add to orchestrator job list
3. Update `review_types` array in Python script
4. Test with manual dispatch
5. Document in this guide

### Updating Claude Model

Change in `claude-review-base.yml`:

```yaml
env:
  CLAUDE_MODEL: 'claude-sonnet-4-20250514'
```

### Monitoring Usage

Track in GitHub Actions:

- Review frequency per type
- Issues found vs PRs reviewed
- Time to complete reviews
- API usage patterns
