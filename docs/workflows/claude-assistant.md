# Claude Assistant Workflow Documentation

## Overview

The Claude Assistant is an AI-powered GitHub bot that responds to mentions in comments and can automatically create pull
requests based on natural language instructions. It leverages Anthropic's Claude AI to understand requests, analyze
code, and implement changes.

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ GitHub Comment  │────▶│ Mention Detection│────▶│ Action Router   │
│   "@claude"     │     │ (@claude check)  │     │ (comment/PR)    │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                           │
                              ┌────────────────────────────┴────────────────────────────┐
                              │                                                         │
                              ▼                                                         ▼
                    ┌─────────────────┐                                       ┌─────────────────┐
                    │ Comment Response│                                       │ PR Creation     │
                    │ (Analysis/Help) │                                       │(@claude open-pr)│
                    └─────────────────┘                                       └─────────────────┘
                              │                                                         │
                              ▼                                                         ▼
                    ┌─────────────────┐                                       ┌─────────────────┐
                    │ Post Reply      │                                       │ Create Branch   │
                    │ Comment         │                                       │ Make Changes    │
                    └─────────────────┘                                       │ Commit & Push   │
                                                                              │ Open PR         │
                                                                              └─────────────────┘
```

## Features

### 1. 💬 Comment Response Mode

Responds to general questions, provides code analysis, and offers suggestions.

**Triggers**:

- Any comment containing `@claude` (without `open-pr`)
- Works on issues and pull requests
- Responds to PR review comments

**Capabilities**:

- Code review and analysis
- Explanations and documentation
- Architecture suggestions
- Debugging help
- Running linters (ruff for Python)

**Available Tools**:

- `Bash`: Git commands, Python scripts, linters
- `View`: Read file contents
- `GlobTool`: Find files by pattern
- `GrepTool`: Search file contents

### 2. 🚀 PR Creation Mode

Automatically implements requested changes and creates a pull request.

**Triggers**:

- Comments containing `@claude open-pr`
- Works on both issues and existing PRs

**Capabilities**:

- Implements code changes based on natural language
- Creates feature branches automatically
- Commits changes with descriptive messages
- Opens PRs with detailed descriptions
- Targets appropriate branches (not always main)

**Smart Branch Targeting**:

- **From Issue**: Creates PR targeting the default branch (usually main)
- **From PR**: Creates PR targeting the PR's feature branch (iterative development)

**Available Tools**:

- All comment mode tools plus:
- `Edit`: Modify existing files
- `Replace`: Replace code sections
- `Write`: Create new files
- `mcp__github_file_ops__commit_files`: Commit changes
- `mcp__github__update_issue_comment`: Update progress

## Usage Examples

### Basic Comment Response

```markdown
@claude Can you explain what this function does and suggest improvements?
```

Claude will analyze the context and provide explanations.

### Code Review Request

```markdown
@claude Please review the error handling in this PR. Are there any edge cases we're missing?
```

### Simple PR Creation

```markdown
@claude open-pr Add type hints to all functions in utils.py
```

### Complex PR Creation

```markdown
@claude open-pr Refactor the authentication module to:

1. Use async/await instead of callbacks
2. Add proper error handling for network failures
3. Include retry logic with exponential backoff
4. Update the tests to match
```

### Iterative Development on PR

When commenting on an existing PR:

```markdown
@claude open-pr Based on the review feedback, please:

- Extract the validation logic into a separate function
- Add unit tests for the edge cases mentioned
- Update the docstrings
```

This creates a PR that merges into the current PR's branch, not main.

## Configuration

### Required Secrets

```yaml
ANTHROPIC_API_KEY  # Claude API access key
GITHUB_TOKEN       # Usually provided automatically
```

### Environment Variables

```yaml
CLAUDE_MODEL: 'claude-sonnet-4-20250514' # Model version
```

### Permissions Required

```yaml
permissions:
  contents: write # For creating branches and commits
  pull-requests: write # For creating PRs and comments
  issues: write # For commenting on issues
  id-token: write # For authentication
```

## Workflow Details

### 1. Detection Phase

The workflow detects the type of request:

- Searches for `@claude open-pr` to determine PR creation mode
- Falls back to comment mode for general `@claude` mentions

### 2. Comment Response Flow

1. **Checkout**: Shallow clone (depth 1) for performance
2. **Claude Analysis**: Runs with read-only tools
3. **Response**: Posts findings as a comment

### 3. PR Creation Flow

1. **Checkout**: Full clone (depth 0) for complete history
2. **Branch Setup**:
   - Determines target branch based on context
   - Creates unique branch name with timestamp
   - Checks out new branch
3. **Claude Implementation**:
   - Makes requested changes using editing tools
   - Commits using MCP GitHub tools
4. **Push & PR**:
   - Pushes branch to remote
   - Creates PR with descriptive title and body
   - Posts success/failure comment

### Branch Naming Convention

```
claude/auto-{issue_number}-{timestamp}
```

Example: `claude/auto-123-2024-01-15T10-30-45`

### PR Description Format

```markdown
🤖 **Automated PR created by Claude**

**Original request:**

> [User's @claude open-pr comment]

**Context:** This PR addresses the request from [issue/PR #X] **Target:** This PR will merge into `branch-name` (not
main)

**Changes made:**

- X commit(s) with: [commit message]

**Branch flow:** `claude/auto-X-timestamp` → `target-branch`

---

_This PR was automatically created by Claude Code Assistant._
```

## Error Handling

### Common Issues and Solutions

1. **No commits created**

   - Claude may have failed to use the commit tool
   - Solution: Ensure request is clear and specific

2. **Branch push failed**

   - Permissions or conflict issues
   - Solution: Check GitHub token permissions

3. **PR creation failed**
   - Target branch may be protected
   - Solution: Review branch protection rules

### Debug Information

The workflow provides extensive debugging:

- Git state before and after Claude runs
- Branch information and commit counts
- Success/failure messages with context

### Failure Comments

When PR creation fails, Claude posts a diagnostic comment:

```markdown
⚠️ **Unable to create PR**

**Reason:** No changes were committed

**Debug info:**

- Expected branch: `claude/auto-123-...`
- Target branch: `main`
- Has commits: false
- Claude execution: success

**Possible solutions:**

- Try a simpler, more specific request
- Check if the changes conflict with existing code
- Ensure Claude used mcp**github_file_ops**commit_files
```

## Best Practices

### 1. Writing Requests

**DO:**

- Be specific about what changes you want
- Break complex requests into numbered steps
- Mention specific files when possible
- Provide context about the goal

**DON'T:**

- Make vague requests like "improve this code"
- Ask for massive refactors in one go
- Request changes to protected files
- Include sensitive information

### 2. Iterative Development

- Use PR-to-PR workflow for incremental changes
- Review Claude's PRs before merging
- Can chain multiple Claude PRs for complex features

### 3. Code Quality

- Claude follows project standards (CLAUDE.md)
- Reviews can be requested before PR creation
- Automated tests still run on Claude's PRs

## Security Considerations

1. **Access Control**: Only users with comment permissions can trigger
2. **Code Review**: All PRs go through normal review process
3. **No Direct Commits**: Always creates PRs, never pushes to main
4. **Audit Trail**: All actions are logged in GitHub

## Limitations

1. **Context Window**: Very large codebases may exceed limits
2. **Complex Logic**: May struggle with highly complex refactors
3. **External Dependencies**: Cannot install new packages
4. **Binary Files**: Cannot modify images, compiled files, etc.

## Advanced Usage

### Custom Instructions

The workflow accepts custom instructions through the `CLAUDE.md` file:

- Coding standards
- Project-specific patterns
- Forbidden practices
- Preferred libraries

### Tool Restrictions

For security, certain bash commands are limited:

- Network operations restricted
- No system modifications
- Read-only access to most areas

### Timeout Configuration

- Comment responses: 30 minutes
- PR creation: 45 minutes
- Adjustable via workflow modification

## Future Enhancements

Potential improvements being considered:

1. **Batch Operations**: Multiple file changes in one request
2. **Test Generation**: Automatic test creation for changes
3. **Dependency Updates**: Handling package.json, uv.lock
4. **Cross-PR Context**: Understanding related PRs
5. **Approval Workflow**: Require approval before PR creation
