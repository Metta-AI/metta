# Workflow Documentation

This directory contains detailed documentation for our GitHub Actions workflows. Each document provides comprehensive
information about specific workflows, their configuration, and usage.

## Available Documentation

### 📊 [PR Summary Workflow](./pr-summary.md)

Comprehensive documentation for the automated PR summary system that generates weekly summaries of merged pull requests
and posts them to Discord.

**Key Topics:**

- Two-phase AI summary generation using Google Gemini models
- Intelligent caching strategy to minimize API calls
- Custom GitHub Actions for PR fetching and Discord posting
- Performance optimizations and error handling
- Configuration options and usage examples

**Components Covered:**

- Main workflow (`pr_summary.yml`)
- PR Digest Action (`pr-digest/`)
- Summary Generation Script (`generate_pr_summary.py`)
- Discord Webhook Action (`discord-webhook/`)

### 🔍 [Claude Review System](./claude-review-system.md)

Comprehensive documentation for our AI-powered code review system that provides targeted, actionable feedback on pull
requests.

**Key Topics:**

- Four specialized review types (README, Comments, Einops, Type Annotations)
- "Silent on no issues" philosophy
- Two-stage review process with conditional output
- Creating custom review types
- Best practices and troubleshooting

**Components Covered:**

- Base workflow system (`claude-review-base.yml`)
- Individual review workflows
- JSON output format and GitHub review creation
- Configuration and customization options

### 🤖 [Claude Assistant](./claude-assistant.md)

Documentation for the interactive Claude bot that responds to mentions in comments and can automatically create pull
requests.

**Key Topics:**

- Two modes: comment responses and PR creation
- Natural language understanding for code changes
- Smart branch targeting for iterative development
- MCP tools for GitHub operations
- Error handling and debugging

**Components Covered:**

- Main workflow (`claude.yml`)
- Comment detection and routing
- Branch creation and management
- PR creation with detailed descriptions
- Failure recovery and diagnostics

## Workflow Categories

### 🤖 AI-Powered Workflows

- **[PR Summary](./pr-summary.md)**: Automated weekly summaries using Gemini AI
- **[Claude Reviews](./claude-review-system.md)**: Targeted code review system using Claude AI
- **[Claude Assistant](./claude-assistant.md)**: Interactive bot for questions and automated PR creation

### 💬 Interactive Workflows

- **[Claude Assistant](./claude-assistant.md)**: Responds to @claude mentions for help and PR creation

### 📝 Documentation Workflows

- **[PR Summary](./pr-summary.md)**: Generates comprehensive PR summaries
- **[Claude README Review](./claude-review-system.md#1-📝-readme-accuracy-review)**: Ensures documentation accuracy

### 🔍 Code Quality Workflows

- **[Claude Type Annotations](./claude-review-system.md#4-🏷️-type-annotations-review)**: Python type coverage
- **[Claude Comments Review](./claude-review-system.md#2-💬-code-comments-review)**: Comment cleanup
- **[Claude Einops](./claude-review-system.md#3-🔄-einops-suggestions-review)**: Tensor operation improvements

## Contributing

When adding new workflow documentation:

1. **File Naming**: Use kebab-case (e.g., `workflow-name.md`)
2. **Structure**: Include sections for:

   - Overview
   - Architecture diagram (if applicable)
   - Components breakdown
   - Configuration requirements
   - Usage examples
   - Error handling
   - Performance considerations

3. **Update this README**: Add your new documentation to the appropriate section

## Quick Links

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Workflow Syntax Reference](https://docs.github.com/en/actions/reference/workflow-syntax-for-github-actions)
- [Custom Actions Guide](https://docs.github.com/en/actions/creating-actions)

## Documentation Standards

All workflow documentation should include:

- **Clear Overview**: What the workflow does and why
- **Architecture**: Visual or textual representation of components
- **Configuration**: Required secrets, environment variables, inputs
- **Examples**: Real-world usage scenarios
- **Troubleshooting**: Common issues and solutions
- **Performance**: Optimization strategies and caching details
