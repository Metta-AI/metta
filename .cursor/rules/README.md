# Cursor Rules Configuration (September 2025 Standards)

Modern Cursor IDE rule configuration for Metta AI using the latest 2025 numbered `.mdc` format with enhanced Agent mode capabilities.

## Rule Organization (2025 Standard)

### Numbered File Convention
- **001-099**: Core/workspace rules  
- **100-199**: Integration rules
- **200-299**: Pattern-specific rules
- Higher numbers take precedence for conflicting rules

### Current Configuration
- **001_workspace.mdc** (Always) - Project architecture and core workflows
- **002_rules.mdc** (Always) - AI assistant behavioral guidelines  
- **003_docs.mdc** (Auto) - Documentation and planning standards
- **004_tools.mdc** (Auto) - Tool and script usage patterns
- **101_python.mdc** (Auto: `*.py`) - Python coding standards and formatting
- **102_ml_rl.mdc** (Auto: `metta/**/*.py`) - ML/RL specific patterns
- **201_testing.mdc** (Auto: `tests/**/*.py`) - Testing guidelines and patterns  
- **202_frontend.mdc** (Auto: `**/*.ts`, `**/*.tsx`) - TypeScript/React standards

## 2025 Features Integration

### Enhanced Agent Mode (v0.46+)
- **Multi-step edits**: Autonomous execution across multiple files
- **Shell command execution**: With approval or "Yolo mode"
- **Improved codebase understanding**: Better context selection and token efficiency
- **GitHub integration**: Direct PR interaction with @Cursor tags

### Modern Rule Application
- **Auto Attached**: Rules activate based on glob patterns when editing specific files
- **Agent Requested**: Rules applied based on description relevance
- **Always**: Core rules that apply to all interactions

### Best Practices (Under 25 Lines Each)
- Concise/shorthand formatting
- File tagging with `@filename.ext`
- Specific, actionable directives
- Context-aware application at dialog start

## Migration Benefits

### From Legacy .cursorrules
- **Better Organization**: Domain-specific rules with numbered precedence
- **Token Efficiency**: Only relevant rules applied per context
- **Enhanced Metadata**: Descriptions, glob patterns, application control
- **Version Control**: Individual rule tracking and updates
- **Agent Compatibility**: Optimized for enhanced Agent mode capabilities

### Context-Aware Intelligence  
Rules now integrate with Cursor's improved:
- **Codebase Search**: Better ranking and indexing
- **File Reading**: Full file access without 2MB caps
- **Directory Exploration**: Complete tree traversal with metadata
- **Grep Matching**: Reduced noise and improved relevance

## Usage Patterns

### Automatic Application
- Python editing → `101_python.mdc` + `102_ml_rl.mdc` activate
- Testing → `201_testing.mdc` activates  
- Frontend work → `202_frontend.mdc` activates
- Always active → `001_workspace.mdc` + `002_rules.mdc`

### Manual Selection
Use Cursor's rule selection interface to apply specific rules when needed, or reference them with `/Generate Cursor Rules` command after chat conversations.

## Development Workflow

### New Rule Creation
1. Use Command Palette → "New Cursor Rule"
2. Follow NNN_name.mdc naming convention
3. Include appropriate metadata (description, globs, alwaysApply)
4. Keep under 25 lines with shorthand formatting

### Rule Testing
Best tested at start of new dialog sessions for optimal context-aware application.

This configuration leverages the latest September 2025 Cursor IDE capabilities for enhanced AI-assisted development on the Metta AI reinforcement learning project.