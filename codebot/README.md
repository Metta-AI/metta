# Codebot: AI-Powered Code Assistant Framework

## Overview

Codebot is a framework for building AI-powered code assistants that amplify human engineering capabilities. It provides a unified interface for executing code operations through different LLM interaction modes.

## Quick Start

```bash
# One-shot execution (default, fastest)
codebot test              # Write tests for changed code
codebot lint              # Fix linting issues
codebot review            # Review current changes

# Persistent mode (maintains conversation state)
codebot test -p           # Use claude -p for stateful interaction
codebot review -p         # Iterative review with context retention

# Interactive mode (launches Claude Code)
codebot test -i           # Interactive test writing
codebot review -i         # Interactive code review session

# Workflows (compose multiple commands)
codebot test-debug        # Run test → debug → test loop
codebot webapp            # Full web app development flow
```

## Core Concepts

### Commands
Single-purpose operations that perform specific tasks. Each command:
- Automatically gathers context (git diff, clipboard, relevant files)
- Can execute in three modes: one-shot (default), persistent (-p), or interactive (-i)
- Returns structured file changes

### Workflows
Compose multiple commands into multi-step processes with:
- Sequential and parallel execution
- Conditional branching based on results
- Interactive handoff points

### Context Management
Smart context assembly that:
- Prioritizes by relevance and recency
- Respects token limits
- Includes project structure (parent READMEs)
- Filters by file extensions when appropriate

## Architecture

### Execution Modes

1. **One-shot (default)**: Direct LLM API call for fastest performance
2. **Persistent (-p)**: Uses `claude -p` to maintain conversation state across commands
3. **Interactive (-i)**: Launches Claude Code for human-in-the-loop refinement

### Data Model

```python
@dataclass
class Command:
    name: str
    prompt_template: str
    default_paths: List[str]
    output_schema: OutputSchema

@dataclass
class ExecutionContext:
    git_diff: str
    clipboard: str
    relevant_files: List[FileContent]
    mode: Literal["oneshot", "persistent", "interactive"]

@dataclass
class CommandOutput:
    file_changes: List[FileChange]
    metadata: Dict[str, Any]
```

## Implementation Status

### Available Now
- Core command framework
- Context management system
- Git diff and clipboard integration
- Basic workflow composition

### In Development
- Claude Code MCP integration
- Remote execution subscriptions
- Advanced workflow patterns

### Future Considerations
- Performance optimizations
- Extended tool integrations
- Custom command plugins

## Design Principles

1. **Start Simple**: Focus on high-value commands before scaling
2. **Best Context**: Provide the most relevant information within token limits
3. **Clear Contracts**: Type-safe interfaces for reliable composition
4. **Human-Centric**: Amplify developer capabilities, don't replace them
5. **Flexible Execution**: Support different interaction modes for different needs

## Next Steps

See the [Design Overview](DESIGN.md) for detailed architecture and implementation guidance.