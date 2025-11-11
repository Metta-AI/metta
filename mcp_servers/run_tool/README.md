# Run Tool MCP Server

Model Context Protocol server for executing and discovering Metta `run.py` commands.

## Overview

This MCP server provides AI assistants (Claude Desktop, Cursor, etc.) with the ability to:
- **Discover** available recipes and tools
- **Execute** training, evaluation, play, and replay commands
- **Validate** commands before execution
- **Get help** on tool arguments and configurations

## Installation

```bash
uv pip install -e mcp_servers/run_tool
```

## Configuration

The server automatically detects the Metta repository root. You can override this with:

```bash
export METTA_REPO_ROOT="/path/to/metta"
export METTA_RUN_TOOL_TIMEOUT=3600  # Timeout in seconds (default: 3600)
export LOG_LEVEL="INFO"  # Logging level
```

## Available Tools

### Discovery Tools

- **`list_recipes`** - List all available recipes with their tools
- **`list_tools_in_recipe`** - List tools in a specific recipe
- **`list_recipes_for_tool`** - Find recipes that support a tool type (e.g., 'train', 'evaluate')
- **`get_tool_arguments`** - Get available arguments for a tool with types and defaults
- **`validate_command`** - Validate a command without executing it

### Execution Tools

- **`run_tool`** - Generic tool for executing any run.py command
- **`train`** - Convenience wrapper for training commands
- **`evaluate`** - Convenience wrapper for evaluation commands

## Usage Examples

### Discover Available Recipes

```python
# List all recipes
await mcp_client.call_tool("list_recipes", {})

# List tools in a recipe
await mcp_client.call_tool("list_tools_in_recipe", {"recipe": "arena"})

# Find recipes that support training
await mcp_client.call_tool("list_recipes_for_tool", {"tool_type": "train"})
```

### Get Tool Information

```python
# Get arguments for a tool
await mcp_client.call_tool("get_tool_arguments", {
    "tool_path": "train arena"
})

# Validate a command
await mcp_client.call_tool("validate_command", {
    "tool_path": "train arena",
    "arguments": {"run": "my_experiment"}
})
```

### Execute Commands

```python
# Train a model
await mcp_client.call_tool("train", {
    "recipe": "arena",
    "arguments": {
        "run": "my_experiment",
        "trainer.total_timesteps": 1000000
    }
})

# Evaluate a policy
await mcp_client.call_tool("evaluate", {
    "recipe": "arena",
    "arguments": {
        "policy_uris": "file://./train_dir/my_run/checkpoints/my_run:v12.pt"
    }
})

# Generic execution
await mcp_client.call_tool("run_tool", {
    "tool_path": "play arena",
    "arguments": {
        "policy_uri": "file://./checkpoints/policy.pt"
    }
})
```

## How It Works

1. **Discovery**: Uses `metta.common.tool.recipe_registry` to discover recipes and tools
2. **Execution**: Executes tools programmatically using `metta.common.tool.run_tool` utilities
3. **Validation**: Validates tool paths and arguments before execution
4. **Error Handling**: Provides helpful error messages and suggestions

See `CURSOR_SETUP.md` or `CLAUDE_DESKTOP_SETUP.md` for client-specific setup instructions.


