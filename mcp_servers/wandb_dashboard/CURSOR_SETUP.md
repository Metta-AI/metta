# Cursor MCP Integration Guide

## Quick Setup for Cursor

### 1. Verify Installation

First, make sure the MCP server is properly installed:

```bash
cd /home/morganm/workspace/metta/mcp_servers/wandb_dashboard
uv pip install -e .

# Test the server
wandb-mcp-server --help
```

### 2. Configure Cursor MCP

Cursor uses a `.cursor-mcp` directory in your project root to configure MCP servers. Let's set it up:

```bash
# From your metta project root
mkdir -p .cursor-mcp
```

Create the MCP configuration file:

```bash
cat > .cursor-mcp/config.json << 'EOF'
{
  "mcpServers": {
    "wandb-dashboard": {
      "command": "wandb-mcp-server",
      "args": [],
      "env": {
        "WANDB_API_KEY": "your_wandb_api_key_here"
      },
      "cwd": "/home/morganm/workspace/metta"
    }
  }
}
EOF
```

### 3. Set Environment Variables

Make sure your WandB credentials are available:

```bash
# Add to your shell profile (~/.bashrc, ~/.zshrc, etc.)
export WANDB_API_KEY="your_actual_api_key"
# Note: Entity and project are specified in natural language prompts, not env vars!
```

### 4. Restart Cursor

Close and restart Cursor to pick up the new MCP configuration.

### 5. Test the Integration

Once Cursor restarts, you should be able to use natural language commands like:

```
@wandb-dashboard Create a training dashboard for metta-ai/training-experiments showing loss and accuracy metrics
```

```
@wandb-dashboard List all available metrics in metta-research/model-evaluation project
```

```
@wandb-dashboard Add a scatter plot panel showing learning rate vs validation accuracy to metta-ai/hyperparameter-search
```

## Alternative Configuration

If the `.cursor-mcp` approach doesn't work, you can also try adding MCP configuration to your Cursor settings:

1. Open Cursor Settings (Cmd/Ctrl + ,)
2. Search for "MCP" or "Model Context Protocol"
3. Add the server configuration:

```json
{
  "mcp.servers": {
    "wandb-dashboard": {
      "command": "wandb-mcp-server",
      "env": {
        "WANDB_API_KEY": "your_api_key"
      }
    }
  }
}
```

## Troubleshooting

### Server Not Found

```bash
# Make sure the command is in your PATH
which wandb-mcp-server
# Should show: /home/morganm/workspace/metta/.venv/bin/wandb-mcp-server
```

### Authentication Issues

```bash
# Check WandB status
wandb status
# Should show: Logged in as your-username
```

### Test Server Manually

```bash
# Test the server starts correctly
timeout 3 wandb-mcp-server || echo "Server working (timed out as expected)"
```

## Available Commands

Once configured, you can use these natural language commands:

- **Create Dashboard**: "Create a dashboard for metta-ai/training-runs"
- **Add Panels**: "Add a line plot showing loss over time to metta-research/experiments"
- **List Metrics**: "What metrics are available in metta-ai/hyperparameter-search project?"
- **Clone Dashboard**: "Copy dashboard from metta-ai/model-a to metta-ai/model-b"
- **Update Dashboard**: "Modify dashboard in metta-research/evaluation to include accuracy metrics"

## Debugging

Enable verbose logging by setting:

```bash
export LOG_LEVEL=DEBUG
```

Then restart the MCP server to see detailed logs.
