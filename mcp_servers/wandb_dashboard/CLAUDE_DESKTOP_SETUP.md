# Claude Desktop Integration Guide

## Quick Setup

1. **Install the MCP Server:**

```bash
cd /path/to/metta/mcp_servers/wandb_dashboard
uv pip install -e .
```

2. **Set up WandB Authentication:**

```bash
export WANDB_API_KEY="your_wandb_api_key"
export WANDB_ENTITY="your_wandb_entity"    # optional
export WANDB_PROJECT="your_default_project" # optional
```

3. **Add to Claude Desktop Config:**

Edit your `claude_desktop_config.json` file (usually located at
`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "wandb-dashboard": {
      "command": "wandb-mcp-server",
      "env": {
        "WANDB_API_KEY": "your_wandb_api_key",
        "WANDB_ENTITY": "your_entity",
        "WANDB_PROJECT": "your_project"
      }
    }
  }
}
```

4. **Restart Claude Desktop**

## Usage Examples

Once set up, you can use natural language commands like:

### Create a Dashboard

```
Create a training dashboard for my "image-classification" project showing loss and accuracy metrics over time.
```

### Add Panels

```
Add a scatter plot panel to my existing dashboard showing the relationship between learning rate and validation accuracy.
```

### List Available Metrics

```
What metrics are available in my "neural-network" project?
```

### Clone a Dashboard

```
Clone my "training-overview" dashboard and call it "experiment-comparison".
```

## Troubleshooting

### Server Won't Start

- Check that wandb-mcp-server is in your PATH
- Verify WandB authentication: `wandb status`
- Check Claude Desktop logs for error messages

### Authentication Issues

- Run `wandb login` to re-authenticate
- Verify WANDB_API_KEY environment variable is set
- Check that your API key has the necessary permissions

### Dashboard Creation Fails

- Ensure you have write permissions to the specified WandB entity/project
- Verify that the entity and project exist in WandB
- Check that metric names exist in your project runs

## Available Tools

- `create_dashboard` - Create new dashboards
- `update_dashboard` - Modify existing dashboards
- `list_dashboards` - List available dashboards
- `add_panel` - Add panels to dashboards
- `list_available_metrics` - Show available metrics
- `get_dashboard_config` - Get dashboard configuration
- `clone_dashboard` - Clone existing dashboards

## Support

For issues or questions:

1. Check the [main README](README.md) for detailed documentation
2. Verify your WandB setup with `wandb status`
3. Test the server manually with `wandb-mcp-server`
