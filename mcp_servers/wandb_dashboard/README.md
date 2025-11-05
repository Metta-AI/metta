# WandB Dashboard MCP Server

Model Context Protocol (MCP) server that enables Large Language Models to create, modify, and manage Weights & Biases
dashboards through natural language commands.

## Features

- **Dashboard Management**: Create, update, and clone WandB workspace dashboards
- **Panel Configuration**: Add various panel types (line plots, bar plots, scalar charts, scatter plots)
- **Metrics Discovery**: List available metrics from WandB projects
- **Template Support**: Pre-built dashboard templates for common use cases
- **Natural Language Interface**: LLMs can manage dashboards through conversational commands

## Installation

1. Install dependencies:

```bash
uv pip install -e .
```

2. Set up WandB authentication:

```bash
export WANDB_API_KEY="your_wandb_api_key"
export WANDB_ENTITY="your_wandb_entity"  # optional
export WANDB_PROJECT="your_default_project"  # optional
```

## Usage

### Starting the MCP Server

```bash
wandb-mcp-server
```

Or programmatically:

```python
import asyncio
from wandb_dashboard.server import main

asyncio.run(main())
```

### Integration with Claude Desktop

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "wandb-dashboard": {
      "command": "wandb-mcp-server"
    }
  }
}
```

## Available Tools

### `create_dashboard`

Create a new WandB workspace dashboard.

**Parameters:**

- `name` (string): Dashboard name
- `entity` (string): WandB entity (user/team)
- `project` (string): WandB project
- `description` (string, optional): Dashboard description
- `sections` (array, optional): Dashboard sections configuration

**Example:**

```json
{
  "name": "Training Overview",
  "entity": "my-team",
  "project": "my-project",
  "sections": [
    {
      "name": "Loss Metrics",
      "panels": [{ "type": "line_plot", "config": { "x": "Step", "y": ["loss", "val_loss"] } }]
    }
  ]
}
```

### `update_dashboard`

Update an existing dashboard.

**Parameters:**

- `dashboard_url` (string): URL of the dashboard to update
- `modifications` (object): Changes to apply

### `list_dashboards`

List available dashboards for an entity/project.

**Parameters:**

- `entity` (string): WandB entity
- `project` (string, optional): WandB project
- `filters` (object, optional): Additional filters

### `add_panel`

Add a panel to an existing dashboard section.

**Parameters:**

- `dashboard_url` (string): Dashboard URL
- `section_name` (string): Target section name
- `panel_type` (string): Panel type (`line_plot`, `bar_plot`, `scalar_chart`, `scatter_plot`)
- `panel_config` (object): Panel configuration

### `list_available_metrics`

List available metrics for a project.

**Parameters:**

- `entity` (string): WandB entity
- `project` (string): WandB project
- `run_filters` (object, optional): Filters for runs

### `get_dashboard_config`

Get the configuration of an existing dashboard.

**Parameters:**

- `dashboard_url` (string): Dashboard URL

### `clone_dashboard`

Clone an existing dashboard.

**Parameters:**

- `source_url` (string): Source dashboard URL
- `new_name` (string): Name for the cloned dashboard

## Panel Types

### Line Plot

Displays metrics as line charts over time.

```json
{ "type": "line_plot", "config": { "x": "Step", "y": ["loss", "accuracy"] } }
```

### Bar Plot

Shows metrics as bar charts.

```json
{ "type": "bar_plot", "config": { "metrics": ["accuracy", "f1_score"] } }
```

### Scalar Chart

Displays aggregated scalar metrics.

```json
{ "type": "scalar_chart", "config": { "metric": "val_accuracy", "groupby_aggfunc": "max" } }
```

### Scatter Plot

Creates scatter plots for metric correlation analysis.

```json
{ "type": "scatter_plot", "config": { "x": "learning_rate", "y": "val_accuracy" } }
```

## Dashboard Templates

The server includes pre-built templates for common dashboard types:

- **`training_overview`**: Loss and performance metrics
- **`hyperparameter_analysis`**: Learning rate and hyperparameter analysis
- **`model_comparison`**: Validation metrics for comparing models

Templates can be used when creating dashboards to quickly set up common visualization patterns.

## Configuration

Configuration is handled through environment variables and the `WandBMCPConfig` class:

- `WANDB_API_KEY`: WandB API key (required)
- `WANDB_ENTITY`: Default entity
- `WANDB_PROJECT`: Default project
- `WANDB_BASE_URL`: WandB API base URL (default: https://api.wandb.ai)
- `LOG_LEVEL`: Logging level (default: INFO)

## Example Conversations

### Creating a Training Dashboard

```
Human: Create a training dashboard for my "image-classifier" project showing loss and accuracy metrics.
```
