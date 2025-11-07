# Observatory MCP Server

Model Context Protocol server for accessing Metta Observatory backend functionality.

## Installation

```bash
uv pip install -e mcp_servers/observatory
```

## Configuration

Set environment variables (optional, defaults provided):

```bash
export METTA_MCP_BACKEND_URL="https://api.observatory.softmax-research.net"
export METTA_MCP_MACHINE_TOKEN="your_token"  # Optional, auto-loaded from ~/.metta/config.yaml
export LOG_LEVEL="INFO"
```

To authenticate, run:

```bash
uv run devops/observatory_login.py <backend_url> <backend_url>
```

## Available Tools

- `get_training_runs` - List all training runs
- `get_policies` - Get all policies and training runs
- `search_policies` - Search policies with filters
- `get_eval_names` - Get available evaluation names
- `get_available_metrics` - Get available metrics
- `generate_scorecard` - Generate scorecard data
- `run_sql_query` - Execute SQL queries
- `generate_ai_query` - Generate SQL from natural language

See `CODEX_SETUP.md`, `CLAUDE_DESKTOP_SETUP.md`, or `CURSOR_SETUP.md` for client-specific setup.

