# Run Tool MCP Server - Claude Desktop Setup

## Installation

1. Install the MCP server:

```bash
cd /path/to/metta
uv pip install -e mcp_servers/run_tool
```

2. Add to Claude Desktop's MCP settings (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "run-tool-mcp": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/metta", "run-tool-mcp"],
      "env": {
        "METTA_REPO_ROOT": "/path/to/metta"
      }
    }
  }
}
```

Replace `/path/to/metta` with the actual path to your Metta repository.

**Note**: On macOS, the config file location is:

- `~/Library/Application Support/Claude/claude_desktop_config.json`

On Windows:

- `%APPDATA%\Claude\claude_desktop_config.json`

On Linux:

- `~/.config/Claude/claude_desktop_config.json`

## Restart Claude Desktop

After updating the config file, restart Claude Desktop for the changes to take effect.

## Verification

You can verify the server is working by asking Claude:

- "What recipes are available in the Metta codebase?"
- "Show me how to train a model using the arena recipe"
- "What arguments does the train command accept?"

## Troubleshooting

- **Server not found**: Make sure `uv pip install -e mcp_servers/run_tool` completed successfully
- **Command not found**: Ensure `uv` is in your PATH, or use the full path to `uv`
- **Repository not found**: Set `METTA_REPO_ROOT` environment variable to the correct path
- **Permission errors**: Make sure the repository path is accessible and the run.py script is executable
