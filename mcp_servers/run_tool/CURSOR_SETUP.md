# Run Tool MCP Server - Cursor Setup

## Installation

1. Install the MCP server:
```bash
cd /path/to/metta
uv pip install -e mcp_servers/run_tool
```

2. Add to Cursor's MCP settings (`~/.cursor/mcp.json` or Cursor Settings → Features → Model Context Protocol):

```json
{
  "mcpServers": {
    "run-tool-mcp": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "/path/to/metta",
        "run-tool-mcp"
      ],
      "env": {
        "METTA_REPO_ROOT": "/path/to/metta"
      }
    }
  }
}
```

Replace `/path/to/metta` with the actual path to your Metta repository.

## Verification

After restarting Cursor, you should see the Run Tool MCP server in the MCP panel. You can test it by asking:

- "What recipes are available?"
- "What tools does the arena recipe have?"
- "How do I train a model?"

## Troubleshooting

- **Server not found**: Make sure `uv pip install -e mcp_servers/run_tool` completed successfully
- **Command not found**: Ensure the `run-tool-mcp` entry point is in your PATH or use full path
- **Repository not found**: Set `METTA_REPO_ROOT` environment variable to the correct path

