# Run Tool MCP Server - Codex CLI Setup

## Installation

1. Install the MCP server:

   ```bash
   cd /path/to/metta
   uv pip install -e mcp_servers/run_tool
   ```

2. Register globally with Codex CLI:

   ```bash
   codex mcp add run-tool-mcp \
     --env METTA_REPO_ROOT="$(pwd)" \
     --env LOG_LEVEL="INFO" \
     $(pwd)/.venv/bin/run-tool-mcp
   ```

   Replace `$(pwd)` with the actual path to your Metta repository if needed, or use the full path:

   ```bash
   codex mcp add run-tool-mcp \
     --env METTA_REPO_ROOT="/path/to/metta" \
     --env LOG_LEVEL="INFO" \
     /path/to/metta/.venv/bin/run-tool-mcp
   ```

   **Note**: If the command is not in `.venv/bin`, you can also use `uv run`:

   ```bash
   codex mcp add run-tool-mcp \
     --env METTA_REPO_ROOT="$(pwd)" \
     --env LOG_LEVEL="INFO" \
     uv run --project "$(pwd)" run-tool-mcp
   ```

3. Verify the server is registered:

   ```bash
   codex mcp list
   ```

   You should see `run-tool-mcp` in the list.

## Alternative: Project-Level Configuration

You can also configure it in `.codex/settings.local.json`:

```json
{
  "mcpServers": {
    "run-tool-mcp": {
      "command": "uv",
      "args": [
        "run",
        "--project",
        "/path/to/metta",
        "run-tool-mcp"
      ],
      "env": {
        "METTA_REPO_ROOT": "/path/to/metta",
        "LOG_LEVEL": "INFO"
      }
    }
  }
}
```

Replace `/path/to/metta` with your actual repository path.

## Verification

After setting up, open a new Codex session in the Metta repository and test by asking:

- "What recipes are available in the Metta codebase?"
- "What tools does the arena recipe have?"
- "How do I train a model using the arena recipe?"
- "What arguments does the train command accept?"

## Troubleshooting

- **Server not found**: Make sure `uv pip install -e mcp_servers/run_tool` completed successfully
- **Command not found**: Ensure the `run-tool-mcp` entry point is in your PATH or use the full path
- **Repository not found**: Set `METTA_REPO_ROOT` environment variable to the correct path
- **Check registration**: Run `codex mcp list` to see if the server is registered
- **Remove and re-add**: If having issues, try `codex mcp remove run-tool-mcp` then re-add it


