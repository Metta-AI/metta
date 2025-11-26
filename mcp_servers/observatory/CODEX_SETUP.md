# Codex CLI Setup

1. Install the server:

   ```bash
   uv pip install -e mcp_servers/observatory
   ```

2. Set environment variables (optional):

   ```bash
   export METTA_MCP_BACKEND_URL="https://api.observatory.softmax-research.net"
   ```

   To authenticate:

   ```bash
   uv run devops/observatory_login.py <backend_url> <backend_url>
   ```

3. Register globally:

   ```bash
   codex mcp add observatory \
     --env METTA_MCP_BACKEND_URL="https://api.observatory.softmax-research.net" \
     --env PYTHONPATH="$(pwd)/mcp_servers/observatory" \
     --env LOG_LEVEL="INFO" \
     $(pwd)/.venv/bin/observatory-mcp
   ```

   Or use project-level config in `.codex/settings.local.json`.

4. Test with: `Use the observatory MCP server to get a list of training runs`
