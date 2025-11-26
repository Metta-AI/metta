# Claude Desktop Setup

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

3. Add to `.claude/settings.json`:

   ```json
   {
     "mcpServers": {
       "observatory": {
         "command": "observatory-mcp",
         "env": {
           "METTA_MCP_BACKEND_URL": "https://api.observatory.softmax-research.net",
           "PYTHONPATH": "$(pwd)/mcp_servers/observatory",
           "LOG_LEVEL": "INFO"
         }
       }
     }
   }
   ```

4. Restart Claude Desktop and test with: `Use the observatory MCP server to get a list of training runs`
