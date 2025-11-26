# Cursor Setup

1. Install the server:

   ```bash
   uv pip install -e mcp_servers/observatory
   ```

2. Edit `~/.cursor/mcp.json`:

   ```json
   {
     "mcpServers": {
       "observatory": {
         "command": "uv",
         "args": ["run", "--project", "/path/to/metta", "python", "-m", "observatory_mcp.server"],
         "env": {
           "METTA_MCP_BACKEND_URL": "https://api.observatory.softmax-research.net",
           "PYTHONPATH": "/path/to/metta/mcp_servers/observatory",
           "LOG_LEVEL": "INFO"
         },
         "cwd": "/path/to/metta",
         "type": "stdio"
       }
     }
   }
   ```

   Replace `/path/to/metta` with your actual repository path.

3. To authenticate:

   ```bash
   uv run devops/observatory_login.py <backend_url> <backend_url>
   ```

4. Restart Cursor and test with: `Use the observatory MCP server to get a list of training runs`
