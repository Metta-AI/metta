# Claude Desktop Setup

Steps to expose the PR similarity MCP server to Claude Desktop.

1. Install the server (from the repo root):

   ```bash
   uv pip install -e mcp_servers/pr_similarity
   ```

2. Ensure credentials exist:

   ```bash
   export GEMINI_API_KEY="your_api_key"
   ```

   Optionally override the cache location:

   ```bash
   export PR_SIMILARITY_CACHE_PATH="/absolute/path/to/pr_embeddings.json"
   ```

3. The versioned `.claude/settings.json` already registers `metta-pr-similarity`, so no extra configuration
   is required. Simply restart Claude Desktop so it reloads the project settings.

4. You should see `metta-pr-similarity` listed in the MCP servers panel. Use the `find_similar_prs`
   tool with a short bug description to sanity check the connection.
