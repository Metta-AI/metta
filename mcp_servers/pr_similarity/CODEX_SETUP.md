# Codex CLI Setup

1. Install the server from the repo root:

   ```bash
   uv pip install -e mcp_servers/pr_similarity
   ```

2. Make sure your Codex shell session exports the Gemini API key:

   ```bash
   export GEMINI_API_KEY="your_api_key"
   ```

   Optional override:

   ```bash
   export PR_SIMILARITY_CACHE_PATH="/absolute/path/to/pr_embeddings.json"
   ```

3. The repo’s `.codex/settings.local.json` already runs
   `codex mcp add metta-pr-similarity -- metta-pr-similarity-mcp` during Codex session setup, so no manual
   configuration is needed. Open a new Codex session in this directory and verify with `codex mcp list`.

4. Ask Codex to call `find_similar_prs` with a short description. You should receive JSON output listing
   the top matching PRs from the cache.
