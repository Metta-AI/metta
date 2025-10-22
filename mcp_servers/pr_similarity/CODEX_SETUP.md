# Codex CLI Setup

1. Install the server from the repo root:

   ```bash
   uv pip install -e mcp_servers/pr_similarity
   ```

2. Ensure `GEMINI_API_KEY` is exported in the shell where you launch Codex:

   Optional cache override:

   ```bash
   export PR_SIMILARITY_CACHE_PATH="/absolute/path/to/mcp_servers/pr_similarity/cache/pr_embeddings"
   ```

3. `metta install` registers the server with Codex, so no manual configuration is needed. Open a new Codex
   session in this directory and verify with `codex mcp list`.

4. Ask Codex to call `find_similar_prs` with a short description. You should receive JSON output listing
   the top matching PRs from the cache.
