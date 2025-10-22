# PR Similarity MCP Server

Model Context Protocol (MCP) server that exposes the PR similarity search tool backed by Gemini embeddings.
It lets coding agents send a bug description and fetch the most relevant historical pull requests from the
`pr_embeddings.json` cache.

## Installation

```bash
uv pip install -e mcp_servers/pr_similarity
```

## Environment

| Variable | Purpose | Required | Default |
| --- | --- | --- | --- |
| `GEMINI_API_KEY` | API key used to call the Gemini embedding endpoint | ✅ | – |
| `PR_SIMILARITY_CACHE_PATH` | Override location for the embeddings cache | ❌ | `pr_embeddings.json` in the repo root |

The installer does not bundle credentials. Make sure `GEMINI_API_KEY` is exported before starting the server.

## Usage

Start the server (it speaks stdio MCP):

```bash
metta-pr-similarity-mcp
```

Once running, agents can invoke the `find_similar_prs` tool with:

```json
{
  "name": "find_similar_prs",
  "arguments": {
    "description": "network error when saving eval runs",
    "top_k": 3,
    "api_key": "AIza... (optional override)"
  }
}
```

The response is a JSON blob containing ranked PR metadata along with their similarity scores.

## Client integration

### Claude Desktop

The repository versioned `.claude/settings.json` already registers the server under
`metta-pr-similarity`. Restart Claude Desktop (or reload the project) and it will discover the
`metta-pr-similarity-mcp` command automatically.

### Codex CLI

`.codex/settings.local.json` runs `codex mcp add metta-pr-similarity -- metta-pr-similarity-mcp`
during session setup, so any Codex CLI that enters this repository will auto-register the server.
You can confirm with `codex mcp list` after opening a new Codex session.

Make sure the executable is on `PATH` (e.g. use the repo virtualenv). Once wired up, Codex and Claude can
query the same similarity data the original CLI script uses.
