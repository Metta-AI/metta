"""MCP server for PR similarity lookups."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, cast

import mcp.types as types
from mcp.server import Server
from mcp.server.stdio import stdio_server

if TYPE_CHECKING:
    from metta.tools.pr_similarity import EmbeddingRecord as EmbeddingRecordType


def _find_repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / ".git").exists():
            return parent
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return current.parents[-1]


REPO_ROOT = _find_repo_root()


def _load_pr_similarity_module():
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    import metta.tools.pr_similarity as pr_similarity  # type: ignore[import-not-found]

    return pr_similarity


PR_SIMILARITY = _load_pr_similarity_module()
API_KEY_ENV = PR_SIMILARITY.API_KEY_ENV
DEFAULT_CACHE_PATH = PR_SIMILARITY.DEFAULT_CACHE_PATH
DEFAULT_TOP_K = PR_SIMILARITY.DEFAULT_TOP_K
find_similar_prs = PR_SIMILARITY.find_similar_prs
load_cache = PR_SIMILARITY.load_cache
require_api_key = PR_SIMILARITY.require_api_key

if TYPE_CHECKING:
    EmbeddingRecord = EmbeddingRecordType
else:
    EmbeddingRecord = PR_SIMILARITY.EmbeddingRecord

CACHE_ENV = "PR_SIMILARITY_CACHE_PATH"
SERVER_NAME = "metta-pr-similarity"
RESOURCE_URI = "pr-similarity://cache-info"


@dataclass
class ServerContext:
    cache_path: Path
    api_key: str | None
    default_model: str | None
    task_type: str


def determine_cache_path() -> Path:
    override = os.getenv(CACHE_ENV)
    if override:
        return Path(override).expanduser().resolve()
    return (REPO_ROOT / DEFAULT_CACHE_PATH).resolve()


def summarize_description(description: str, limit: int = 240) -> str:
    clean = " ".join(description.split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 1] + "…"


def build_context() -> ServerContext:
    cache_path = determine_cache_path()
    try:
        api_key = require_api_key(API_KEY_ENV)
    except EnvironmentError as error:
        logging.getLogger(__name__).warning("%s", error)
        api_key = None
    default_model: str | None = None
    task_type = PR_SIMILARITY.TASK_FALLBACK

    if cache_path.exists():
        try:
            metadata, records = load_cache(cache_path)
            del records
            default_model = metadata.model
            task_type = metadata.task_type
        except Exception as error:
            logging.getLogger(__name__).warning("Failed to read cache metadata: %s", error)
    else:
        raise FileNotFoundError(f"Embedding cache not found: {cache_path}")

    return ServerContext(
        cache_path=cache_path,
        api_key=api_key,
        default_model=default_model,
        task_type=task_type,
    )


def format_record(score: float, record: EmbeddingRecord, rank: int) -> Dict[str, Any]:
    return {
        "rank": rank,
        "score": round(score, 6),
        "pr_number": record.pr_number,
        "title": record.title,
        "author": record.author,
        "additions": record.additions,
        "deletions": record.deletions,
        "files_changed": record.files_changed,
        "commit_sha": record.commit_sha,
        "authored_at": record.authored_at,
        "summary": summarize_description(record.description),
    }


def build_server(context: ServerContext) -> Server:
    app = Server(SERVER_NAME)

    @app.list_tools()
    async def list_tools() -> List[types.Tool]:
        return [
            types.Tool(
                name="find_similar_prs",
                description="Return pull requests similar to a given query using text embedding similarity search."
                + "Helpful for finding relevant historical PRs when debugging new issues.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "description": {
                            "type": "string",
                            "description": "Bug or issue description to compare against historical pull requests.",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of similar pull requests to return.",
                            "default": DEFAULT_TOP_K,
                            "minimum": 1,
                            "maximum": 25,
                        },
                        "model": {
                            "type": "string",
                            "description": "Optional model override. Defaults to the model stored with the cache.",
                        },
                        "api_key": {
                            "type": "string",
                            "description": f"Optional Gemini API key override if {API_KEY_ENV} is unavailable.",
                        },
                    },
                    "required": ["description"],
                },
            ),
        ]

    @app.call_tool()
    async def call_tool(name: str, arguments: Dict[str, Any]) -> List[types.Content]:
        if name != "find_similar_prs":
            return [types.TextContent(type="text", text=f"Unknown tool: {name}")]

        description = str(arguments.get("description", "")).strip()
        if not description:
            return [
                types.TextContent(
                    type="text",
                    text="Error: 'description' must be provided and non-empty.",
                ),
            ]

        top_k_raw = arguments.get("top_k", DEFAULT_TOP_K)
        try:
            top_k = int(top_k_raw)
        except (TypeError, ValueError):
            return [
                types.TextContent(
                    type="text",
                    text="Error: 'top_k' must be an integer.",
                ),
            ]
        if top_k <= 0:
            return [
                types.TextContent(
                    type="text",
                    text="Error: 'top_k' must be greater than zero.",
                ),
            ]

        model_override = arguments.get("model")
        api_key_override = arguments.get("api_key")
        api_key_candidate = None
        if api_key_override is not None:
            api_key_candidate = str(api_key_override).strip()
        api_key = api_key_candidate or context.api_key or os.getenv(API_KEY_ENV)
        if not api_key:
            return [
                types.TextContent(
                    type="text",
                    text=(
                        f"Error: Provide an 'api_key' argument or set {API_KEY_ENV} before calling find_similar_prs."
                    ),
                ),
            ]
        try:
            metadata, results = find_similar_prs(
                description=description,
                top_k=top_k,
                cache_path=context.cache_path,
                model_override=model_override,
                api_key=api_key,
            )
        except Exception as error:  # pragma: no cover - surface error to caller
            logging.getLogger(__name__).exception("Failed to compute similar PRs")
            return [types.TextContent(type="text", text=f"Error: {error}")]

        payload = {
            "query": {
                "description": description,
                "top_k": top_k,
                "model": model_override or metadata.model,
            },
            "cache": {
                "path": str(context.cache_path),
                "task_type": metadata.task_type,
            },
            "results": [format_record(score, record, rank=index + 1) for index, (score, record) in enumerate(results)],
        }

        return [
            types.TextContent(
                type="text",
                text=json.dumps(payload, indent=2),
            ),
        ]

    @app.list_resources()
    async def list_resources() -> List[types.Resource]:
        return [
            types.Resource(
                uri=cast(Any, RESOURCE_URI),
                name="PR Similarity Cache Info",
                description=(
                    "Metadata about the PR embedding cache used for similarity search "
                    f"(default model: {context.default_model or 'unspecified'})."
                ),
                mimeType="application/json",
            ),
        ]

    @app.read_resource()
    async def read_resource(uri: str) -> str:
        if uri != RESOURCE_URI:
            raise ValueError(f"Unknown resource URI: {uri}")

        metadata, records = load_cache(context.cache_path)
        payload = {
            "cache_path": str(context.cache_path),
            "model": metadata.model,
            "task_type": metadata.task_type,
            "entries": len(records),
            "environment": {
                "api_key_env": API_KEY_ENV,
                "cache_path_env": CACHE_ENV,
            },
        }
        return json.dumps(payload, indent=2)

    return app


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger(__name__)
    logger.info("Starting PR similarity MCP server")

    context = build_context()
    logger.info(
        "PR similarity context ready (cache=%s, default_model=%s, task_type=%s)",
        context.cache_path,
        context.default_model,
        context.task_type,
    )
    server = build_server(context)

    async with stdio_server() as (read_stream, write_stream):
        logger.info("Entering MCP stdio server loop")
        await server.run(read_stream, write_stream, server.create_initialization_options())


def cli_main() -> None:
    asyncio.run(main())


__all__ = ["main", "cli_main"]


if __name__ == "__main__":
    cli_main()
