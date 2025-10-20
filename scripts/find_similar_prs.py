from __future__ import annotations

import argparse
import json
import math
import os
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from google import genai
from google.genai import types

API_KEY_ENV = "GEMINI_API_KEY"
DEFAULT_CACHE_PATH = Path("pr_embeddings.json")
DEFAULT_RESULTS = 10
TASK_FALLBACK = "semantic_similarity"


@dataclass
class EmbeddingRecord:
    pr_number: int
    title: str
    description: str
    author: str
    additions: int
    deletions: int
    files_changed: int
    commit_sha: str
    authored_at: str
    vector: List[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Embed an issue description and return the most similar PRs based on cached embeddings."),
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=DEFAULT_CACHE_PATH,
        help=f"Path to the PR embedding cache (default: {DEFAULT_CACHE_PATH}).",
    )
    parser.add_argument(
        "--model",
        help="Embedding model identifier (default: inferred from cache metadata).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_RESULTS,
        help=f"Number of similar PRs to return (default: {DEFAULT_RESULTS}).",
    )
    parser.add_argument(
        "description",
        help="Issue description to embed and compare against cached PR embeddings.",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI color codes in the output.",
    )
    return parser.parse_args()


def load_cache(path: Path) -> Tuple[Dict[str, str], List[EmbeddingRecord]]:
    if not path.exists():
        raise FileNotFoundError(f"Embedding cache not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    metadata = {
        "model": data.get("model"),
        "task_type": data.get("task_type", TASK_FALLBACK),
    }

    records: List[EmbeddingRecord] = []
    for item in data.get("entries", []):
        records.append(
            EmbeddingRecord(
                pr_number=int(item["pr_number"]),
                title=item.get("title", ""),
                description=item.get("description", ""),
                author=item.get("author", ""),
                additions=int(item.get("additions", 0)),
                deletions=int(item.get("deletions", 0)),
                files_changed=int(item.get("files_changed", 0)),
                commit_sha=item.get("commit_sha", ""),
                authored_at=item.get("authored_at", ""),
                vector=list(item["vector"]),
            ),
        )
    if not records:
        raise ValueError(f"No embedding entries found in cache {path}")
    return metadata, records


def read_description(args: argparse.Namespace) -> str:
    return args.description.strip()


def embed_text(
    client: genai.Client,
    text: str,
    model: str,
    task_type: str,
) -> List[float]:
    config = types.EmbedContentConfig(task_type=task_type)
    response = client.models.embed_content(
        model=model,
        contents=[text],
        config=config,
    )
    return list(response.embeddings[0].values)


def cosine_similarity(vector_a: List[float], vector_b: List[float]) -> float:
    dot = math.fsum(a * b for a, b in zip(vector_a, vector_b, strict=False))
    norm_a = math.sqrt(math.fsum(a * a for a in vector_a))
    norm_b = math.sqrt(math.fsum(b * b for b in vector_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def summarize_description(description: str, limit: int = 240) -> str:
    clean = " ".join(description.split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 1] + "…"


def supports_color(args: argparse.Namespace) -> bool:
    if args.no_color:
        return False
    return sys.stdout.isatty()


def colorize(enabled: bool, code: str, text: str) -> str:
    if not enabled:
        return text
    RESET = "\033[0m"
    return f"{code}{text}{RESET}"


def format_description_block(text: str, width: int = 90, indent: int = 12, label: str = "Summary") -> str:
    wrapper = textwrap.TextWrapper(width=width, subsequent_indent=" " * indent)
    wrapped = wrapper.fill(text)
    if not wrapped:
        return ""
    lines = wrapped.splitlines()
    if lines:
        lines[0] = f"{label}: {lines[0]}"
    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    api_key = os.getenv(API_KEY_ENV)
    if not api_key:
        raise EnvironmentError(f"Set {API_KEY_ENV} before running this script.")

    metadata, records = load_cache(args.cache_path)
    model = args.model or metadata.get("model")
    if not model:
        raise ValueError("Embedding model is not specified in cache; use --model.")
    task_type = metadata.get("task_type", TASK_FALLBACK)

    description = read_description(args)

    client = genai.Client(api_key=api_key)
    query_vector = embed_text(client, description, model=model, task_type=task_type)

    scored: List[Tuple[float, EmbeddingRecord]] = []
    for record in records:
        score = cosine_similarity(query_vector, record.vector)
        scored.append((score, record))

    scored.sort(key=lambda item: item[0], reverse=True)
    top_results = scored[: max(1, args.top_k)]

    use_color = supports_color(args)
    BOLD = "\033[1m"
    CYAN = "\033[36m"
    YELLOW = "\033[33m"
    MAGENTA = "\033[35m"
    GREEN = "\033[32m"
    BLUE = "\033[34m"

    header = colorize(use_color, BOLD + CYAN, f"Top {len(top_results)} similar PRs")
    print(f"{header} {colorize(use_color, CYAN, f'(model={model})')}")
    print(colorize(use_color, CYAN, "─" * 72))

    for rank, (score, record) in enumerate(top_results, start=1):
        summary = summarize_description(record.description)
        summary_block = format_description_block(summary) if summary else ""

        title_line = f"{rank}. PR #{record.pr_number} — {record.title}"
        score_line = f"score={score:.4f}"
        print(colorize(use_color, BOLD + YELLOW, title_line))
        print(colorize(use_color, GREEN, f"    {score_line}"))

        meta_line = (
            f"    Author: {record.author} · Additions: {record.additions} · "
            f"Deletions: {record.deletions} · Files changed: {record.files_changed}"
        )
        print(colorize(use_color, MAGENTA, meta_line))

        if summary_block:
            for line in summary_block.splitlines():
                print(colorize(use_color, BLUE, f"    {line}"))

        commit_line = f"    Commit: {record.commit_sha} · Authored at: {record.authored_at}"
        print(colorize(use_color, CYAN, commit_line))
        print(colorize(use_color, CYAN, "-" * 72))


if __name__ == "__main__":
    main()
