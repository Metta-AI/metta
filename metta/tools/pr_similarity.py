from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple, cast

from google import genai
from google.genai import types

API_KEY_ENV = "GEMINI_API_KEY"
DEFAULT_CACHE_PATH = Path("pr_embeddings.json")
DEFAULT_TOP_K = 10
TASK_FALLBACK = "semantic_similarity"


@dataclass(frozen=True)
class CacheMetadata:
    model: str | None
    task_type: str


@dataclass(frozen=True)
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


def require_api_key(env_var: str = API_KEY_ENV) -> str:
    api_key = os.getenv(env_var)
    if not api_key:
        raise EnvironmentError(f"Set {env_var} before running this command.")
    return api_key


def load_cache(path: Path) -> Tuple[CacheMetadata, List[EmbeddingRecord]]:
    if not path.exists():
        raise FileNotFoundError(f"Embedding cache not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    metadata = CacheMetadata(
        model=data.get("model"),
        task_type=data.get("task_type", TASK_FALLBACK),
    )

    entries: List[EmbeddingRecord] = []
    for item in data.get("entries", []):
        entries.append(
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

    if not entries:
        raise ValueError(f"No embedding entries found in cache {path}")

    return metadata, entries


def create_client(api_key: str) -> genai.Client:
    return genai.Client(api_key=api_key)


def embed_text(
    client: genai.Client,
    text: str,
    model: str,
    task_type: str,
) -> List[float]:
    config = types.EmbedContentConfig(task_type=task_type)
    response = client.models.embed_content(
        model=model,
        contents=text,
        config=config,
    )
    embeddings = response.embeddings
    if not embeddings:
        raise ValueError("Gemini response did not include embeddings.")
    first_embedding = embeddings[0]
    values = first_embedding.values
    if values is None:
        raise ValueError("Gemini embedding did not include vector values.")
    return list(cast(Sequence[float], values))


def cosine_similarity(vector_a: Sequence[float], vector_b: Sequence[float]) -> float:
    dot = math.fsum(a * b for a, b in zip(vector_a, vector_b, strict=False))
    norm_a = math.sqrt(math.fsum(a * a for a in vector_a))
    norm_b = math.sqrt(math.fsum(b * b for b in vector_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def score_records(
    query_vector: Sequence[float],
    records: Iterable[EmbeddingRecord],
) -> List[Tuple[float, EmbeddingRecord]]:
    scored: List[Tuple[float, EmbeddingRecord]] = []
    for record in records:
        score = cosine_similarity(query_vector, record.vector)
        scored.append((score, record))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored


def find_similar_prs(
    description: str,
    *,
    top_k: int = DEFAULT_TOP_K,
    cache_path: Path = DEFAULT_CACHE_PATH,
    model_override: str | None = None,
    api_key: str | None = None,
    client: genai.Client | None = None,
) -> Tuple[CacheMetadata, List[Tuple[float, EmbeddingRecord]]]:
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    metadata, records = load_cache(cache_path)
    model = model_override or metadata.model
    if not model:
        raise ValueError("Embedding model is not specified in the cache. Pass --model to override it.")

    key = api_key or require_api_key()
    genai_client = client or create_client(key)
    query_vector = embed_text(genai_client, description, model=model, task_type=metadata.task_type)
    scored = score_records(query_vector, records)
    return metadata, scored[: max(1, top_k)]
