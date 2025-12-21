from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple, cast

import numpy as np
from google import genai
from google.genai import types

DEFAULT_CACHE_PATH = Path("mcp_servers/pr_similarity/cache/pr_embeddings")
DEFAULT_TOP_K = 10
TASK_FALLBACK = "semantic_similarity"
API_KEY_ENV = "GEMINI_API_KEY"


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
    merged_at: str


def _require_merged_at(entry: Dict[str, Any]) -> str:
    merged_at = entry.get("merged_at")
    if not merged_at:
        raise ValueError(
            "Embedding cache entry is missing 'merged_at'. Rebuild the cache with "
            "`python mcp_servers/pr_similarity/build_cache.py`.",
        )
    return str(merged_at)


def _parse_iso_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        timestamp = datetime.fromisoformat(value)
    except ValueError:
        return None
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp


def _coerce_min_authored_at(value: datetime | str | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        result = value
    else:
        try:
            result = datetime.fromisoformat(value)
        except ValueError as error:
            raise ValueError("min_authored_at must be a valid ISO 8601 timestamp.") from error
    if result.tzinfo is None:
        result = result.replace(tzinfo=timezone.utc)
    return result


def require_api_key(env_var: str = API_KEY_ENV) -> str:
    api_key = os.getenv(env_var)
    if not api_key:
        raise EnvironmentError(f"Set {env_var} before running this command.")
    return api_key


def resolve_cache_paths(path: Path) -> Tuple[Path, Path]:
    base = path if path.suffix == "" else path.with_suffix("")
    meta_path = base.with_suffix(".json")
    vectors_path = base.with_suffix(".npz")
    return meta_path, vectors_path


def _load_legacy_cache(path: Path) -> Tuple[CacheMetadata, List[EmbeddingRecord]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    metadata = CacheMetadata(
        model=data.get("model"),
        task_type=data.get("task_type", TASK_FALLBACK),
    )

    entries: List[EmbeddingRecord] = []
    for item in data.get("entries", []):
        vector_data = item.get("vector")
        if vector_data is None:
            raise ValueError(
                "Legacy embedding cache is missing inline vector data. "
                "Make sure both JSON and NPZ files from the new format are present.",
            )
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
                vector=list(vector_data),
                merged_at=_require_merged_at(item),
            ),
        )

    if not entries:
        raise ValueError(f"No embedding entries found in cache {path}")

    return metadata, entries


def load_cache(path: Path) -> Tuple[CacheMetadata, List[EmbeddingRecord]]:
    meta_path, vectors_path = resolve_cache_paths(path)
    if meta_path.exists() and vectors_path.exists():
        with meta_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        metadata = CacheMetadata(
            model=data.get("model"),
            task_type=data.get("task_type", TASK_FALLBACK),
        )

        payload = np.load(vectors_path)
        pr_numbers = payload["pr_numbers"]
        vectors = payload["vectors"]
        vector_by_pr = {int(pr): vectors[index].tolist() for index, pr in enumerate(pr_numbers)}

        entries: List[EmbeddingRecord] = []
        for item in data.get("entries", []):
            pr_number = int(item["pr_number"])
            vector = vector_by_pr.get(pr_number)
            if vector is None:
                continue
            entries.append(
                EmbeddingRecord(
                    pr_number=pr_number,
                    title=item.get("title", ""),
                    description=item.get("description", ""),
                    author=item.get("author", ""),
                    additions=int(item.get("additions", 0)),
                    deletions=int(item.get("deletions", 0)),
                    files_changed=int(item.get("files_changed", 0)),
                    commit_sha=item.get("commit_sha", ""),
                    authored_at=item.get("authored_at", ""),
                    vector=vector,
                    merged_at=_require_merged_at(item),
                ),
            )

        if not entries:
            raise ValueError(f"No embedding entries found in cache {vectors_path}")

        return metadata, entries

    if meta_path.exists():
        return _load_legacy_cache(meta_path)

    raise FileNotFoundError(f"Embedding cache not found: {meta_path} / {vectors_path}")


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
    min_authored_at: datetime | str | None = None,
) -> Tuple[CacheMetadata, List[Tuple[float, EmbeddingRecord]]]:
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    metadata, records = load_cache(cache_path)
    min_date = _coerce_min_authored_at(min_authored_at)
    if min_date is not None:
        filtered_records: List[EmbeddingRecord] = []
        for record in records:
            timestamp = _parse_iso_datetime(record.merged_at)
            if timestamp is None:
                raise ValueError(
                    "Embedding cache entry is missing a valid 'merged_at' timestamp. "
                    "Rebuild the cache with `python mcp_servers/pr_similarity/build_cache.py`.",
                )
            if timestamp >= min_date:
                filtered_records.append(record)
        records = filtered_records
        if not records:
            return metadata, []

    model = model_override or metadata.model
    if not model:
        raise ValueError("Embedding model is not specified in the cache. Pass --model to override it.")

    key = api_key or require_api_key()
    genai_client = client or create_client(key)
    query_vector = embed_text(genai_client, description, model=model, task_type=metadata.task_type)
    scored = score_records(query_vector, records)
    return metadata, scored[: max(1, top_k)]
