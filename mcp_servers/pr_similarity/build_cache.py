from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import boto3
import numpy as np
from botocore.exceptions import ClientError
from google import genai
from google.genai import types

from metta.tools.pr_similarity import resolve_cache_paths

DEFAULT_MODEL = "gemini-embedding-001"
DEFAULT_CACHE_PATH = Path("mcp_servers/pr_similarity/cache/pr_embeddings")
API_KEY_ENV = "GEMINI_API_KEY"
DEFAULT_BATCH_SIZE = 16
TASK_TYPE = "semantic_similarity"
DEFAULT_MIN_DESCRIPTION_LINES = 0
PR_NUMBER_RE = re.compile(r"#(\d+)\b")
LOG_FORMAT = "%H%x1f%an%x1f%aI%x1f%cI%x1f%s%x1f%b%x1e"


def _require_merged_at(entry: Dict[str, object]) -> str:
    merged_at = entry.get("merged_at")
    if not merged_at:
        raise ValueError(
            "Embedding cache entry is missing 'merged_at'. Rebuild the cache from scratch "
            "to populate merge timestamps.",
        )
    return str(merged_at)


@dataclass
class PullRequestSnapshot:
    pr_number: int
    title: str
    description: str
    author: str
    additions: int
    deletions: int
    files_changed: int
    commit_sha: str
    authored_at: str
    merged_at: str


@dataclass
class EmbeddingRecord:
    pr_number: int
    vector: List[float]
    title: str
    description: str
    author: str
    additions: int
    deletions: int
    files_changed: int
    commit_sha: str
    authored_at: str
    merged_at: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Gemini embeddings for PR history entries and cache them locally.",
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=DEFAULT_CACHE_PATH,
        help=f"Where to store the embedding cache (default: {DEFAULT_CACHE_PATH}).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Embedding model identifier (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Number of PR texts to embed per request (default: {DEFAULT_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--max-prs",
        type=int,
        help="Optional upper bound on number of PRs to embed (useful for quick tests).",
    )
    parser.add_argument(
        "--download-from-s3",
        action="store_true",
        help="Download cache from S3 before processing (useful for cronjobs).",
    )
    parser.add_argument(
        "--no-upload",
        action="store_false",
        dest="upload",
        default=True,
        help="Skip uploading cache files to S3 after writing.",
    )
    parser.add_argument(
        "--s3-bucket",
        default="softmax-public",
        help="S3 bucket for cache storage (default: softmax-public).",
    )
    parser.add_argument(
        "--s3-prefix",
        default="pr-cache/",
        help="S3 prefix for cache files (default: pr-cache/).",
    )
    parser.add_argument(
        "--min-description-lines",
        type=int,
        default=DEFAULT_MIN_DESCRIPTION_LINES,
        help=("Skip PRs whose descriptions contain this many non-empty lines or fewer (default: 0)."),
    )
    parser.add_argument(
        "--allow-full-rebuild",
        action="store_true",
        help="Allow embedding all PRs when no existing cache is found (required for first-time setup).",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Force a full rebuild by ignoring existing cache and re-embedding all PRs.",
    )
    return parser.parse_args()


def _run_git_log() -> str:
    return subprocess.check_output(
        ["git", "log", "--all", "--remotes", "--pretty=format:" + LOG_FORMAT],
        text=True,
    )


def _collect_stats(commit_sha: str) -> tuple[int, int, int]:
    additions = deletions = files_changed = 0
    output = subprocess.check_output(
        ["git", "show", "--numstat", "--format=", commit_sha],
        text=True,
    )
    for line in output.splitlines():
        if "\t" not in line:
            continue
        add_str, del_str, _ = line.split("\t", 2)
        additions += int(add_str) if add_str.isdigit() else 0
        deletions += int(del_str) if del_str.isdigit() else 0
        files_changed += 1
    return additions, deletions, files_changed


def _extract_snapshot(raw_entry: str) -> PullRequestSnapshot | None:
    raw_entry = raw_entry.strip("\n")
    if not raw_entry:
        return None

    lines = raw_entry.split("\n")
    header_fields = lines[0].split("\x1f")
    if len(header_fields) < 6:
        return None

    commit_sha, author, authored_at, merged_at, subject, body_first = header_fields[:6]
    body_lines: List[str] = []
    if body_first:
        body_lines.append(body_first)
    body_lines.extend(lines[1:])

    match = PR_NUMBER_RE.search(subject)
    if not match:
        return None

    pr_number = int(match.group(1))
    filtered_body = [line for line in body_lines if line.strip()]

    if subject.lower().startswith("merge pull request #") and filtered_body:
        title = filtered_body[0].strip()
        description_lines = filtered_body[1:]
    else:
        title = subject.split(" (#")[0].strip()
        description_lines = filtered_body

    description = "\n".join(description_lines).strip()
    additions, deletions, files_changed = _collect_stats(commit_sha)

    return PullRequestSnapshot(
        pr_number=pr_number,
        title=title or subject,
        description=description,
        author=author,
        additions=additions,
        deletions=deletions,
        files_changed=files_changed,
        commit_sha=commit_sha,
        authored_at=authored_at,
        merged_at=merged_at,
    )


def collect_snapshots() -> List[PullRequestSnapshot]:
    snapshots: Dict[int, PullRequestSnapshot] = {}
    for raw_entry in _run_git_log().split("\x1e"):
        snapshot = _extract_snapshot(raw_entry)
        if snapshot is None:
            continue
        snapshots[snapshot.pr_number] = snapshot
    return [snapshots[key] for key in sorted(snapshots)]


def load_existing_cache(path: Path) -> Tuple[Dict[int, EmbeddingRecord], Dict[str, object]]:
    meta_path, vectors_path = resolve_cache_paths(path)

    if meta_path.exists() and vectors_path.exists():
        with meta_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        payload = np.load(vectors_path)
        pr_numbers = payload["pr_numbers"]
        vectors = payload["vectors"]
        vector_by_pr = {int(pr): vectors[index].tolist() for index, pr in enumerate(pr_numbers)}

        metadata = {
            "model": data.get("model", DEFAULT_MODEL),
            "task_type": data.get("task_type", TASK_TYPE),
        }

        entries: Dict[int, EmbeddingRecord] = {}
        for item in data.get("entries", []):
            pr_number = int(item["pr_number"])
            vector = vector_by_pr.get(pr_number)
            if vector is None:
                continue
            record = EmbeddingRecord(
                pr_number=pr_number,
                vector=vector,
                title=item.get("title", ""),
                description=item.get("description", ""),
                author=item.get("author", ""),
                additions=int(item.get("additions", 0)),
                deletions=int(item.get("deletions", 0)),
                files_changed=int(item.get("files_changed", 0)),
                commit_sha=item.get("commit_sha", ""),
                authored_at=item.get("authored_at", ""),
                merged_at=_require_merged_at(item),
            )
            entries[pr_number] = record

        return entries, metadata

    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        metadata = {
            "model": data.get("model", DEFAULT_MODEL),
            "task_type": data.get("task_type", TASK_TYPE),
        }

        entries: Dict[int, EmbeddingRecord] = {}
        for item in data.get("entries", []):
            record = EmbeddingRecord(
                pr_number=int(item["pr_number"]),
                vector=list(item["vector"]),
                title=item.get("title", ""),
                description=item.get("description", ""),
                author=item.get("author", ""),
                additions=int(item.get("additions", 0)),
                deletions=int(item.get("deletions", 0)),
                files_changed=int(item.get("files_changed", 0)),
                commit_sha=item.get("commit_sha", ""),
                authored_at=item.get("authored_at", ""),
                merged_at=_require_merged_at(item),
            )
            entries[record.pr_number] = record
        return entries, metadata

    return {}, {"model": DEFAULT_MODEL, "task_type": TASK_TYPE}


def count_description_lines(text: str) -> int:
    return sum(1 for line in text.splitlines() if line.strip())


def build_embedding_text(snapshot: PullRequestSnapshot) -> str:
    description = snapshot.description.strip() or "No additional description provided."
    segments = [
        f"PR #{snapshot.pr_number}: {snapshot.title}",
        f"Author: {snapshot.author}",
        f"Description: {description}",
        (
            "Code stats: "
            f"files_changed={snapshot.files_changed}, "
            f"additions={snapshot.additions}, "
            f"deletions={snapshot.deletions}"
        ),
        f"Commit: {snapshot.commit_sha}",
    ]
    if snapshot.merged_at:
        segments.append(f"Merged at: {snapshot.merged_at}")
    return "\n".join(segments)


def chunked(items: List[PullRequestSnapshot], size: int) -> List[List[PullRequestSnapshot]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def generate_embeddings(
    client: genai.Client,
    snapshots: List[PullRequestSnapshot],
    model: str,
) -> List[List[float]]:
    texts = [build_embedding_text(snapshot) for snapshot in snapshots]
    config = types.EmbedContentConfig(task_type=TASK_TYPE)
    response = client.models.embed_content(
        model=model,
        contents=texts,
        config=config,
    )
    vectors: List[List[float]] = []
    for embedding in response.embeddings:
        vectors.append(list(embedding.values))
    return vectors


def write_cache(
    path: Path,
    metadata: Dict[str, object],
    records: Dict[int, EmbeddingRecord],
) -> None:
    meta_path, vectors_path = resolve_cache_paths(path)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    ordered = [records[key] for key in sorted(records)]
    entries_payload = []
    vectors: List[List[float]] = []
    pr_numbers: List[int] = []

    for record in ordered:
        entries_payload.append(
            {
                "pr_number": record.pr_number,
                "title": record.title,
                "description": record.description,
                "author": record.author,
                "additions": record.additions,
                "deletions": record.deletions,
                "files_changed": record.files_changed,
                "commit_sha": record.commit_sha,
                "authored_at": record.authored_at,
                "merged_at": record.merged_at,
            },
        )
        pr_numbers.append(record.pr_number)
        vectors.append(record.vector)

    meta_payload = {
        "model": metadata.get("model", DEFAULT_MODEL),
        "task_type": metadata.get("task_type", TASK_TYPE),
        "entries": entries_payload,
    }
    meta_path.write_text(json.dumps(meta_payload, indent=2) + "\n", encoding="utf-8")

    np.savez_compressed(
        vectors_path,
        pr_numbers=np.asarray(pr_numbers, dtype=np.int64),
        vectors=np.asarray(vectors, dtype=np.float32),
    )


def download_cache_from_s3(
    cache_path: Path,
    bucket: str = "softmax-public",
    prefix: str = "pr-cache/",
) -> bool:
    """Download PR embedding cache from S3 if it doesn't exist locally.

    Returns True if cache was downloaded or already exists, False if download failed.
    """
    meta_path, vectors_path = resolve_cache_paths(cache_path)

    if meta_path.exists() and vectors_path.exists():
        print(f"Cache already exists at {cache_path}")
        return True

    try:
        s3 = boto3.client("s3")
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(bucket, prefix + meta_path.name, str(meta_path))
        s3.download_file(bucket, prefix + vectors_path.name, str(vectors_path))
        print(f"Downloaded PR similarity cache from s3://{bucket}/{prefix}")
        return True
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") == "404":
            print("Cache not found in S3 (this is expected on first run)")
        else:
            print(f"Error downloading cache from S3: {e}")
        return False
    except Exception as e:
        print(f"Unable to download PR similarity cache: {e}")
        return False


def upload_cache_to_s3(cache_path: Path, bucket: str = "softmax-public", prefix: str = "pr-cache/") -> None:
    """Upload cache files to S3."""
    meta_path, vectors_path = resolve_cache_paths(cache_path)

    if not meta_path.exists() or not vectors_path.exists():
        raise FileNotFoundError(f"Cache files not found: {meta_path} or {vectors_path}")

    try:
        s3 = boto3.client("s3")
        s3.upload_file(str(meta_path), bucket, prefix + meta_path.name)
        s3.upload_file(str(vectors_path), bucket, prefix + vectors_path.name)
        print(f"Uploaded cache to s3://{bucket}/{prefix}")
    except Exception as e:
        raise RuntimeError(f"Failed to upload cache to S3: {e}") from e


def _get_api_key() -> str:
    """Get API key from environment variable or AWS Secrets Manager."""
    api_key = os.getenv(API_KEY_ENV)
    if api_key:
        return api_key

    # Try AWS Secrets Manager
    try:
        client = boto3.client("secretsmanager", region_name="us-east-1")
        response = client.get_secret_value(SecretId="GEMINI-API-KEY")
        return response["SecretString"].strip()
    except Exception as e:
        raise EnvironmentError(
            f"Set {API_KEY_ENV} environment variable or ensure AWS Secrets Manager access to 'GEMINI-API-KEY': {e}"
        ) from e


def main() -> None:
    args = parse_args()

    if args.download_from_s3:
        print("Downloading cache from S3...")
        download_succeeded = download_cache_from_s3(args.cache_path, args.s3_bucket, args.s3_prefix)
        meta_path, vectors_path = resolve_cache_paths(args.cache_path)
        cache_exists_locally = meta_path.exists() and vectors_path.exists()

        if not download_succeeded and not cache_exists_locally and not args.force_rebuild:
            raise FileNotFoundError(
                f"Failed to download cache from S3 and no local cache exists at {args.cache_path}. "
                "This would trigger a full rebuild of all PR embeddings, which is expensive. "
                "If this is intentional (e.g., first-time setup), run with --allow-full-rebuild or --force-rebuild."
            )

    api_key = _get_api_key()

    if args.force_rebuild:
        existing_records, metadata = {}, {"model": args.model, "task_type": TASK_TYPE}
        print("Force rebuild enabled: ignoring existing cache and re-embedding all PRs.")
    else:
        existing_records, metadata = load_existing_cache(args.cache_path)
        metadata["model"] = args.model
        metadata["task_type"] = TASK_TYPE

    snapshots = collect_snapshots()
    if args.max_prs is not None:
        snapshots = snapshots[: args.max_prs]

    eligible_snapshots: List[PullRequestSnapshot] = []
    for snapshot in snapshots:
        if count_description_lines(snapshot.description) <= args.min_description_lines:
            continue
        eligible_snapshots.append(snapshot)

    eligible_numbers = {snapshot.pr_number for snapshot in eligible_snapshots}
    removed = 0
    for pr_number in list(existing_records.keys()):
        if pr_number not in eligible_numbers:
            existing_records.pop(pr_number)
            removed += 1
    if removed:
        print(f"Removed {removed} PRs from cache due to description length filter.")

    pending = [snapshot for snapshot in eligible_snapshots if snapshot.pr_number not in existing_records]

    if not existing_records and pending and not args.allow_full_rebuild and not args.force_rebuild:
        raise RuntimeError(
            f"No existing cache found and {len(pending)} PRs would be embedded. "
            "This would be expensive. If this is intentional, run with --allow-full-rebuild or --force-rebuild flag."
        )

    if not pending:
        if removed:
            write_cache(args.cache_path, metadata, existing_records)
            print("Updated cache after applying description length filter.")
            if args.upload:
                upload_cache_to_s3(args.cache_path, args.s3_bucket, args.s3_prefix)
        else:
            print("Embedding cache already up to date.")
        return

    client = genai.Client(api_key=api_key)

    for batch in chunked(pending, args.batch_size):
        vectors = generate_embeddings(client, batch, args.model)
        for snapshot, vector in zip(batch, vectors, strict=False):
            record = EmbeddingRecord(
                pr_number=snapshot.pr_number,
                vector=vector,
                title=snapshot.title,
                description=snapshot.description,
                author=snapshot.author,
                additions=snapshot.additions,
                deletions=snapshot.deletions,
                files_changed=snapshot.files_changed,
                commit_sha=snapshot.commit_sha,
                authored_at=snapshot.authored_at,
                merged_at=snapshot.merged_at,
            )
            existing_records[record.pr_number] = record
        print(f"Embedded {len(batch)} PRs (total cached: {len(existing_records)})")

    write_cache(args.cache_path, metadata, existing_records)
    print(f"Wrote embedding cache to {args.cache_path}")
    if args.upload:
        upload_cache_to_s3(args.cache_path, args.s3_bucket, args.s3_prefix)


if __name__ == "__main__":
    main()
