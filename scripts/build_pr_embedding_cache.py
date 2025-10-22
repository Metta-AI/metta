from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
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
LOG_FORMAT = "%H%x1f%an%x1f%aI%x1f%s%x1f%b%x1e"


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
        "--min-description-lines",
        type=int,
        default=DEFAULT_MIN_DESCRIPTION_LINES,
        help=("Skip PRs whose descriptions contain this many non-empty lines or fewer (default: 0)."),
    )
    return parser.parse_args()


def _run_git_log() -> str:
    return subprocess.check_output(
        ["git", "log", "--pretty=format:" + LOG_FORMAT],
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
    if len(header_fields) < 5:
        return None

    commit_sha, author, authored_at, subject, body_first = header_fields[:5]
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
    if snapshot.authored_at:
        segments.append(f"Authored at: {snapshot.authored_at}")
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


def main() -> None:
    args = parse_args()

    api_key = os.getenv(API_KEY_ENV)
    if not api_key:
        raise EnvironmentError(f"Set {API_KEY_ENV} before running this script.")

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
    if not pending:
        if removed:
            write_cache(args.cache_path, metadata, existing_records)
            print("Updated cache after applying description length filter.")
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
            )
            existing_records[record.pr_number] = record
        print(f"Embedded {len(batch)} PRs (total cached: {len(existing_records)})")

    write_cache(args.cache_path, metadata, existing_records)
    print(f"Wrote embedding cache to {args.cache_path}")


if __name__ == "__main__":
    main()
