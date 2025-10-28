from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

PR_NUMBER_RE = re.compile(r"#(\d+)\b")
LOG_FORMAT = "%H%x1f%an%x1f%ae%x1f%aI%x1f%s%x1f%b%x1e"


@dataclass
class PullRequestEntry:
    pr_number: int
    title: str
    author: str
    author_email: str
    authored_at: str
    commit_sha: str
    description: str
    files_changed: int
    additions: int
    deletions: int

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _run_git_log() -> str:
    return subprocess.check_output(
        ["git", "log", "--pretty=format:" + LOG_FORMAT],
        text=True,
    )


def _collect_stats(commit_sha: str) -> tuple[int, int, int]:
    additions = 0
    deletions = 0
    files_changed = 0
    output = subprocess.check_output(
        ["git", "show", "--numstat", "--format=", commit_sha],
        text=True,
    )
    for line in output.splitlines():
        if "\t" not in line:
            continue
        add_str, del_str, _ = line.split("\t", 2)
        add = int(add_str) if add_str.isdigit() else 0
        delete = int(del_str) if del_str.isdigit() else 0
        additions += add
        deletions += delete
        files_changed += 1
    return additions, deletions, files_changed


def _extract_entry(raw_entry: str) -> PullRequestEntry | None:
    raw_entry = raw_entry.strip("\n")
    if not raw_entry:
        return None

    lines = raw_entry.split("\n")
    header = lines[0]
    fields = header.split("\x1f")
    if len(fields) < 6:
        return None

    commit_sha, author, author_email, authored_at, subject, body_first = fields[:6]
    body_lines: List[str] = [body_first] if body_first else []
    for line in lines[1:]:
        body_lines.append(line)

    pr_match = PR_NUMBER_RE.search(subject)
    if not pr_match:
        return None

    pr_number = int(pr_match.group(1))
    filtered_body_lines = [line for line in body_lines if line.strip()]

    if subject.lower().startswith("merge pull request #") and filtered_body_lines:
        title = filtered_body_lines[0].strip()
        description_lines = filtered_body_lines[1:]
    else:
        title = subject.split(" (#")[0].strip()
        description_lines = filtered_body_lines

    description = "\n".join(description_lines).strip()
    additions, deletions, files_changed = _collect_stats(commit_sha)

    return PullRequestEntry(
        pr_number=pr_number,
        title=title or subject,
        author=author,
        author_email=author_email,
        authored_at=authored_at,
        commit_sha=commit_sha,
        description=description,
        files_changed=files_changed,
        additions=additions,
        deletions=deletions,
    )


def collect_pull_requests() -> Dict[int, PullRequestEntry]:
    output = _run_git_log()
    entries: Dict[int, PullRequestEntry] = {}
    for raw_entry in output.split("\x1e"):
        entry = _extract_entry(raw_entry)
        if entry is None:
            continue
        entries[entry.pr_number] = entry
    return entries


def load_existing(path: Path) -> Dict[int, PullRequestEntry]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        raw_entries = json.load(handle)
    entries: Dict[int, PullRequestEntry] = {}
    for raw_entry in raw_entries:
        pr_number = int(raw_entry["pr_number"])
        entries[pr_number] = PullRequestEntry(
            pr_number=pr_number,
            title=raw_entry.get("title", ""),
            author=raw_entry.get("author", ""),
            author_email=raw_entry.get("author_email", ""),
            authored_at=raw_entry.get("authored_at", ""),
            commit_sha=raw_entry.get("commit_sha", ""),
            description=raw_entry.get("description", ""),
            files_changed=int(raw_entry.get("files_changed", 0)),
            additions=int(raw_entry.get("additions", 0)),
            deletions=int(raw_entry.get("deletions", 0)),
        )
    return entries


def merge_entries(
    existing: Dict[int, PullRequestEntry],
    discovered: Dict[int, PullRequestEntry],
) -> List[PullRequestEntry]:
    merged = {**existing}
    merged.update(discovered)
    return [merged[key] for key in sorted(merged)]


def write_entries(path: Path, entries: List[PullRequestEntry]) -> None:
    payload = [entry.to_dict() for entry in entries]
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract PR metadata from Git history into JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("pr_history.json"),
        help="Path to the JSON file to update (default: pr_history.json)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    existing_entries = load_existing(args.output)
    discovered_entries = collect_pull_requests()
    merged_entries = merge_entries(existing_entries, discovered_entries)
    write_entries(args.output, merged_entries)


if __name__ == "__main__":
    main()
