#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple


def run_git(args: Sequence[str]) -> str:
    """Run a git command and return stdout."""
    completed = subprocess.run(
        ["git", *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def find_merge_base(base_refs: Sequence[str]) -> Tuple[str, str]:
    """Return the merge-base SHA and the ref it succeeded against."""
    for ref in base_refs:
        try:
            base_sha = run_git(["merge-base", "HEAD", ref])
        except subprocess.CalledProcessError:
            continue
        if base_sha:
            return base_sha, ref
    raise RuntimeError(
        "Unable to determine merge base. "
        "Provide one or more valid refs with --base-ref."
    )


def git_diff(base_sha: str, pathspec: str) -> str:
    """Return the diff between base_sha and the current workspace for pathspec."""
    completed = subprocess.run(
        ["git", "diff", base_sha, "--", pathspec],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


@dataclass
class ModelCompatibilityReport:
    """Create a structured view of diff data that can be serialized as YAML."""

    base_refs: Sequence[str]
    path: str

    def to_dict(self) -> Dict[str, str]:
        """Return a dictionary containing merge-base info and path diff."""
        base_sha, used_ref = find_merge_base(self.base_refs)
        diff_text = git_diff(base_sha, self.path)
        return {
            "base_ref": used_ref,
            "merge_base_sha": base_sha,
            "path": self.path,
            "diff": diff_text,
        }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Show the merge-base SHA with main and the diff against the current "
            "workspace restricted to a path."
        )
    )
    parser.add_argument(
        "--base-ref",
        action="append",
        help=(
            "Reference to compare against. "
            "Can be given multiple times to specify fallbacks. "
            "Defaults to origin/main then main."
        ),
    )
    parser.add_argument(
        "--path",
        default="agent/src/metta/agent",
        help="Path to include in the diff (default: agent/src/metta/agent).",
    )
    args = parser.parse_args(argv)

    base_refs = args.base_ref or ["origin/main", "main"]

    report = ModelCompatibilityReport(base_refs=base_refs, path=args.path)

    try:
        result = report.to_dict()
    except RuntimeError as exc:
        parser.error(str(exc))
    except subprocess.CalledProcessError as exc:
        parser.error(
            f"git command failed with exit code {exc.returncode}: "
            f"{exc.stderr.strip() if exc.stderr else 'no stderr'}"
        )

    print(f"Merge base against {result['base_ref']}: {result['merge_base_sha']}")

    diff_text = result["diff"]
    if diff_text:
        print("--- diff start ---")
        print(diff_text.rstrip())
        print("--- diff end ---")
    else:
        print("No differences detected for the specified path.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
