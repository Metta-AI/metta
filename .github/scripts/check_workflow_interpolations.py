from __future__ import annotations

import re
import sys
from pathlib import Path

# Expressions that should never appear inside `run:` blocks without being
# funneled through environment variables first.
DANGEROUS_SUBSTRINGS = (
    "github.event.comment.body",
    "github.event.comment.user.login",
    "github.event.issue.title",
    "github.event.issue.body",
    "github.event.pull_request.title",
    "github.event.pull_request.body",
    "github.event.pull_request.head.ref",
    "github.event.pull_request.head.label",
    "github.event.pull_request.head.repo",
    "github.event.pull_request.user.login",
    "github.actor",
    "github.triggering_actor",
    "github.head_ref",
    "github.ref_name",
    "inputs.",
    "matrix.",
)

EXPR_RE = re.compile(r"\${{\s*([^}]+)\s*}}")


def iter_run_blocks(path: Path):
    lines = path.read_text().splitlines()
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        stripped = line.lstrip()
        indent = len(line) - len(stripped)
        if stripped.startswith("run:"):
            after = stripped[len("run:") :].strip()
            block = []
            line_numbers = []

            if after and not after.startswith(("|", ">")):
                block.append(after)
                line_numbers.append(idx + 1)

            idx += 1
            while idx < len(lines):
                current_line = lines[idx]
                current_stripped = current_line.strip()
                current_indent = len(current_line) - len(current_line.lstrip())

                if current_stripped == "":
                    block.append("")
                    line_numbers.append(idx + 1)
                    idx += 1
                    continue

                if current_indent <= indent:
                    break

                block.append(current_line[current_indent:])
                line_numbers.append(idx + 1)
                idx += 1

            yield block, line_numbers
        else:
            idx += 1


def main() -> int:
    issues: list[str] = []
    for workflow_path in sorted(Path(".github/workflows").glob("*.yml")):
        for block, line_numbers in iter_run_blocks(workflow_path):
            block_text = "\n".join(block)
            for match in EXPR_RE.finditer(block_text):
                expr = match.group(1)
                for needle in DANGEROUS_SUBSTRINGS:
                    if needle in expr:
                        line_offset = block_text[: match.start()].count("\n")
                        line_no = line_numbers[min(line_offset, len(line_numbers) - 1)]
                        issues.append(
                            f"{workflow_path}:{line_no}: unsafe interpolation of `{expr.strip()}` in run block"
                        )
                        break

    if issues:
        print("Found unsafe workflow interpolations:", file=sys.stderr)
        for issue in issues:
            print(f"  - {issue}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
