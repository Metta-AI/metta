#!/usr/bin/env python3
"""
Normalize markdown files by removing blank lines except in specific cases.

Preserves blank lines after:
- List items (lines with '  - ' or '  * ')
- Closing code blocks (lines with '```')
- Numbered list items (lines with '  1. ', '  2. ', etc.)
"""

import re
import sys


def should_preserve_blank_line(prev_line: str) -> bool:
    """Check if blank line after this line should be preserved."""
    if not prev_line:
        return False

    # Preserve blank lines after list items (bullet or dash)
    if re.match(r"^\s*[-*]\s", prev_line):
        return True

    # Preserve blank lines after closing code blocks
    if re.match(r"^\s*```\s*$", prev_line):
        return True

    # Preserve blank lines after numbered list items
    if re.match(r"^\s*\d+\.\s", prev_line):
        return True

    return False


def normalize_markdown(content: str) -> str:
    """Remove blank lines except those following special patterns or inside code blocks."""
    lines = content.split("\n")
    result = []
    prev_line = ""
    in_code_block = False

    for line in lines:
        # Check if we're entering or exiting a code block
        if re.match(r"^\s*```", line):
            in_code_block = not in_code_block
            result.append(line)
            if line.strip():
                prev_line = line
            continue

        # If we're inside a code block, preserve everything unchanged
        if in_code_block:
            result.append(line)
            continue

        # If this is a blank line (and we're not in a code block)
        if not line.strip():
            # Only keep it if the previous line warrants preservation
            if should_preserve_blank_line(prev_line):
                result.append(line)
        else:
            result.append(line)

        # Update prev_line only if current line is not blank
        if line.strip():
            prev_line = line

    return "\n".join(result)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: normalize_markdown.py <file>", file=sys.stderr)
        sys.exit(1)

    filepath = sys.argv[1]

    with open(filepath, "r") as f:
        content = f.read()

    normalized = normalize_markdown(content)

    with open(filepath, "w") as f:
        f.write(normalized)
