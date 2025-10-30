from __future__ import annotations

from pathlib import Path
import re

MAP_DIR = Path(__file__).resolve().parents[2] / "cogames" / "src" / "cogames" / "maps"
FILES = sorted(MAP_DIR.glob("machina_eval_exp*.map"))

PROTECT_START = 17  # 1-based within map_data block
PROTECT_END = 27


def normalize_file(path: Path) -> tuple[bool, str]:
    text = path.read_text()
    lines = text.splitlines()

    # locate map_data block
    map_data_idx = None
    for i, line in enumerate(lines):
        if line.strip() == "map_data: |-":
            map_data_idx = i
            break
    if map_data_idx is None:
        return False, f"{path.name}: map_data not found"

    start = map_data_idx + 1
    # consume leading indentation line(s) (blank or spaces)
    # Next lines are indented map rows until 'char_to_name_map:'
    end = None
    for i in range(start, len(lines)):
        if lines[i].strip().startswith("char_to_name_map:"):
            end = i
            break
    if end is None:
        return False, f"{path.name}: char_to_name_map not found"

    content = lines[start:end]
    # Strip common leading spaces
    def leading_spaces(s: str) -> int:
        return len(re.match(r"^ *", s).group(0))

    common = min((leading_spaces(l) for l in content if l.strip() != ""), default=0)
    rows = [l[common:] for l in content]

    # Determine target width from the first non-empty row
    non_empty = [r for r in rows if r.strip() != ""]
    if not non_empty:
        return False, f"{path.name}: empty map"
    target = len(non_empty[0])

    changed = False
    new_rows: list[str] = []
    for idx, r in enumerate(rows, start=1):
        rr = r.rstrip("\n")
        # Skip protected rows 17–27 (inclusive)
        if PROTECT_START <= idx <= PROTECT_END:
            new_rows.append(rr)
            continue
        w = len(rr)
        if w < target:
            rr = rr + "." * (target - w)
            changed = True
        elif w > target:
            rr = rr[:target]
            changed = True
        new_rows.append(rr)

    if changed:
        # Re-apply indentation
        indent = " " * common
        lines[start:end] = [indent + r for r in new_rows]
        path.write_text("\n".join(lines) + "\n")
        return True, f"{path.name}: normalized to width {target}"
    else:
        return True, f"{path.name}: OK width {target}"


def main() -> None:
    for p in FILES:
        ok, msg = normalize_file(p)
        print(msg)


if __name__ == "__main__":
    main()
