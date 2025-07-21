#!/usr/bin/env python3
"""
Direct fix for remaining sampling syntax.
"""

import re
from pathlib import Path


def fix_file(file_path: Path) -> int:
    """Fix all sampling syntax in a file."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    original_content = content
    
    # First, fix the nested comments by removing everything after the first "# Fixed value"
    lines = content.split('\n')
    new_lines = []
    
    for line in lines:
        if '# Fixed value (was' in line and '${sampling:' in line:
            # Extract the key:value part and the sampling expression
            match = re.search(r'^(\s*[\w_]+:\s*)(\S+).*\$\{sampling:([^}]+)\}', line)
            if match:
                indent_key = match.group(1)
                value = match.group(2)
                sampling_expr = match.group(3)
                new_lines.append(f"{indent_key}{value}  # Fixed value (was ${{sampling:{sampling_expr}}})")
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)
    
    content = '\n'.join(new_lines)
    
    # Now handle any remaining ${sampling:...} that aren't in comments
    def replace_sampling(match):
        params = match.group(1).split(',')
        if len(params) == 3:
            center = params[2].strip()
            return f"{center}  # Fixed value (was {match.group(0)})"
        return match.group(0)
    
    content = re.sub(r'\$\{sampling:([^}]+)\}', replace_sampling, content)
    
    changes = 1 if content != original_content else 0
    
    if changes > 0:
        with open(file_path, 'w') as f:
            f.write(content)
    
    return changes


def main():
    """Main function."""
    workspace_root = Path("/workspace")
    
    # All files that need fixing
    files_to_fix = [
        "configs/env/mettagrid/terrain_from_numpy.yaml",
        "configs/env/mettagrid/extended_sequence/backchain1.yaml",
        "configs/env/mettagrid/extended_sequence/backchain2.yaml",
        "configs/env/mettagrid/extended_sequence/backchain3.yaml",
        "configs/env/mettagrid/game/map_builder/maze.yaml",
        "configs/env/mettagrid/cooperation/experimental/central_table_layout.yaml",
        "configs/env/mettagrid/cooperation/experimental/confined_room_coord.yaml",
        "configs/env/mettagrid/cooperation/experimental/two_rooms_coord.yaml",
        "configs/env/mettagrid/multiagent/experiments/boxshare.yaml",
        "configs/env/mettagrid/multiagent/experiments/boxy.yaml",
        "configs/env/mettagrid/multiagent/experiments/cylinder_world.yaml",
        "configs/env/mettagrid/multiagent/experiments/defaults.yaml",
        "configs/env/mettagrid/multiagent/experiments/manhatten.yaml",
        "configs/env/mettagrid/multiagent/experiments/narrow_world.yaml",
        "configs/env/mettagrid/multiagent/experiments/terrain_from_numpy.yaml",
        "configs/env/mettagrid/multiagent/experiments/varied_terrain.yaml",
        "configs/env/mettagrid/multiagent/multiagent/boxshare.yaml",
        "configs/env/mettagrid/navigation/evals/emptyspace_sparse.yaml",
        "configs/env/mettagrid/navigation_sequence/experiments/cylinder_world.yaml",
        "configs/env/mettagrid/navigation_sequence/experiments/hard_mem_defaults.yaml",
        "configs/env/mettagrid/navigation_sequence/experiments/mem_defaults.yaml",
        "configs/env/mettagrid/navigation_sequence/experiments/sequence_defaults.yaml",
        "configs/env/mettagrid/navigation_sequence/experiments/terrain_from_numpy.yaml",
        "configs/env/mettagrid/navigation/training/defaults.yaml",
        "configs/env/mettagrid/navigation/training/varied_terrain_sparse.yaml",
        "tests/tools/map/maze.yaml",
    ]
    
    print("Applying direct fix to all files...")
    
    total_fixed = 0
    for file_path in files_to_fix:
        full_path = workspace_root / file_path
        if full_path.exists():
            fixed = fix_file(full_path)
            if fixed > 0:
                print(f"Fixed: {file_path}")
                total_fixed += 1
    
    print(f"\nFixed {total_fixed} files")
    
    # Verify
    print("\nVerifying all files are clean...")
    issues = 0
    for file_path in files_to_fix:
        full_path = workspace_root / file_path
        if full_path.exists():
            with open(full_path, 'r') as f:
                content = f.read()
                if '${sampling:' in content:
                    print(f"Still has issues: {file_path}")
                    issues += 1
    
    if issues == 0:
        print("✓ All files are clean!")
    else:
        print(f"⚠ {issues} files still have issues")


if __name__ == "__main__":
    main()