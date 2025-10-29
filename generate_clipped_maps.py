#!/usr/bin/env python3
"""Generate clipped versions of eval maps with varying difficulty levels."""

import random
from pathlib import Path


def read_map_file(map_path: Path) -> tuple[str, str, dict]:
    """Read a map file and parse its components."""
    content = map_path.read_text()
    lines = content.split('\n')

    # Find where map_data starts and ends
    map_data_start = None
    map_data_end = None
    char_map_start = None

    for i, line in enumerate(lines):
        if 'map_data: |-' in line:
            map_data_start = i + 1
        elif 'char_to_name_map:' in line:
            map_data_end = i
            char_map_start = i
            break

    # Extract components
    header = '\n'.join(lines[:map_data_start])
    map_grid = lines[map_data_start:map_data_end]
    char_map = '\n'.join(lines[char_map_start:])

    # Parse char_to_name_map
    char_to_name = {}
    for line in lines[char_map_start:]:
        if '": ' in line and '"' in line:
            parts = line.strip().split('": ')
            if len(parts) == 2:
                char = parts[0].strip('"').strip()
                name = parts[1].strip().strip('"')
                char_to_name[char] = name

    return header, map_grid, char_to_name


def find_extractors(map_grid: list[str], char_to_name: dict) -> dict[str, list[tuple[int, int]]]:
    """Find all extractor positions in the map."""
    extractors = {
        'carbon': [],
        'oxygen': [],
        'germanium': [],
        'silicon': []
    }

    # Reverse mapping: char -> resource type
    for char, name in char_to_name.items():
        for resource in extractors.keys():
            if resource in name and 'extractor' in name:
                # Find all positions of this char
                for row_idx, row in enumerate(map_grid):
                    for col_idx, cell in enumerate(row):
                        if cell == char:
                            extractors[resource].append((row_idx, col_idx))

    return extractors


def clip_extractors(map_grid: list[str], extractors: dict, num_to_clip: dict[str, int]) -> list[str]:
    """Create a new map with specified extractors clipped."""
    # Convert to mutable structure
    new_grid = [list(row) for row in map_grid]

    # Clip specified number of each resource type
    for resource, positions in extractors.items():
        if num_to_clip.get(resource, 0) > 0:
            # Randomly select which extractors to clip
            to_clip = random.sample(positions, min(num_to_clip[resource], len(positions)))
            for row, col in to_clip:
                # Mark as clipped by adding a suffix (we'll use lowercase for clipped)
                current_char = new_grid[row][col]
                # Use a marker that we'll add to the char map
                new_grid[row][col] = current_char.lower()

    # Convert back to strings
    return [''.join(row) for row in new_grid]


def update_char_map(char_to_name: dict, clipped_resources: set[str]) -> str:
    """Update character mapping to include clipped extractors."""
    lines = ["char_to_name_map:"]

    # Add original mappings
    for char, name in sorted(char_to_name.items()):
        lines.append(f'  "{char}": {name}')

    # Add clipped versions
    for char, name in sorted(char_to_name.items()):
        for resource in clipped_resources:
            if resource in name and 'extractor' in name:
                # Add lowercase version as clipped
                lines.append(f'  "{char.lower()}": {name}_clipped')

    return '\n'.join(lines)


def generate_clipped_map(
    source_map: Path,
    output_map: Path,
    clip_config: dict[str, int],
    seed: int = 42
):
    """Generate a clipped version of a map."""
    random.seed(seed)

    # Read original map
    header, map_grid, char_to_name = read_map_file(source_map)

    # Find all extractors
    extractors = find_extractors(map_grid, char_to_name)

    print(f"\nProcessing {source_map.name}:")
    for resource, positions in extractors.items():
        total = len(positions)
        to_clip = clip_config.get(resource, 0)
        if to_clip > 0:
            print(f"  {resource}: {to_clip}/{total} clipped")

    # Clip extractors
    new_grid = clip_extractors(map_grid, extractors, clip_config)

    # Update char map
    clipped_resources = {r for r, n in clip_config.items() if n > 0}
    new_char_map = update_char_map(char_to_name, clipped_resources)

    # Write new map
    output_content = header + '\n' + '\n'.join(new_grid) + '\n' + new_char_map + '\n'
    output_map.write_text(output_content)
    print(f"  → Created {output_map.name}")


def main():
    """Generate all clipped map variants."""
    maps_dir = Path("packages/cogames/src/cogames/maps")

    # Define clipping configurations for different difficulty levels
    # Easy: Clip 1 extractor of the bottleneck resource
    # Medium: Clip 1-2 extractors across multiple resources
    # Hard: Clip 2-3 extractors across multiple resources

    configs = {
        # OxygenBottleneck variants (exp02)
        "machina_eval_exp02_clip_easy.map": {
            "source": "machina_eval_exp02.map",
            "clip": {"oxygen": 1},
            "seed": 42
        },
        "machina_eval_exp02_clip_medium.map": {
            "source": "machina_eval_exp02.map",
            "clip": {"oxygen": 2, "carbon": 1},
            "seed": 43
        },
        "machina_eval_exp02_clip_hard.map": {
            "source": "machina_eval_exp02.map",
            "clip": {"oxygen": 2, "carbon": 1, "germanium": 1},
            "seed": 44
        },

        # GermaniumRush variants (exp03)
        "machina_eval_exp03_clip_easy.map": {
            "source": "machina_eval_exp03.map",
            "clip": {"germanium": 1},
            "seed": 45
        },
        "machina_eval_exp03_clip_medium.map": {
            "source": "machina_eval_exp03.map",
            "clip": {"germanium": 2, "carbon": 1},
            "seed": 46
        },
        "machina_eval_exp03_clip_hard.map": {
            "source": "machina_eval_exp03.map",
            "clip": {"germanium": 3, "oxygen": 1},
            "seed": 47
        },

        # SiliconWorkbench variants (exp04)
        "machina_eval_exp04_clip_easy.map": {
            "source": "machina_eval_exp04.map",
            "clip": {"silicon": 1},
            "seed": 48
        },
        "machina_eval_exp04_clip_medium.map": {
            "source": "machina_eval_exp04.map",
            "clip": {"silicon": 2, "carbon": 1},
            "seed": 49
        },
        "machina_eval_exp04_clip_hard.map": {
            "source": "machina_eval_exp04.map",
            "clip": {"silicon": 2, "oxygen": 1, "germanium": 1},
            "seed": 50
        },

        # CarbonDesert variants (exp05)
        "machina_eval_exp05_clip_easy.map": {
            "source": "machina_eval_exp05.map",
            "clip": {"carbon": 1},
            "seed": 51
        },
        "machina_eval_exp05_clip_medium.map": {
            "source": "machina_eval_exp05.map",
            "clip": {"carbon": 1, "oxygen": 1},
            "seed": 52
        },
        "machina_eval_exp05_clip_hard.map": {
            "source": "machina_eval_exp05.map",
            "clip": {"carbon": 2, "oxygen": 1, "silicon": 1},
            "seed": 53
        },

        # SingleUseWorld variants (exp06)
        "machina_eval_exp06_clip_easy.map": {
            "source": "machina_eval_exp06.map",
            "clip": {"oxygen": 1},
            "seed": 54
        },
        "machina_eval_exp06_clip_medium.map": {
            "source": "machina_eval_exp06.map",
            "clip": {"oxygen": 1, "carbon": 1},
            "seed": 55
        },
        "machina_eval_exp06_clip_hard.map": {
            "source": "machina_eval_exp06.map",
            "clip": {"oxygen": 1, "carbon": 1, "germanium": 1},
            "seed": 56
        },
    }

    print("Generating clipped map variants...")
    print("=" * 60)

    for output_name, config in configs.items():
        source_path = maps_dir / config["source"]
        output_path = maps_dir / output_name

        if not source_path.exists():
            print(f"⚠ Warning: Source map {source_path.name} not found, skipping")
            continue

        generate_clipped_map(
            source_path,
            output_path,
            config["clip"],
            config["seed"]
        )

    print("\n" + "=" * 60)
    print("✓ Clipped map generation complete!")


if __name__ == "__main__":
    main()

