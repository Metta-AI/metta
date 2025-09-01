#!/usr/bin/env python3
"""Quick script to check the generated corridor maps.

This provides a simple way to visualize what maps are being generated
by the navigation_with_corridors.py functions.
"""

from experiments.evals.navigation_with_corridors import (
    make_corridors_env,
    make_grid_maze_env,
    make_hard_sequence_env,
    make_radial_large_env,
    make_radial_mini_env,
    make_radial_small_env,
    visualize_env_map,
)


def print_full_ascii_map(grid, title: str):
    """Print the entire map without truncation."""
    print("\n" + "=" * 60)
    print(f"{title} ({grid.shape[0]}x{grid.shape[1]})")
    print("=" * 60)

    char_map = {
        "wall": "#",
        "empty": ".",
        "agent.agent": "@",
        "altar": "_",
    }

    for i in range(grid.shape[0]):
        line_chars = []
        for j in range(grid.shape[1]):
            cell = grid[i, j]
            line_chars.append(char_map.get(cell, "."))
        print("".join(line_chars))


def main():
    """Quick visualization of all procedural maps."""

    print("=" * 60)
    print("QUICK MAP CHECK - Procedural Corridor Generation")
    print("=" * 60)
    print("\nThis shows what maps are actually being generated for the")
    print("navigation evaluation suite.\n")

    # Define all the procedural maps
    maps = [
        ("Corridors Map", make_corridors_env),
        ("Radial Mini", make_radial_mini_env),
        ("Radial Small", make_radial_small_env),
        ("Radial Large", make_radial_large_env),
        ("Grid Maze", make_grid_maze_env),
        ("Hard Sequence", make_hard_sequence_env),
    ]

    # Visualize each one
    for name, env_func in maps:
        print(f"\n{'=' * 60}")
        # Build map but handle our own printing so there is no truncation
        grid = visualize_env_map(env_func, title=name, show_ascii=False)
        print_full_ascii_map(grid, name)

        # Print some statistics
        unique_values = set(grid.flatten())
        print(f"\nMap contains: {', '.join(v for v in unique_values if v != 'wall')}")

        # Count objects
        altar_count = (grid == "altar").sum()
        # Agents may be labeled as "agent.agent" in the grid
        agent_mask = (grid == "agent") | (grid == "agent.agent")
        agent_count = agent_mask.sum()
        print(f"Altars: {altar_count}, Agents: {agent_count}")

    print("\n" + "=" * 60)
    print("Quick check complete!")
    print("\nFor more detailed visualization with plots, run:")
    print("  python experiments/visualize_navigation_maps.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
