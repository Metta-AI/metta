#!/usr/bin/env python3
"""Visualization tool for maps generated in navigation_with_corridors.py

This script generates and displays the procedural corridor maps used in the
navigation evaluation suite, allowing visual inspection of the generated patterns.
"""

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from metta.mettagrid.mapgen.mapgen import MapGen
from metta.mettagrid.mettagrid_config import EnvConfig

from experiments.evals.navigation_with_corridors import (
    make_corridors_env,
    make_grid_maze_env,
    make_radial_large_env,
    make_radial_mini_env,
    make_radial_small_env,
)


def extract_and_build_map(env_config: EnvConfig) -> np.ndarray:
    """Extract map configuration and build the actual map."""
    # Get the instance map configuration
    instance_map_config = env_config.game.map_builder.instance_map

    # Build the map using MapGen
    map_gen = MapGen(instance_map_config)
    game_map = map_gen.build()

    return game_map.grid


def visualize_map(
    grid: np.ndarray,
    title: str = "Map",
    figsize: Tuple[int, int] = (10, 10),
    show_grid: bool = False,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Visualize a map grid using matplotlib.

    Args:
        grid: The map grid array
        title: Title for the plot
        figsize: Figure size (width, height)
        show_grid: Whether to show grid lines
        save_path: Optional path to save the figure

    Returns:
        The matplotlib figure
    """
    # Create color mapping
    color_map = {
        "wall": 0,
        "empty": 1,
        "agent.agent": 2,
        "altar": 3,
        "heart": 4,
        "sword": 5,
        "shield": 6,
    }

    # Convert string grid to numeric
    numeric_grid = np.zeros_like(grid, dtype=int)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            cell = grid[i, j]
            numeric_grid[i, j] = color_map.get(cell, 1)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Define colors
    colors = ["black", "white", "green", "gold", "red", "blue", "purple"]
    cmap = plt.matplotlib.colors.ListedColormap(colors[: len(color_map)])

    # Plot the map
    ax.imshow(numeric_grid, cmap=cmap, interpolation="nearest")

    # Set title and labels
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(f"Width: {grid.shape[1]}")
    ax.set_ylabel(f"Height: {grid.shape[0]}")

    # Optional grid
    if show_grid:
        ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
        ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5)
        ax.tick_params(which="minor", size=0)

    # Add legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, fc="black", label="Wall"),
        plt.Rectangle((0, 0), 1, 1, fc="white", label="Empty"),
        plt.Rectangle((0, 0), 1, 1, fc="green", label="Agent (@)"),
        plt.Rectangle((0, 0), 1, 1, fc="gold", label="Altar (_)"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(1, 1))

    plt.tight_layout()

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")

    return fig


def print_ascii_map(
    grid: np.ndarray, title: str, max_width: int = 80, max_height: int = 40
):
    """Print an ASCII representation of the map.

    Args:
        grid: The map grid array
        title: Title for the map
        max_width: Maximum width to display
        max_height: Maximum height to display
    """
    print(f"\n{'=' * 60}")
    print(f"{title} ({grid.shape[0]}x{grid.shape[1]})")
    print("=" * 60)

    # Convert to ASCII characters
    char_map = {
        "wall": "#",
        "empty": ".",
        "agent.agent": "@",
        "altar": "_",
        "heart": "♥",
        "sword": "†",
        "shield": "○",
    }

    # Print the map
    for i, row in enumerate(grid):
        if i >= max_height:
            print(f"... ({grid.shape[0] - max_height} more rows)")
            break

        line = ""
        for j, cell in enumerate(row):
            if j >= max_width:
                line += "..."
                break
            line += char_map.get(cell, "?")
        print(line)


def visualize_all_maps(show_plots: bool = True, save_plots: bool = False):
    """Generate and visualize all procedural corridor maps.

    Args:
        show_plots: Whether to display plots interactively
        save_plots: Whether to save plots to files
    """
    print("=" * 60)
    print("VISUALIZING PROCEDURAL CORRIDOR MAPS")
    print("=" * 60)

    # Define maps to visualize
    maps_to_generate = [
        ("Corridors Pattern", make_corridors_env, "corridors.png"),
        ("Radial Mini Pattern", make_radial_mini_env, "radial_mini.png"),
        ("Radial Small Pattern", make_radial_small_env, "radial_small.png"),
        ("Radial Large Pattern", make_radial_large_env, "radial_large.png"),
        ("Grid Maze Pattern", make_grid_maze_env, "grid_maze.png"),
    ]

    figures = []

    for title, env_func, save_name in maps_to_generate:
        print(f"\nGenerating: {title}")

        # Generate the environment config
        env_config = env_func()

        # Extract and build the map
        grid = extract_and_build_map(env_config)

        # Print ASCII version
        print_ascii_map(grid, title)

        # Create visualization
        save_path = save_name if save_plots else None
        fig = visualize_map(grid, title, save_path=save_path)
        figures.append(fig)

        # Print stats
        unique, counts = np.unique(grid, return_counts=True)
        print("\nMap statistics:")
        for val, count in zip(unique, counts):
            if val != "wall":
                print(f"  {val:10s}: {count:4d} cells")

    if show_plots:
        plt.show()

    return figures


def compare_with_ascii(ascii_map_name: str, procedural_env_func):
    """Compare an ASCII map with its procedural equivalent.

    Args:
        ascii_map_name: Name of the ASCII map file (without .map extension)
        procedural_env_func: Function that generates the procedural equivalent
    """
    import os

    print(f"\n{'=' * 60}")
    print(f"COMPARING: {ascii_map_name}")
    print("=" * 60)

    # Load ASCII map if it exists
    ascii_path = f"mettagrid/configs/maps/navigation/{ascii_map_name}.map"
    if os.path.exists(ascii_path):
        print(f"\nOriginal ASCII map from {ascii_path}:")
        with open(ascii_path, "r") as f:
            lines = f.readlines()
            for i, line in enumerate(lines[:40]):  # Show first 40 lines
                print(line.rstrip())
                if i == 39 and len(lines) > 40:
                    print(f"... ({len(lines) - 40} more lines)")
    else:
        print(f"ASCII map not found at {ascii_path}")

    # Generate procedural version
    print("\nProcedurally generated equivalent:")
    env_config = procedural_env_func()
    grid = extract_and_build_map(env_config)
    print_ascii_map(grid, f"Procedural {ascii_map_name}")

    # Create side-by-side visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

    # If ASCII exists, try to load and display it
    if os.path.exists(ascii_path):
        ax1.text(
            0.5,
            0.5,
            "ASCII Map\n(See console output)",
            ha="center",
            va="center",
            fontsize=12,
        )
        ax1.set_title(f"Original ASCII: {ascii_map_name}")
        ax1.axis("off")

    # Display procedural version
    color_map = {"wall": 0, "empty": 1, "agent": 2, "altar": 3}
    numeric_grid = np.zeros_like(grid, dtype=int)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            numeric_grid[i, j] = color_map.get(grid[i, j], 1)

    colors = ["black", "white", "green", "gold"]
    cmap = plt.matplotlib.colors.ListedColormap(colors)
    ax2.imshow(numeric_grid, cmap=cmap, interpolation="nearest")
    ax2.set_title(f"Procedural: {ascii_map_name}")
    ax2.set_xlabel(f"Size: {grid.shape[1]}x{grid.shape[0]}")

    plt.suptitle(f"Comparison: {ascii_map_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    return fig


def main():
    """Main function to run visualizations."""
    import argparse

    parser = argparse.ArgumentParser(description="Visualize procedural corridor maps")
    parser.add_argument("--save", action="store_true", help="Save plots to files")
    parser.add_argument(
        "--no-show", action="store_true", help="Do not show plots interactively"
    )
    parser.add_argument(
        "--compare", action="store_true", help="Compare with ASCII versions"
    )
    parser.add_argument(
        "--map",
        type=str,
        help="Specific map to visualize (corridors, radial_mini, etc.)",
    )

    args = parser.parse_args()

    if args.compare:
        # Compare specific maps
        print("\nComparing procedural maps with ASCII originals...")
        compare_with_ascii("corridors", make_corridors_env)
        compare_with_ascii("radial_mini", make_radial_mini_env)
        if not args.no_show:
            plt.show()
    elif args.map:
        # Visualize specific map
        map_funcs = {
            "corridors": make_corridors_env,
            "radial_mini": make_radial_mini_env,
            "radial_small": make_radial_small_env,
            "radial_large": make_radial_large_env,
            "grid_maze": make_grid_maze_env,
        }

        if args.map in map_funcs:
            print(f"\nVisualizing {args.map}...")
            env_config = map_funcs[args.map]()
            grid = extract_and_build_map(env_config)
            print_ascii_map(grid, args.map.replace("_", " ").title())

            visualize_map(grid, args.map.replace("_", " ").title())
            if args.save:
                plt.savefig(f"{args.map}.png", dpi=150, bbox_inches="tight")
                print(f"Saved to {args.map}.png")
            if not args.no_show:
                plt.show()
        else:
            print(f"Unknown map: {args.map}")
            print(f"Available: {', '.join(map_funcs.keys())}")
    else:
        # Visualize all maps
        visualize_all_maps(show_plots=not args.no_show, save_plots=args.save)

    print("\n" + "=" * 60)
    print("Visualization complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
