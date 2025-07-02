#!/usr/bin/env -S uv run
"""Demonstrate the 12 ordered pairs for progressive forgetting experiments."""

import itertools


def main():
    """Show all 12 ordered pairs for the 4 task sets."""
    task_sets = ["navigation", "memory", "navigation_sequence", "object_use"]

    print("Progressive Forgetting Curriculum - Ordered Pairs")
    print("=" * 60)
    print()
    print("Task Sets:")
    for i, task_set in enumerate(task_sets, 1):
        print(f"  {i}. {task_set}")
    print()

    # Generate all ordered pairs
    ordered_pairs = list(itertools.permutations(task_sets, 2))

    print(f"Total Ordered Pairs: {len(ordered_pairs)}")
    print()
    print("All Ordered Pairs (Order Matters!):")
    print("-" * 40)

    for i, (task_set_1, task_set_2) in enumerate(ordered_pairs, 1):
        print(f"{i:2d}. {task_set_1:20s} → {task_set_2}")

    print()
    print("Why Order Matters:")
    print("-" * 20)
    print("• navigation → object_use: Train on navigation first, then switch to object use")
    print("• object_use → navigation: Train on object use first, then switch to navigation")
    print()
    print("These will produce DIFFERENT forgetting patterns because:")
    print("• The first task set establishes the agent's initial knowledge")
    print("• The second task set may interfere with or build upon that knowledge")
    print("• Different task orders can lead to different transfer learning effects")
    print()
    print("Example: Navigation skills might transfer better to object use")
    print("than object use skills transfer to navigation.")


if __name__ == "__main__":
    main()
