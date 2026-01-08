"""Shared test utilities for mettagrid mapgen scenes tests."""

from mettagrid.test_support.mapgen import assert_grid_map


def assert_grid(scene, ascii_grid: str, char_to_name: dict[str, str] | None = None):
    """Wrapper around assert_grid_map that automatically provides default character mappings."""
    # Default char_to_name mapping for common ASCII characters
    default_char_to_name = {
        "#": "wall",
        "_": "assembler",
        ".": "empty",
        "@": "agent.agent",
        "n": "npc",
        "m": "monster",
        " ": "empty",
    }

    # Merge with provided char_to_name
    if char_to_name:
        final_char_to_name = {**default_char_to_name, **char_to_name}
    else:
        final_char_to_name = default_char_to_name

    # Call the original assert_grid_map function
    assert_grid_map(scene, ascii_grid, final_char_to_name)
