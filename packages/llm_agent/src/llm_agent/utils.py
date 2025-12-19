"""Utility functions for LLM agent."""

# Extractor types for resource gathering
EXTRACTOR_TYPES = frozenset(
    {
        "carbon_extractor",
        "oxygen_extractor",
        "germanium_extractor",
        "silicon_extractor",
    }
)

# Mapping from resource name to extractor type
RESOURCE_TO_EXTRACTOR = {
    "carbon": "carbon_extractor",
    "oxygen": "oxygen_extractor",
    "germanium": "germanium_extractor",
    "silicon": "silicon_extractor",
}


def pos_to_dir(x: int, y: int, verbose: bool = False) -> str:
    """Format position as direction from origin.

    Args:
        x: X coordinate (positive = East, negative = West)
        y: Y coordinate (positive = South, negative = North)
        verbose: If True, use verbose format like "3 tiles North and 2 tiles East of origin"

    Returns:
        Direction string like "3N2E" or "origin" (or verbose equivalent)
    """
    if x == 0 and y == 0:
        return "origin (starting point)" if verbose else "origin"

    parts = []
    if y < 0:
        parts.append((abs(y), "North", "N"))
    elif y > 0:
        parts.append((y, "South", "S"))
    if x > 0:
        parts.append((x, "East", "E"))
    elif x < 0:
        parts.append((abs(x), "West", "W"))

    if verbose:
        return " and ".join(f"{n} tiles {d}" for n, d, _ in parts) + " of origin"
    else:
        return "".join(f"{n}{s}" for n, _, s in parts)
