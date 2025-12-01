"""Vibe definitions for Cogs vs Clips game."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Vibe:
    """A vibe with its symbol and searchable name."""

    symbol: str
    name: str
    category: str = "misc"


TRAINING_VIBES = [
    Vibe("😐", "default", category="emotion"),  # neutral
    Vibe("🔋", "charger", category="resource"),  # energy / charge
    Vibe("⚫", "carbon_a", category="resource"),  # C — Carbon
    Vibe("⬛", "carbon_b", category="resource"),
    Vibe("⚪", "oxygen_a", category="resource"),  # O — Oxygen
    Vibe("⬜", "oxygen_b", category="resource"),
    Vibe("🟣", "germanium_a", category="resource"),  # G — Germanium
    Vibe("🟪", "germanium_b", category="resource"),
    Vibe("🟠", "silicon_a", category="resource"),  # S — Silicon
    Vibe("🟧", "silicon_b", category="resource"),
    Vibe("❤️", "heart_a", category="resource"),  # reward / health
    Vibe("💟", "heart_b", category="resource"),
    Vibe("⚙️", "gear", category="gear"),
    Vibe("⭐", "assembler", category="station"),
    Vibe("📦", "chest", category="station"),
    # Vibe("⬛", "wall", category="station"),
    Vibe("❤️", "red-heart"),
]

VIBES = [
    Vibe("😐", "default", category="emotion"),  # neutral
    # Resources
    Vibe("🔋", "charger", category="resource"),  # energy / charge
    Vibe("⚫", "carbon_a", category="resource"),  # C — Carbon
    Vibe("⬛", "carbon_b", category="resource"),
    Vibe("⚪", "oxygen_a", category="resource"),  # O — Oxygen
    Vibe("⬜", "oxygen_b", category="resource"),
    Vibe("🟣", "germanium_a", category="resource"),  # G — Germanium
    Vibe("🟪", "germanium_b", category="resource"),
    Vibe("🟠", "silicon_a", category="resource"),  # S — Silicon
    Vibe("🟧", "silicon_b", category="resource"),
    Vibe("❤️", "heart_a", category="resource"),  # reward / health
    Vibe("💟", "heart_b", category="resource"),
    # Gear
    Vibe("⚙️", "gear", category="gear"),
    # Stations
    Vibe("⭐", "assembler", category="station"),
    Vibe("📦", "chest", category="station"),
    Vibe("⬛", "wall", category="station"),
    # Identity
    Vibe("📎", "paperclip", category="identity"),
    # Directions
    Vibe("⬆️", "up", category="navigation"),
    Vibe("⬇️", "down", category="navigation"),
    Vibe("⬅️", "left", category="navigation"),
    Vibe("➡️", "right", category="navigation"),
    Vibe("↗️", "up-right", category="navigation"),
    Vibe("↘️", "down-right", category="navigation"),
    Vibe("↙️", "down-left", category="navigation"),
    Vibe("↖️", "up-left", category="navigation"),
    Vibe("🔂", "rotate", category="navigation"),
    # --- Tier 4: Combat / Tools / Economy ---
    Vibe("⚔️", "swords"),
    Vibe("🛡️", "shield"),
    Vibe("🔧", "wrench"),
    Vibe("💰", "money"),
    Vibe("🏭", "factory"),
    Vibe("⚡", "lightning"),
    Vibe("🔥", "fire"),
    Vibe("💧", "water"),
    Vibe("🌳", "tree"),
    # --- Tier 5: Miscellaneous ---
    Vibe("🔃", "rotate-clockwise"),
    Vibe("🧭", "compass"),
    Vibe("📍", "pin"),
    Vibe("📌", "pushpin"),
    Vibe("💎", "diamond"),
    Vibe("🪙", "coin"),
    Vibe("🛢️", "oil"),
    Vibe("⛽", "fuel"),
    Vibe("🌾", "wheat"),
    Vibe("🌽", "corn"),
    Vibe("🥕", "carrot"),
    Vibe("🪨", "rock"),
    Vibe("⛰️", "mountain"),
    Vibe("🪵", "wood"),
    Vibe("🌊", "wave"),
    Vibe("🗡️", "dagger"),
    Vibe("🏹", "bow"),
    Vibe("🔨", "hammer"),
    Vibe("⚗️", "alembic"),
    Vibe("🧪", "test-tube"),
    Vibe("📦", "package"),
    Vibe("🎒", "backpack"),
    Vibe("0️⃣", "zero"),
    Vibe("1️⃣", "one"),
    Vibe("2️⃣", "two"),
    Vibe("3️⃣", "three"),
    Vibe("4️⃣", "four"),
    Vibe("5️⃣", "five"),
    Vibe("6️⃣", "six"),
    Vibe("7️⃣", "seven"),
    Vibe("8️⃣", "eight"),
    Vibe("9️⃣", "nine"),
    Vibe("🔟", "ten"),
    Vibe("#️⃣", "hash"),
    Vibe("*️⃣", "asterisk"),
    Vibe("➕", "plus"),
    Vibe("➖", "minus"),
    Vibe("✖️", "multiply"),
    Vibe("➗", "divide"),
    Vibe("💯", "hundred"),
    Vibe("🔢", "numbers"),
    Vibe("❤️", "red-heart"),
    Vibe("🧡", "orange-heart"),
    Vibe("💛", "yellow-heart"),
    Vibe("💚", "green-heart"),
    Vibe("💙", "blue-heart"),
    Vibe("💜", "purple-heart"),
    Vibe("🤍", "white-heart"),
    Vibe("🖤", "black-heart"),
    Vibe("🤎", "brown-heart"),
    Vibe("💕", "two-hearts"),
    Vibe("💖", "sparkling-heart"),
    Vibe("💗", "growing-heart"),
    Vibe("💘", "heart-arrow"),
    Vibe("💝", "heart-ribbon"),
    Vibe("💞", "revolving-hearts"),
    Vibe("💟", "heart-decoration"),
    Vibe("💔", "broken-heart"),
    Vibe("❣️", "heart-exclamation"),
    Vibe("💌", "love-letter"),
    Vibe("😀", "grinning"),
    Vibe("😃", "grinning-big-eyes"),
    Vibe("😄", "grinning-smiling-eyes"),
    Vibe("😁", "beaming"),
    Vibe("😊", "smiling"),
    Vibe("😇", "halo"),
    Vibe("😍", "heart-eyes"),
    Vibe("🤩", "star-struck"),
    Vibe("😘", "kiss"),
    Vibe("😂", "tears-of-joy"),
    Vibe("🤣", "rofl"),
    Vibe("😆", "squinting"),
    Vibe("😢", "crying"),
    Vibe("😭", "sobbing"),
    Vibe("😿", "crying-cat"),
    Vibe("😠", "angry"),
    Vibe("😡", "pouting"),
    Vibe("🤬", "swearing"),
    Vibe("😨", "fearful"),
    Vibe("😰", "anxious"),
    Vibe("🧐", "monocle"),
    Vibe("😕", "confused"),
    Vibe("😪", "sleepy"),
    Vibe("🥱", "yawning"),
    Vibe("🤤", "drooling"),
    Vibe("😋", "savoring"),
    Vibe("😏", "smirking"),
    Vibe("🙄", "rolling-eyes"),
    Vibe("🤡", "clown"),
    Vibe("👻", "ghost"),
    Vibe("🗿", "moai"),
    Vibe("☠️", "skull-crossbones"),
    Vibe("📈", "chart-up"),
    Vibe("📉", "chart-down"),
    Vibe("🚀", "rocket"),
    Vibe("🎯", "target"),
    Vibe("🔴", "red-circle"),
    Vibe("🟠", "orange-circle"),
    Vibe("🟡", "yellow-circle"),
    Vibe("🟢", "green-circle"),
    Vibe("🔵", "blue-circle"),
    Vibe("🟣", "purple-circle"),
    Vibe("🟤", "brown-circle"),
    Vibe("⚫", "black-circle"),
    Vibe("⚪", "white-circle"),
    Vibe("🟧", "orange-square"),
    Vibe("🟨", "yellow-square"),
    Vibe("🟪", "purple-square"),
    Vibe("🟫", "brown-square"),
    Vibe("⬜", "white-square"),
    Vibe("🔺", "red-triangle"),
    Vibe("🔷", "blue-diamond"),
    Vibe("🔹", "small-blue-diamond"),
    Vibe("🔌", "plug"),
    Vibe("✦", "sparkle"),
    Vibe("░", "light-shade"),
    Vibe("▒", "medium-shade"),
]

# Mapping from name to vibe ID for lookups
VIBE_BY_NAME: dict[str, Vibe] = {vibe.name: vibe for vibe in VIBES}
assert len(VIBE_BY_NAME) == len(VIBES), "Duplicate vibes found"


def search_vibes(query: str) -> list[tuple[int, Vibe]]:
    """Search for vibes matching a query string.

    Args:
        query: Search query (case-insensitive)

    Returns:
        List of (vibe_id, Vibe) tuples matching the query
    """
    query_lower = query.lower()
    return [(idx, vibe) for idx, vibe in enumerate(VIBES) if query_lower in vibe.name.lower()]
