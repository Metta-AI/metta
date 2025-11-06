"""Vibe definitions for Cogs vs Clips game."""

import mettagrid.config.vibes

VIBES = [
    mettagrid.config.vibes.Vibe("😐", "default", category="emotion"),  # neutral
    # Resources
    mettagrid.config.vibes.Vibe("🔋", "charger", category="resource"),  # energy / charge
    mettagrid.config.vibes.Vibe("⚫", "carbon", category="resource"),  # C — Carbon
    mettagrid.config.vibes.Vibe("⚪", "oxygen", category="resource"),  # O — Oxygen
    mettagrid.config.vibes.Vibe("🟣", "germanium", category="resource"),  # G — Germanium
    mettagrid.config.vibes.Vibe("🟠", "silicon", category="resource"),  # S — Silicon
    mettagrid.config.vibes.Vibe("❤️", "heart", category="resource"),  # reward / health
    # Gear
    mettagrid.config.vibes.Vibe("⚙️", "gear", category="gear"),
    # Stations
    mettagrid.config.vibes.Vibe("⭐", "assembler", category="station"),
    mettagrid.config.vibes.Vibe("📦", "chest", category="station"),
    mettagrid.config.vibes.Vibe("⬛", "wall", category="station"),
    # Identity
    mettagrid.config.vibes.Vibe("📎", "paperclip", category="identity"),
    # Directions
    mettagrid.config.vibes.Vibe("⬆️", "up", category="navigation"),
    mettagrid.config.vibes.Vibe("⬇️", "down", category="navigation"),
    mettagrid.config.vibes.Vibe("⬅️", "left", category="navigation"),
    mettagrid.config.vibes.Vibe("➡️", "right", category="navigation"),
    mettagrid.config.vibes.Vibe("↗️", "up-right", category="navigation"),
    mettagrid.config.vibes.Vibe("↘️", "down-right", category="navigation"),
    mettagrid.config.vibes.Vibe("↙️", "down-left", category="navigation"),
    mettagrid.config.vibes.Vibe("↖️", "up-left", category="navigation"),
    mettagrid.config.vibes.Vibe("🔂", "rotate", category="navigation"),
    # --- Tier 4: Combat / Tools / Economy ---
    mettagrid.config.vibes.Vibe("⚔️", "swords"),
    mettagrid.config.vibes.Vibe("🛡️", "shield"),
    mettagrid.config.vibes.Vibe("🔧", "wrench"),
    mettagrid.config.vibes.Vibe("💰", "money"),
    mettagrid.config.vibes.Vibe("🏭", "factory"),
    mettagrid.config.vibes.Vibe("⚡", "lightning"),
    mettagrid.config.vibes.Vibe("🔥", "fire"),
    mettagrid.config.vibes.Vibe("💧", "water"),
    mettagrid.config.vibes.Vibe("🌳", "tree"),
    # --- Tier 5: Miscellaneous ---
    mettagrid.config.vibes.Vibe("🔃", "rotate-clockwise"),
    mettagrid.config.vibes.Vibe("🧭", "compass"),
    mettagrid.config.vibes.Vibe("📍", "pin"),
    mettagrid.config.vibes.Vibe("📌", "pushpin"),
    mettagrid.config.vibes.Vibe("💎", "diamond"),
    mettagrid.config.vibes.Vibe("🪙", "coin"),
    mettagrid.config.vibes.Vibe("🛢️", "oil"),
    mettagrid.config.vibes.Vibe("⛽", "fuel"),
    mettagrid.config.vibes.Vibe("🌾", "wheat"),
    mettagrid.config.vibes.Vibe("🌽", "corn"),
    mettagrid.config.vibes.Vibe("🥕", "carrot"),
    mettagrid.config.vibes.Vibe("🪨", "rock"),
    mettagrid.config.vibes.Vibe("⛰️", "mountain"),
    mettagrid.config.vibes.Vibe("🪵", "wood"),
    mettagrid.config.vibes.Vibe("🌊", "wave"),
    mettagrid.config.vibes.Vibe("🗡️", "dagger"),
    mettagrid.config.vibes.Vibe("🏹", "bow"),
    mettagrid.config.vibes.Vibe("🔨", "hammer"),
    mettagrid.config.vibes.Vibe("⚗️", "alembic"),
    mettagrid.config.vibes.Vibe("🧪", "test-tube"),
    mettagrid.config.vibes.Vibe("📦", "package"),
    mettagrid.config.vibes.Vibe("🎒", "backpack"),
    mettagrid.config.vibes.Vibe("0️⃣", "zero"),
    mettagrid.config.vibes.Vibe("1️⃣", "one"),
    mettagrid.config.vibes.Vibe("2️⃣", "two"),
    mettagrid.config.vibes.Vibe("3️⃣", "three"),
    mettagrid.config.vibes.Vibe("4️⃣", "four"),
    mettagrid.config.vibes.Vibe("5️⃣", "five"),
    mettagrid.config.vibes.Vibe("6️⃣", "six"),
    mettagrid.config.vibes.Vibe("7️⃣", "seven"),
    mettagrid.config.vibes.Vibe("8️⃣", "eight"),
    mettagrid.config.vibes.Vibe("9️⃣", "nine"),
    mettagrid.config.vibes.Vibe("🔟", "ten"),
    mettagrid.config.vibes.Vibe("#️⃣", "hash"),
    mettagrid.config.vibes.Vibe("*️⃣", "asterisk"),
    mettagrid.config.vibes.Vibe("➕", "plus"),
    mettagrid.config.vibes.Vibe("➖", "minus"),
    mettagrid.config.vibes.Vibe("✖️", "multiply"),
    mettagrid.config.vibes.Vibe("➗", "divide"),
    mettagrid.config.vibes.Vibe("💯", "hundred"),
    mettagrid.config.vibes.Vibe("🔢", "numbers"),
    mettagrid.config.vibes.Vibe("❤️", "red-heart"),
    mettagrid.config.vibes.Vibe("🧡", "orange-heart"),
    mettagrid.config.vibes.Vibe("💛", "yellow-heart"),
    mettagrid.config.vibes.Vibe("💚", "green-heart"),
    mettagrid.config.vibes.Vibe("💙", "blue-heart"),
    mettagrid.config.vibes.Vibe("💜", "purple-heart"),
    mettagrid.config.vibes.Vibe("🤍", "white-heart"),
    mettagrid.config.vibes.Vibe("🖤", "black-heart"),
    mettagrid.config.vibes.Vibe("🤎", "brown-heart"),
    mettagrid.config.vibes.Vibe("💕", "two-hearts"),
    mettagrid.config.vibes.Vibe("💖", "sparkling-heart"),
    mettagrid.config.vibes.Vibe("💗", "growing-heart"),
    mettagrid.config.vibes.Vibe("💘", "heart-arrow"),
    mettagrid.config.vibes.Vibe("💝", "heart-ribbon"),
    mettagrid.config.vibes.Vibe("💞", "revolving-hearts"),
    mettagrid.config.vibes.Vibe("💟", "heart-decoration"),
    mettagrid.config.vibes.Vibe("💔", "broken-heart"),
    mettagrid.config.vibes.Vibe("❣️", "heart-exclamation"),
    mettagrid.config.vibes.Vibe("💌", "love-letter"),
    mettagrid.config.vibes.Vibe("😀", "grinning"),
    mettagrid.config.vibes.Vibe("😃", "grinning-big-eyes"),
    mettagrid.config.vibes.Vibe("😄", "grinning-smiling-eyes"),
    mettagrid.config.vibes.Vibe("😁", "beaming"),
    mettagrid.config.vibes.Vibe("😊", "smiling"),
    mettagrid.config.vibes.Vibe("😇", "halo"),
    mettagrid.config.vibes.Vibe("😍", "heart-eyes"),
    mettagrid.config.vibes.Vibe("🤩", "star-struck"),
    mettagrid.config.vibes.Vibe("😘", "kiss"),
    mettagrid.config.vibes.Vibe("😂", "tears-of-joy"),
    mettagrid.config.vibes.Vibe("🤣", "rofl"),
    mettagrid.config.vibes.Vibe("😆", "squinting"),
    mettagrid.config.vibes.Vibe("😢", "crying"),
    mettagrid.config.vibes.Vibe("😭", "sobbing"),
    mettagrid.config.vibes.Vibe("😿", "crying-cat"),
    mettagrid.config.vibes.Vibe("😠", "angry"),
    mettagrid.config.vibes.Vibe("😡", "pouting"),
    mettagrid.config.vibes.Vibe("🤬", "swearing"),
    mettagrid.config.vibes.Vibe("😨", "fearful"),
    mettagrid.config.vibes.Vibe("😰", "anxious"),
    mettagrid.config.vibes.Vibe("🧐", "monocle"),
    mettagrid.config.vibes.Vibe("😕", "confused"),
    mettagrid.config.vibes.Vibe("😪", "sleepy"),
    mettagrid.config.vibes.Vibe("🥱", "yawning"),
    mettagrid.config.vibes.Vibe("🤤", "drooling"),
    mettagrid.config.vibes.Vibe("😋", "savoring"),
    mettagrid.config.vibes.Vibe("😏", "smirking"),
    mettagrid.config.vibes.Vibe("🙄", "rolling-eyes"),
    mettagrid.config.vibes.Vibe("🤡", "clown"),
    mettagrid.config.vibes.Vibe("👻", "ghost"),
    mettagrid.config.vibes.Vibe("🗿", "moai"),
    mettagrid.config.vibes.Vibe("☠️", "skull-crossbones"),
    mettagrid.config.vibes.Vibe("📈", "chart-up"),
    mettagrid.config.vibes.Vibe("📉", "chart-down"),
    mettagrid.config.vibes.Vibe("🚀", "rocket"),
    mettagrid.config.vibes.Vibe("🎯", "target"),
    mettagrid.config.vibes.Vibe("🔴", "red-circle"),
    mettagrid.config.vibes.Vibe("🟠", "orange-circle"),
    mettagrid.config.vibes.Vibe("🟡", "yellow-circle"),
    mettagrid.config.vibes.Vibe("🟢", "green-circle"),
    mettagrid.config.vibes.Vibe("🔵", "blue-circle"),
    mettagrid.config.vibes.Vibe("🟣", "purple-circle"),
    mettagrid.config.vibes.Vibe("🟤", "brown-circle"),
    mettagrid.config.vibes.Vibe("⚫", "black-circle"),
    mettagrid.config.vibes.Vibe("⚪", "white-circle"),
    mettagrid.config.vibes.Vibe("🟧", "orange-square"),
    mettagrid.config.vibes.Vibe("🟨", "yellow-square"),
    mettagrid.config.vibes.Vibe("🟪", "purple-square"),
    mettagrid.config.vibes.Vibe("🟫", "brown-square"),
    mettagrid.config.vibes.Vibe("⬜", "white-square"),
    mettagrid.config.vibes.Vibe("🔺", "red-triangle"),
    mettagrid.config.vibes.Vibe("🔷", "blue-diamond"),
    mettagrid.config.vibes.Vibe("🔹", "small-blue-diamond"),
    mettagrid.config.vibes.Vibe("🔌", "plug"),
    mettagrid.config.vibes.Vibe("✦", "sparkle"),
    mettagrid.config.vibes.Vibe("░", "light-shade"),
    mettagrid.config.vibes.Vibe("▒", "medium-shade"),
]

# Mapping from name to vibe ID for lookups
VIBE_BY_NAME: dict[str, mettagrid.config.vibes.Vibe] = {vibe.name: vibe for vibe in VIBES}
assert len(VIBE_BY_NAME) == len(VIBES), "Duplicate vibes found"


def search_vibes(query: str) -> list[tuple[int, mettagrid.config.vibes.Vibe]]:
    """Search for vibes matching a query string.

    Args:
        query: Search query (case-insensitive)

    Returns:
        List of (vibe_id, Vibe) tuples matching the query
    """
    query_lower = query.lower()
    return [(idx, vibe) for idx, vibe in enumerate(VIBES) if query_lower in vibe.name.lower()]
