"""Glyph definitions for Cogs vs Clips game."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Glyph:
    """A glyph with its symbol and searchable name."""

    symbol: str
    name: str


GLYPH_DATA = [
    # --- Tier 0: Core Starter Set (13 glyphs) ---
    Glyph("🙂", "default"),  # neutral
    Glyph("⬆️", "up"),  # directions
    Glyph("⬇️", "down"),
    Glyph("⬅️", "left"),
    Glyph("➡️", "right"),
    Glyph("😄", "happy"),  # positive emotion
    Glyph("😡", "angry"),  # conflict / aggression
    Glyph("❤️", "heart"),  # reward / health
    Glyph("🔋", "battery"),  # energy / charge
    Glyph("⚫", "carbon"),  # C — Carbon
    Glyph("⚪", "oxygen"),  # O — Oxygen
    Glyph("🟣", "germanium"),  # G — Germanium
    Glyph("🟠", "silicon"),  # S — Silicon
    # --- Tier 1: Identity and Team Culture ---
    Glyph("👽", "alien"),
    Glyph("🤖", "robot"),
    Glyph("🟩", "green-square"),
    Glyph("🟦", "blue-square"),
    Glyph("🟥", "red-square"),
    # --- Tier 2: Directional Nuance ---
    Glyph("↗️", "up-right"),
    Glyph("↘️", "down-right"),
    Glyph("↙️", "down-left"),
    Glyph("↖️", "up-left"),
    Glyph("🔄", "rotate"),
    # --- Tier 3: Expression Nuance ---
    Glyph("😢", "sad"),
    Glyph("🤔", "thinking"),
    Glyph("😱", "screaming"),
    Glyph("😎", "sunglasses"),
    Glyph("😴", "sleeping"),
    Glyph("👀", "eyes"),
    Glyph("✨", "sparkles"),
    Glyph("💀", "skull"),
    Glyph("💩", "poop"),
    # --- Tier 4: Combat / Tools / Economy ---
    Glyph("⚔️", "swords"),
    Glyph("🛡️", "shield"),
    Glyph("🔧", "wrench"),
    Glyph("⚙️", "gear"),
    Glyph("💰", "money"),
    Glyph("🏭", "factory"),
    Glyph("⚡", "lightning"),
    Glyph("🔥", "fire"),
    Glyph("💧", "water"),
    Glyph("🌳", "tree"),
    # --- Tier 5: Miscellaneous ---
    Glyph("🔃", "rotate-clockwise"),
    Glyph("🔂", "rotate-loop"),
    Glyph("🧭", "compass"),
    Glyph("📍", "pin"),
    Glyph("📌", "pushpin"),
    Glyph("💎", "diamond"),
    Glyph("🪙", "coin"),
    Glyph("🛢️", "oil"),
    Glyph("⛽", "fuel"),
    Glyph("🌾", "wheat"),
    Glyph("🌽", "corn"),
    Glyph("🥕", "carrot"),
    Glyph("🪨", "rock"),
    Glyph("⛰️", "mountain"),
    Glyph("🪵", "wood"),
    Glyph("🌊", "wave"),
    Glyph("🗡️", "dagger"),
    Glyph("🏹", "bow"),
    Glyph("🔨", "hammer"),
    Glyph("📎", "paperclip"),
    Glyph("⚗️", "alembic"),
    Glyph("🧪", "test-tube"),
    Glyph("📦", "package"),
    Glyph("🎒", "backpack"),
    Glyph("0️⃣", "zero"),
    Glyph("1️⃣", "one"),
    Glyph("2️⃣", "two"),
    Glyph("3️⃣", "three"),
    Glyph("4️⃣", "four"),
    Glyph("5️⃣", "five"),
    Glyph("6️⃣", "six"),
    Glyph("7️⃣", "seven"),
    Glyph("8️⃣", "eight"),
    Glyph("9️⃣", "nine"),
    Glyph("🔟", "ten"),
    Glyph("#️⃣", "hash"),
    Glyph("*️⃣", "asterisk"),
    Glyph("➕", "plus"),
    Glyph("➖", "minus"),
    Glyph("✖️", "multiply"),
    Glyph("➗", "divide"),
    Glyph("💯", "hundred"),
    Glyph("🔢", "numbers"),
    Glyph("❤️", "red-heart"),
    Glyph("🧡", "orange-heart"),
    Glyph("💛", "yellow-heart"),
    Glyph("💚", "green-heart"),
    Glyph("💙", "blue-heart"),
    Glyph("💜", "purple-heart"),
    Glyph("🤍", "white-heart"),
    Glyph("🖤", "black-heart"),
    Glyph("🤎", "brown-heart"),
    Glyph("💕", "two-hearts"),
    Glyph("💖", "sparkling-heart"),
    Glyph("💗", "growing-heart"),
    Glyph("💘", "heart-arrow"),
    Glyph("💝", "heart-ribbon"),
    Glyph("💞", "revolving-hearts"),
    Glyph("💟", "heart-decoration"),
    Glyph("💔", "broken-heart"),
    Glyph("❣️", "heart-exclamation"),
    Glyph("💌", "love-letter"),
    Glyph("😀", "grinning"),
    Glyph("😃", "grinning-big-eyes"),
    Glyph("😄", "grinning-smiling-eyes"),
    Glyph("😁", "beaming"),
    Glyph("😊", "smiling"),
    Glyph("😇", "halo"),
    Glyph("😍", "heart-eyes"),
    Glyph("🤩", "star-struck"),
    Glyph("😘", "kiss"),
    Glyph("😂", "tears-of-joy"),
    Glyph("🤣", "rofl"),
    Glyph("😆", "squinting"),
    Glyph("😢", "crying"),
    Glyph("😭", "sobbing"),
    Glyph("😿", "crying-cat"),
    Glyph("😠", "angry"),
    Glyph("😡", "pouting"),
    Glyph("🤬", "swearing"),
    Glyph("😨", "fearful"),
    Glyph("😰", "anxious"),
    Glyph("🧐", "monocle"),
    Glyph("😕", "confused"),
    Glyph("😪", "sleepy"),
    Glyph("🥱", "yawning"),
    Glyph("🤤", "drooling"),
    Glyph("😋", "savoring"),
    Glyph("😏", "smirking"),
    Glyph("🙄", "rolling-eyes"),
    Glyph("🤡", "clown"),
    Glyph("👻", "ghost"),
    Glyph("🗿", "moai"),
    Glyph("☠️", "skull-crossbones"),
    Glyph("📈", "chart-up"),
    Glyph("📉", "chart-down"),
    Glyph("🚀", "rocket"),
    Glyph("🎯", "target"),
    Glyph("⭐", "star"),
    Glyph("🔴", "red-circle"),
    Glyph("🟠", "orange-circle"),
    Glyph("🟡", "yellow-circle"),
    Glyph("🟢", "green-circle"),
    Glyph("🔵", "blue-circle"),
    Glyph("🟣", "purple-circle"),
    Glyph("🟤", "brown-circle"),
    Glyph("⚫", "black-circle"),
    Glyph("⚪", "white-circle"),
    Glyph("🟧", "orange-square"),
    Glyph("🟨", "yellow-square"),
    Glyph("🟪", "purple-square"),
    Glyph("🟫", "brown-square"),
    Glyph("⬛", "black-square"),
    Glyph("⬜", "white-square"),
    Glyph("🔺", "red-triangle"),
    Glyph("🔷", "blue-diamond"),
    Glyph("🔹", "small-blue-diamond"),
    Glyph("🔌", "plug"),
    Glyph("✦", "sparkle"),
    Glyph("░", "light-shade"),
    Glyph("▒", "medium-shade"),
]

# Use only Tier 0 glyphs for now.
GLYPH_DATA = GLYPH_DATA[:13]

# For backward compatibility - list of just symbols
GLYPHS = [glyph.symbol for glyph in GLYPH_DATA]

# Mapping from name to glyph ID for lookups
GLYPH_NAMES = {glyph.name: idx for idx, glyph in enumerate(GLYPH_DATA)}


def find_glyph_by_name(name: str) -> int | None:
    """Find glyph ID by name (case-insensitive).

    Args:
        name: Glyph name to search for

    Returns:
        Glyph ID if found, None otherwise
    """
    return GLYPH_NAMES.get(name.lower())


def search_glyphs(query: str) -> list[tuple[int, Glyph]]:
    """Search for glyphs matching a query string.

    Args:
        query: Search query (case-insensitive)

    Returns:
        List of (glyph_id, Glyph) tuples matching the query
    """
    query_lower = query.lower()
    return [(idx, glyph) for idx, glyph in enumerate(GLYPH_DATA) if query_lower in glyph.name.lower()]
