"""Glyph definitions for Cogs vs Clips game."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Glyph:
    """A glyph with its symbol and searchable name."""

    symbol: str
    name: str


GLYPH_DATA = [
    Glyph("🙂", "default"),
    Glyph("⬆️", "up"),
    Glyph("⬇️", "down"),
    Glyph("⬅️", "left"),
    Glyph("➡️", "right"),
    Glyph("↗️", "up-right"),
    Glyph("↘️", "down-right"),
    Glyph("↙️", "down-left"),
    Glyph("↖️", "up-left"),
    Glyph("🔄", "rotate"),
    # Glyph("🔃", "rotate-clockwise"),
    # Glyph("🔂", "rotate-loop"),
    # Glyph("🧭", "compass"),
    # Glyph("📍", "pin"),
    # Glyph("📌", "pushpin"),
    # Glyph("💎", "diamond"),
    # Glyph("💰", "money"),
    # Glyph("🪙", "coin"),
    # Glyph("⚡", "lightning"),
    # Glyph("🔋", "battery"),
    # Glyph("🛢️", "oil"),
    # Glyph("⛽", "fuel"),
    # Glyph("🌾", "wheat"),
    # Glyph("🌽", "corn"),
    # Glyph("🥕", "carrot"),
    # Glyph("🪨", "rock"),
    # Glyph("⛰️", "mountain"),
    # Glyph("🪵", "wood"),
    # Glyph("🌳", "tree"),
    # Glyph("💧", "water"),
    # Glyph("🌊", "wave"),
    # Glyph("🔥", "fire"),
    # Glyph("⚔️", "swords"),
    # Glyph("🗡️", "dagger"),
    # Glyph("🏹", "bow"),
    # Glyph("🛡️", "shield"),
    # Glyph("🔧", "wrench"),
    # Glyph("🔨", "hammer"),
    # Glyph("⚙️", "gear"),
    # Glyph("📎", "paperclip"),
    # Glyph("⚗️", "alembic"),
    # Glyph("🧪", "test-tube"),
    # Glyph("📦", "package"),
    # Glyph("🎒", "backpack"),
    # Glyph("🏭", "factory"),
    # Glyph("0️⃣", "zero"),
    # Glyph("1️⃣", "one"),
    # Glyph("2️⃣", "two"),
    # Glyph("3️⃣", "three"),
    # Glyph("4️⃣", "four"),
    # Glyph("5️⃣", "five"),
    # Glyph("6️⃣", "six"),
    # Glyph("7️⃣", "seven"),
    # Glyph("8️⃣", "eight"),
    # Glyph("9️⃣", "nine"),
    # Glyph("🔟", "ten"),
    # Glyph("#️⃣", "hash"),
    # Glyph("*️⃣", "asterisk"),
    # Glyph("➕", "plus"),
    # Glyph("➖", "minus"),
    # Glyph("✖️", "multiply"),
    # Glyph("➗", "divide"),
    # Glyph("💯", "hundred"),
    # Glyph("🔢", "numbers"),
    # Glyph("❤️", "red-heart"),
    # Glyph("🧡", "orange-heart"),
    # Glyph("💛", "yellow-heart"),
    # Glyph("💚", "green-heart"),
    # Glyph("💙", "blue-heart"),
    # Glyph("💜", "purple-heart"),
    # Glyph("🤍", "white-heart"),
    # Glyph("🖤", "black-heart"),
    # Glyph("🤎", "brown-heart"),
    # Glyph("💕", "two-hearts"),
    # Glyph("💖", "sparkling-heart"),
    # Glyph("💗", "growing-heart"),
    # Glyph("💘", "heart-arrow"),
    # Glyph("💝", "heart-ribbon"),
    # Glyph("💞", "revolving-hearts"),
    # Glyph("💟", "heart-decoration"),
    # Glyph("💔", "broken-heart"),
    # Glyph("❣️", "heart-exclamation"),
    # Glyph("💌", "love-letter"),
    # Glyph("😀", "grinning"),
    # Glyph("😃", "grinning-big-eyes"),
    # Glyph("😄", "grinning-smiling-eyes"),
    # Glyph("😁", "beaming"),
    # Glyph("😊", "smiling"),
    # Glyph("😇", "halo"),
    # Glyph("😍", "heart-eyes"),
    # Glyph("🤩", "star-struck"),
    # Glyph("😘", "kiss"),
    # Glyph("😂", "tears-of-joy"),
    # Glyph("🤣", "rofl"),
    # Glyph("😆", "squinting"),
    # Glyph("😢", "crying"),
    # Glyph("😭", "sobbing"),
    # Glyph("😿", "crying-cat"),
    # Glyph("😠", "angry"),
    # Glyph("😡", "pouting"),
    # Glyph("🤬", "swearing"),
    # Glyph("😨", "fearful"),
    # Glyph("😰", "anxious"),
    # Glyph("😱", "screaming"),
    # Glyph("🤔", "thinking"),
    # Glyph("🧐", "monocle"),
    # Glyph("😕", "confused"),
    # Glyph("😴", "sleeping"),
    # Glyph("😪", "sleepy"),
    # Glyph("🥱", "yawning"),
    # Glyph("🤤", "drooling"),
    # Glyph("😋", "savoring"),
    # Glyph("😎", "sunglasses"),
    # Glyph("😏", "smirking"),
    # Glyph("🙄", "rolling-eyes"),
    # Glyph("🤡", "clown"),
    # Glyph("🤖", "robot"),
    # Glyph("👻", "ghost"),
    # Glyph("🗿", "moai"),
    # Glyph("👀", "eyes"),
    # Glyph("💀", "skull"),
    # Glyph("☠️", "skull-crossbones"),
    # Glyph("📈", "chart-up"),
    # Glyph("📉", "chart-down"),
    # Glyph("🚀", "rocket"),
    # Glyph("🎯", "target"),
    # Glyph("⭐", "star"),
    # Glyph("✨", "sparkles"),
    # Glyph("💩", "poop"),
    # Glyph("🔴", "red-circle"),
    # Glyph("🟠", "orange-circle"),
    # Glyph("🟡", "yellow-circle"),
    # Glyph("🟢", "green-circle"),
    # Glyph("🔵", "blue-circle"),
    # Glyph("🟣", "purple-circle"),
    # Glyph("🟤", "brown-circle"),
    # Glyph("⚫", "black-circle"),
    # Glyph("⚪", "white-circle"),
    # Glyph("🟥", "red-square"),
    # Glyph("🟧", "orange-square"),
    # Glyph("🟨", "yellow-square"),
    # Glyph("🟩", "green-square"),
    # Glyph("🟦", "blue-square"),
    # Glyph("🟪", "purple-square"),
    # Glyph("🟫", "brown-square"),
    # Glyph("⬛", "black-square"),
    # Glyph("⬜", "white-square"),
    # Glyph("🔺", "red-triangle"),
    # Glyph("🔷", "blue-diamond"),
    # Glyph("🔹", "small-blue-diamond"),
    # Glyph("🔌", "plug"),
    # Glyph("✦", "sparkle"),
    # Glyph("░", "light-shade"),
    # Glyph("▒", "medium-shade"),
]

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
