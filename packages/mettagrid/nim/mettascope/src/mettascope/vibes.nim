## Vibe panel allows you to set vibe frequency for the agent.
## Vibe are emoji like symbols that the agent can use to communicate with
## the world and other agents.

import
  fidget2,
  common, panels

type
  Vibe* = object
    name*: string
    symbol*: string

proc Glyph(symbol: string, name: string): Vibe =
  Vibe(symbol: symbol, name: name)

let vibes*: seq[Vibe] = @[
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
  # Glyph("↗️", "up-right"),
  # Glyph("↘️", "down-right"),
  # Glyph("↙️", "down-left"),
  # Glyph("↖️", "up-left"),
  # Glyph("🔄", "rotate"),
  # # --- Tier 3: Expression Nuance ---
  # Glyph("😢", "sad"),
  # Glyph("🤔", "thinking"),
  # Glyph("😱", "screaming"),
  # Glyph("😎", "sunglasses"),
  # Glyph("😴", "sleeping"),
  # Glyph("👀", "eyes"),
  # Glyph("✨", "sparkles"),
  # Glyph("💀", "skull"),
  # Glyph("💩", "poop"),
  # # --- Tier 4: Combat / Tools / Economy ---
  # Glyph("⚔️", "swords"),
  # Glyph("🛡️", "shield"),
  # Glyph("🔧", "wrench"),
  # Glyph("⚙️", "gear"),
  # Glyph("💰", "money"),
  # Glyph("🏭", "factory"),
  # Glyph("⚡", "lightning"),
  # Glyph("🔥", "fire"),
  # Glyph("💧", "water"),
  # Glyph("🌳", "tree"),
  # # --- Tier 5: Miscellaneous ---
  # Glyph("🔃", "rotate-clockwise"),
  # Glyph("🔂", "rotate-loop"),
  # Glyph("🧭", "compass"),
  # Glyph("📍", "pin"),
  # Glyph("📌", "pushpin"),
  # Glyph("💎", "diamond"),
  # Glyph("🪙", "coin"),
  # Glyph("🛢️", "oil"),
  # Glyph("⛽", "fuel"),
  # Glyph("🌾", "wheat"),
  # Glyph("🌽", "corn"),
  # Glyph("🥕", "carrot"),
  # Glyph("🪨", "rock"),
  # Glyph("⛰️", "mountain"),
  # Glyph("🪵", "wood"),
  # Glyph("🌊", "wave"),
  # Glyph("🗡️", "dagger"),
  # Glyph("🏹", "bow"),
  # Glyph("🔨", "hammer"),
  # Glyph("📎", "paperclip"),
  # Glyph("⚗️", "alembic"),
  # Glyph("🧪", "test-tube"),
  # Glyph("📦", "package"),
  # Glyph("🎒", "backpack"),
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
  # Glyph("🧐", "monocle"),
  # Glyph("😕", "confused"),
  # Glyph("😪", "sleepy"),
  # Glyph("🥱", "yawning"),
  # Glyph("🤤", "drooling"),
  # Glyph("😋", "savoring"),
  # Glyph("😏", "smirking"),
  # Glyph("🙄", "rolling-eyes"),
  # Glyph("🤡", "clown"),
  # Glyph("👻", "ghost"),
  # Glyph("🗿", "moai"),
  # Glyph("☠️", "skull-crossbones"),
  # Glyph("📈", "chart-up"),
  # Glyph("📉", "chart-down"),
  # Glyph("🚀", "rocket"),
  # Glyph("🎯", "target"),
  # Glyph("⭐", "star"),
  # Glyph("🔴", "red-circle"),
  # Glyph("🟠", "orange-circle"),
  # Glyph("🟡", "yellow-circle"),
  # Glyph("🟢", "green-circle"),
  # Glyph("🔵", "blue-circle"),
  # Glyph("🟣", "purple-circle"),
  # Glyph("🟤", "brown-circle"),
  # Glyph("⚫", "black-circle"),
  # Glyph("⚪", "white-circle"),
  # Glyph("🟧", "orange-square"),
  # Glyph("🟨", "yellow-square"),
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

find "/UI/Main/**/VibePanel":
  find "**/Button":
    onClick:
      let row = thisNode.parent.childIndex
      let column = thisNode.childIndex
      let dataId = row * 10 + column
      echo "vibeButton clicked: ", vibes[dataId].name

proc updateVibePanel*() =
  ## Updates the vibe panel to display the current vibe frequency for the agent.
  let panel = panels.vibeTemplate.copy()
  panel.position = vec2(0, 0)
  let rowTemplate = panel.find("Row")
  let buttonTemplate = rowTemplate.find("Button").copy()
  rowTemplate.removeChildren()
  panel.removeChildren()
  var row: Node
  for id, vibe in vibes:
    if id mod 10 == 0:
      row = rowTemplate.copy()
      panel.addChild(row)
    let button = buttonTemplate.copy()
    button.find("**/Icon").fills[0].imageRef = "../../vibe/" & vibe.name
    row.addChild(button)
  vibePanel.node.removeChildren()
  vibePanel.node.addChild(panel)
