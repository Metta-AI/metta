## Color management for villages and altars
## Shared color data accessed by both game.nim and map_generation.nim

import std/tables, vmath, chroma

# Global village color management
var agentVillageColors*: seq[Color] = @[]
var altarColors*: Table[IVec2, Color] = initTable[IVec2, Color]()