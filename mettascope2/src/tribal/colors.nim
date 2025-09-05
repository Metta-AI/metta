## Color management for villages and altars
## Shared color data accessed by both game logic and rendering

import std/tables, vmath, chroma

# Global village color management
var agentVillageColors*: seq[Color] = @[]
var altarColors*: Table[IVec2, Color] = initTable[IVec2, Color]()