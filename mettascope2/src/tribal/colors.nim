## Color management for villages and altars
## Shared color data accessed by both game logic and rendering

import std/[tables, math], vmath, chroma

# Global village color management
var agentVillageColors*: seq[Color] = @[]
var altarColors*: Table[IVec2, Color] = initTable[IVec2, Color]()

proc generateEntityColor*(entityType: string, id: int, fallbackColor: Color = color(0.5, 0.5, 0.5, 1.0)): Color =
  ## Unified color generation for all entity types
  ## Uses golden angle for optimal color distribution
  case entityType:
  of "agent":
    if id >= 0 and id < agentVillageColors.len:
      return agentVillageColors[id]
    # Fallback using mathematical constants for variety
    let f = id.float32
    return color(
      f * PI mod 1.0,
      f * E mod 1.0,
      f * sqrt(2.0) mod 1.0,
      1.0
    )
  of "village":
    # Warm colors for villages using golden angle
    let hue = (id.float32 * 137.5) mod 360.0 / 360.0
    let saturation = 0.7 + (id.float32 * 0.13) mod 0.3
    let lightness = 0.5 + (id.float32 * 0.17) mod 0.2
    return color(hue, saturation, lightness, 1.0)
  else:
    return fallbackColor

proc getAltarColor*(pos: IVec2): Color =
  ## Get altar color by position, with white fallback
  altarColors.getOrDefault(pos, color(1.0, 1.0, 1.0, 1.0))