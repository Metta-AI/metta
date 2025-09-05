import tribal_game
## Rendering module for environment visualization
## Handles ASCII rendering of the game state

import std/strformat
import environment_core, 

proc render*(env: Environment): string =
  ## Render the environment as an ASCII string
  for y in 0 ..< MapHeight:
    for x in 0 ..< MapWidth:
      var cell = " "
      # First check terrain
      case env.terrain[x][y]
      of Water:
        cell = "~"
      of Wheat:
        cell = "."
      of Tree:
        cell = "T"
      of Empty:
        cell = " "
      
      # Then check for entities (they override terrain)
      for thing in env.things:
        if thing.pos.x == x and thing.pos.y == y:
          case thing.kind
          of Agent:
            cell = $thing.agentId
            if thing.agentId >= 10:
              cell = "A"
          of Wall:
            cell = "#"
          of Mine:
            cell = "M"
          of Converter:
            cell = "C"
          of Altar:
            cell = "@"
          of Temple:
            cell = "!"
          of Clippy:
            cell = "c"
          of Armory:
            cell = "R"
          of Forge:
            cell = "F"
          of ClayOven:
            cell = "O"
          of WeavingLoom:
            cell = "L"
          break
      result.add(cell)
    result.add("\n")

proc dumpMap*(env: Environment): string =
  ## Dump the map in a format that can be reloaded
  for thing in env.things:
    if thing.kind == Agent:
      result.add fmt"{thing.kind} {thing.id} {thing.agentId} {thing.pos.x} {thing.pos.y}" & "\n"
    else:
      result.add fmt"{thing.kind} {thing.id} {thing.pos.x} {thing.pos.y}" & "\n"