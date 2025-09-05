import std/[strformat, random, strutils, tables, times, math], vmath, chroma
import environment
export environment

const
  ForgeWoodCost* = 1
  ForgeCooldown* = 5
  SpearRange* = 2

proc getManhattanDistance*(pos1, pos2: IVec2): int =
  return abs(pos1.x - pos2.x) + abs(pos1.y - pos2.y)

proc hasSpear*(agent: Thing): bool =
  ## Check if an agent has a spear
  return agent.kind == Agent and agent.inventorySpear > 0

proc useForgeAction*(env: Environment, id: int, agent: Thing, forge: Thing) =
  if forge.cooldown > 0:
    return
  
  if agent.inventoryWood <= 0:
    return
  
  if agent.inventorySpear > 0:
    return
  
  agent.inventoryWood -= ForgeWoodCost
  agent.inventorySpear = 1
  forge.cooldown = ForgeCooldown
  
  env.updateObservations(AgentInventoryWoodLayer, agent.pos, agent.inventoryWood)
  env.updateObservations(AgentInventorySpearLayer, agent.pos, agent.inventorySpear)
  agent.reward += 0.5

proc attackWithSpearAction*(env: Environment, id: int, agent: Thing, targetDirection: int) =
  if agent.inventorySpear <= 0:
    return
  
  var attackPositions: seq[IVec2] = @[]
  case targetDirection:
  of 0:  # North
    attackPositions.add(agent.pos + ivec2(0, -1))
    attackPositions.add(agent.pos + ivec2(0, -2))
  of 1:  # South
    attackPositions.add(agent.pos + ivec2(0, 1))
    attackPositions.add(agent.pos + ivec2(0, 2))
  of 2:  # East
    attackPositions.add(agent.pos + ivec2(1, 0))
    attackPositions.add(agent.pos + ivec2(2, 0))
  of 3:  # West
    attackPositions.add(agent.pos + ivec2(-1, 0))
    attackPositions.add(agent.pos + ivec2(-2, 0))
  else:
    return
  
  var hitClippy = false
  var clippyToRemove: Thing = nil
  
  for attackPos in attackPositions:
    if attackPos.x < 0 or attackPos.x >= MapWidth or 
       attackPos.y < 0 or attackPos.y >= MapHeight:
      continue
    
    let target = env.getThing(attackPos)
    if not isNil(target) and target.kind == Clippy:
      clippyToRemove = target
      hitClippy = true
      break
  
  if hitClippy and not isNil(clippyToRemove):
    env.grid[clippyToRemove.pos.x][clippyToRemove.pos.y] = nil
    let idx = env.things.find(clippyToRemove)
    if idx >= 0:
      env.things.del(idx)
    
    agent.inventorySpear = 0
    
    env.updateObservations(AgentInventorySpearLayer, agent.pos, agent.inventorySpear)
    agent.reward += 2.0
  else:
    agent.inventorySpear = 0
    env.updateObservations(AgentInventorySpearLayer, agent.pos, agent.inventorySpear)

proc useClayOvenAction*(env: Environment, id: int, agent: Thing, ovenPos: IVec2) =
  discard

proc isThreatenedBySpear*(env: Environment, pos: IVec2): bool =
  for agent in env.agents:
    if agent.hasSpear():
      if getManhattanDistance(agent.pos, pos) <= SpearRange:
        return true
  return false

proc renderWithWeapons*(env: Environment): string =
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
      # Then override with objects if present
      for thing in env.things:
        if thing.pos.x == x and thing.pos.y == y:
          case thing.kind
          of Agent:
            if thing.hasSpear():
              cell = "Λ"  # Agent with spear
            else:
              cell = "A"  # Regular agent
          of Wall:
            cell = "#"
          of Mine:
            cell = "m"
          of Converter:
            cell = "g"
          of Altar:
            cell = "a"
          of Spawner:
            cell = "s"
          of Clippy:
            cell = "C"
          of Armory:
            cell = "Ω"  # Greek omega for armory
          of Forge:
            cell = "F"
          of ClayOven:
            cell = "O"
          of WeavingLoom:
            cell = "W"
          break
      result.add(cell)
    result.add("\n")