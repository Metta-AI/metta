## Attack module - extends tribal.nim with weapon crafting and combat mechanics
## Implements Forge for spear crafting and combat system

import std/[strformat, random, strutils, tables, times, math], vmath, chroma
import game
export game

# New constants for attack system
const
  ForgeWoodCost* = 1  # Wood needed to craft a spear
  ForgeCooldown* = 5  # Cooldown after crafting
  SpearRange* = 2     # Attack range with spear (Manhattan distance)

proc getManhattanDistance*(pos1, pos2: IVec2): int =
  ## Calculate Manhattan distance between two positions
  return abs(pos1.x - pos2.x) + abs(pos1.y - pos2.y)

proc useForgeAction*(env: Environment, id: int, agent: Thing, forge: Thing) =
  ## Use a forge to craft a spear from wood
  # Check if forge is ready (not on cooldown)
  if forge.cooldown > 0:
    return
  
  # Check if agent has wood
  if agent.inventoryWood <= 0:
    return
  
  # Check if agent already has a spear (limit 1 for balance)
  if agent.inventorySpear > 0:
    return
  
  # Craft the spear
  agent.inventoryWood -= ForgeWoodCost
  agent.inventorySpear = 1
  forge.cooldown = ForgeCooldown
  
  # Update observations
  env.updateObservations(AgentInventoryWoodLayer, agent.pos, agent.inventoryWood)
  env.updateObservations(AgentInventorySpearLayer, agent.pos, agent.inventorySpear)
  
  # Give small reward for crafting
  agent.reward += 0.5

proc attackWithSpearAction*(env: Environment, id: int, agent: Thing, targetDirection: int) =
  ## Attack with a spear in the specified direction
  ## targetDirection: 0=N, 1=S, 2=E, 3=W
  
  # Check if agent has a spear
  if agent.inventorySpear <= 0:
    return
  
  # Calculate attack vector based on direction (range of 2)
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
  
  # Check for Clippys at attack positions
  var hitClippy = false
  var clippyToRemove: Thing = nil
  
  for attackPos in attackPositions:
    # Check bounds
    if attackPos.x < 0 or attackPos.x >= MapWidth or 
       attackPos.y < 0 or attackPos.y >= MapHeight:
      continue
    
    # Check for Clippy at this position
    let target = env.getThing(attackPos)
    if not isNil(target) and target.kind == Clippy:
      clippyToRemove = target
      hitClippy = true
      break
  
  if hitClippy and not isNil(clippyToRemove):
    # Remove the Clippy
    env.grid[clippyToRemove.pos.x][clippyToRemove.pos.y] = nil
    let idx = env.things.find(clippyToRemove)
    if idx >= 0:
      env.things.del(idx)
    
    # Consume the spear
    agent.inventorySpear = 0
    
    # Update observations
    env.updateObservations(AgentInventorySpearLayer, agent.pos, agent.inventorySpear)
    
    # Give reward for destroying Clippy
    agent.reward += 2.0  # Reward for successful combat
  else:
    # Missed - still consume spear (risk/reward balance)
    agent.inventorySpear = 0
    env.updateObservations(AgentInventorySpearLayer, agent.pos, agent.inventorySpear)

proc useClayOvenAction*(env: Environment, id: int, agent: Thing, ovenPos: IVec2) =
  ## Use a clay oven - placeholder for future implementation
  ## Could be used for crafting shields, pottery for water storage, etc.
  discard  # Not implemented yet

# Note: The step function is now handled in tribal.nim with the attack action integrated

# Helper function to check if position is within spear range of any agent with spear
proc isThreatenedBySpear*(env: Environment, pos: IVec2): bool =
  ## Check if a position is within spear range of any agent with a spear
  for agent in env.agents:
    if agent.inventorySpear > 0:
      if getManhattanDistance(agent.pos, pos) <= SpearRange:
        return true
  return false

# Enhanced render function to show agents with spears
proc renderWithWeapons*(env: Environment): string =
  ## Render environment showing agents with spears differently
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
            if thing.inventorySpear > 0:
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
          of Temple:
            cell = "t"
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