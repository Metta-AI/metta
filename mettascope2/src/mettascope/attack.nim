## Attack module - extends tribal.nim with weapon crafting and combat mechanics
## Implements Forge for spear crafting and combat system

import std/[strformat, random, strutils, tables, times, math], vmath, chroma
import tribal
export tribal

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
    inc env.stats[id].actionInvalid
    return
  
  # Check if agent has wood
  if agent.inventoryWood <= 0:
    inc env.stats[id].actionInvalid
    return
  
  # Check if agent already has a spear (limit 1 for balance)
  if agent.inventorySpear > 0:
    inc env.stats[id].actionInvalid
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
  
  inc env.stats[id].actionUse

proc attackWithSpearAction*(env: Environment, id: int, agent: Thing, targetDirection: int) =
  ## Attack with a spear in the specified direction
  ## targetDirection: 0=N, 1=S, 2=E, 3=W
  
  # Check if agent has a spear
  if agent.inventorySpear <= 0:
    inc env.stats[id].actionInvalid
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
    inc env.stats[id].actionInvalid
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
    
    inc env.stats[id].actionUse
  else:
    # Missed - still consume spear (risk/reward balance)
    agent.inventorySpear = 0
    env.updateObservations(AgentInventorySpearLayer, agent.pos, agent.inventorySpear)
    inc env.stats[id].actionInvalid

proc useClayOvenAction*(env: Environment, id: int, agent: Thing, ovenPos: IVec2) =
  ## Use a clay oven - placeholder for future implementation
  ## Could be used for crafting shields, pottery for water storage, etc.
  inc env.stats[id].actionInvalid  # Not implemented yet

# Extended step function that includes attack mechanics
proc stepWithAttack*(env: Environment, actions: ptr array[MapAgents, array[2, uint8]]) =
  ## Enhanced step function that includes attack actions
  ## Action 9 = Attack with spear
  
  inc env.currentStep
  for id, action in actions[]:
    let agent = env.agents[id]
    if agent.frozen > 0:
      continue

    case action[0]:
    of 0: env.noopAction(id, agent)
    of 1: env.moveAction(id, agent, action[1].int)
    of 2: env.rotateAction(id, agent, action[1].int)
    of 3: 
      # Enhanced use action - check what we're using
      if action[1] > 3:
        inc env.stats[id].actionInvalid
        continue
      
      # Calculate target position based on direction argument
      var usePos = agent.pos
      case action[1]:
      of 0: usePos.y -= 1  # North
      of 1: usePos.y += 1  # South
      of 2: usePos.x += 1  # East
      of 3: usePos.x -= 1  # West
      else:
        inc env.stats[id].actionInvalid
        continue
      
      let targetThing = env.getThing(usePos)
      if not isNil(targetThing):
        case targetThing.kind:
        of Forge:
          env.useForgeAction(id, agent, usePos)
        of ClayOven:
          env.useClayOvenAction(id, agent, usePos)
        else:
          # Fall back to original use action for other objects
          env.useAction(id, agent, action[1].int)
      else:
        env.useAction(id, agent, action[1].int)
    
    of 4: env.attackAction(id, agent, action[1].int)  # Legacy attack (unused)
    of 5: env.getAction(id, agent, action[1].int)
    of 6: env.shieldAction(id, agent, action[1].int)
    of 7: env.giftAction(id, agent)
    of 8: env.swapAction(id, agent, action[1].int)
    of 9: env.attackWithSpearAction(id, agent, action[1].int)  # New spear attack
    else: inc env.stats[id].actionInvalid
  
  # Continue with rest of step logic (cooldowns, Clippys, etc.)
  # This would include all the existing step logic from tribal.nim
  
  # Update forge cooldowns
  for thing in env.things:
    if thing.kind == Forge and thing.cooldown > 0:
      thing.cooldown -= 1
    elif thing.kind == ClayOven and thing.cooldown > 0:
      thing.cooldown -= 1

# Helper function to check if position is within spear range of any agent with spear
proc isThreatenedBySpear*(env: Environment, pos: IVec2): bool =
  ## Check if a position is within spear range of any agent with a spear
  for agent in env.agents:
    if agent.hasSpear():
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