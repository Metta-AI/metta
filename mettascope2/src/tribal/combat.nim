## Combat system
## Handles offensive and defensive combat mechanics including spears, armor, and other weapons
## 
## TODO: Defense items (hats, armor) are defined in objects.nim but not yet implemented
##       When implementing defense mechanics, consider moving DefenseItem enum here
##       Current items: Hat (1 hit protection), Armor (3 hit protection)

import vmath
import environment
export environment

const
  ForgeWoodCost* = 1
  ForgeCooldown* = 5
  SpearRange* = 2
  WeavingLoomWheatCost* = 1
  WeavingLoomCooldown* = 15
  ArmoryOreCost* = 1
  ArmoryCooldown* = 20
  ArmorMaxUses* = 3

proc getManhattanDistance*(pos1, pos2: IVec2): int =
  return abs(pos1.x - pos2.x) + abs(pos1.y - pos2.y)

proc hasSpear*(agent: Thing): bool =
  ## Check if an agent has a spear
  return agent.kind == Agent and agent.inventorySpear > 0

proc hasDefense*(agent: Thing): bool =
  ## Check if an agent has any defense items
  return agent.kind == Agent and (agent.inventoryHat > 0 or agent.inventoryArmor > 0)


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

proc defendAgainstAttack*(agent: Thing, env: Environment): bool =
  ## Check if agent can defend against an attack and consume defense items
  ## Returns true if agent survived the attack, false if agent should die
  
  # First check armor (provides 3 hits of protection)
  if agent.inventoryArmor > 0:
    agent.inventoryArmor -= 1
    env.updateObservations(AgentInventoryArmorLayer, agent.pos, agent.inventoryArmor)
    return true
  
  # Then check hat (provides 1 hit of protection)
  if agent.inventoryHat > 0:
    agent.inventoryHat = 0
    env.updateObservations(AgentInventoryHatLayer, agent.pos, agent.inventoryHat)
    return true
  
  # No defense items - agent dies
  return false

proc isThreatenedBySpear*(env: Environment, pos: IVec2): bool =
  for agent in env.agents:
    if agent.hasSpear():
      if getManhattanDistance(agent.pos, pos) <= SpearRange:
        return true
  return false

