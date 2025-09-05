## Combat system
## Handles offensive and defensive combat mechanics including spears, armor, and other weapons

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

