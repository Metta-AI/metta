import vmath
import environment, common
export environment

const
  SpearRange* = 2

proc defendAgainstAttack*(agent: Thing, env: Environment): bool =
  
  if agent.inventoryArmor > 0:
    agent.inventoryArmor -= 1
    env.updateObservations(AgentInventoryArmorLayer, agent.pos, agent.inventoryArmor)
    return true
  
  if agent.inventoryHat > 0:
    agent.inventoryHat = 0
    env.updateObservations(AgentInventoryHatLayer, agent.pos, agent.inventoryHat)
    return true
  
  return false

proc isThreatenedBySpear*(env: Environment, pos: IVec2): bool =
  for agent in env.agents:
    if agent.kind == Agent and agent.inventorySpear > 0:
      if manhattanDistance(agent.pos, pos) <= SpearRange:
        return true
  return false