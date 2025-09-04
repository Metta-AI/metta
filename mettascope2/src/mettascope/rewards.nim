## Reward system for reinforcement learning
## Defines reward values and reward application logic

import vmath
import environment_core

# Shaped reward constants
const
  # Resource gathering (very small)
  RewardGetWater* = 0.001
  RewardGetWheat* = 0.001
  RewardGetWood* = 0.002  # Slightly higher as it leads to spears
  RewardMineOre* = 0.003  # Slightly higher as it leads to batteries
  
  # Crafting (small)
  RewardConvertOreToBattery* = 0.01  # Converting ore to battery
  RewardCraftSpear* = 0.01  # Crafting spear at forge
  
  # Combat (moderate - helps protect altar)
  RewardDestroyClippy* = 0.1  # Destroying a clippy
  
  # Primary objective (large)
  RewardDepositBattery* = 1.0  # Depositing battery at altar (already in code)

proc applyTeamAltarReward*(env: Environment) =
  ## Apply team rewards based on altar heart counts
  ## This creates cooperative incentives where all agents from a village
  ## benefit when their altar gains hearts
  
  # First, calculate the total hearts for each altar
  var altarHearts = initTable[IVec2, int]()
  var altarAgents = initTable[IVec2, seq[int]]()
  
  # Map agents to their home altars
  for i, agent in env.agents:
    if agent.homeAltar.x >= 0 and agent.homeAltar.y >= 0:
      if agent.homeAltar notin altarAgents:
        altarAgents[agent.homeAltar] = @[]
      altarAgents[agent.homeAltar].add(i)
  
  # Get heart counts for each altar
  for thing in env.things:
    if thing.kind == Altar:
      altarHearts[thing.pos] = thing.hearts
  
  # Apply team rewards
  for altarPos, agentIds in altarAgents:
    if altarPos in altarHearts:
      let hearts = altarHearts[altarPos]
      # Small team reward for each heart the altar has
      let teamReward = hearts.float32 * 0.01
      for agentId in agentIds:
        env.agents[agentId].reward += teamReward

proc applyResourceGatheringReward*(agent: Thing, resourceType: string): float32 =
  ## Apply shaped reward for gathering resources
  case resourceType:
  of "water": return RewardGetWater
  of "wheat": return RewardGetWheat
  of "wood": return RewardGetWood
  of "ore": return RewardMineOre
  else: return 0.0

proc applyCraftingReward*(agent: Thing, craftType: string): float32 =
  ## Apply shaped reward for crafting items
  case craftType:
  of "battery": return RewardConvertOreToBattery
  of "spear": return RewardCraftSpear
  else: return 0.0

proc applyCombatReward*(agent: Thing, targetType: ThingKind): float32 =
  ## Apply shaped reward for combat actions
  case targetType:
  of Clippy: return RewardDestroyClippy
  else: return 0.0