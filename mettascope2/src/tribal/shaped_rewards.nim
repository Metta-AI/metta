## Shaped Rewards Module
## Defines reward values for different actions to encourage meaningful agent behavior
## These rewards guide agents through: exploration → resource gathering → crafting → combat → cooperation

import tribal_game
export tribal

# ============ Resource Gathering Rewards ============
# Small rewards for collecting basic resources
const
  RewardGetWater* = 0.02      # Collecting water from tiles
  RewardGetWheat* = 0.02      # Harvesting wheat 
  RewardGetWood* = 0.03       # Chopping wood (slightly higher as it's needed for spears)
  RewardMineOre* = 0.05       # Mining ore (first step in battery chain)

# ============ Crafting & Production Rewards ============
# Medium rewards for transforming resources
const
  RewardConvertOreToBattery* = 0.1   # Using converter to make batteries
  RewardCraftSpear* = 0.15           # Using forge to craft spear (defense priority)
  RewardCraftAtClayOven* = 0.1       # Future: crafting at clay oven
  RewardCraftAtWeavingLoom* = 0.1    # Future: crafting at weaving loom
  RewardCraftAtArmory* = 0.1         # Future: crafting at armory

# ============ Combat & Defense Rewards ============
# High rewards for successful combat
const
  RewardDestroyClippy* = 0.5         # Destroying a clippy with spear
  RewardDefendAltar* = 0.3           # Bonus when destroying clippy near own altar
  RewardTeamDefense* = 0.2           # Bonus for destroying clippy near ally

# ============ Cooperation & Contribution Rewards ============
# Highest rewards for team contributions
const
  RewardDepositBatteryAtAltar* = 1.0  # Contributing battery to altar (main goal)
  RewardRespawnAlly* = 0.5            # Future: helping respawn teammate
  RewardShareResource* = 0.1          # Future: gifting resources to allies

# ============ Movement & Exploration Rewards ============
# Tiny rewards to encourage exploration
const
  RewardExploreNewTile* = 0.001       # First time visiting a tile
  RewardMoveTowardObjective* = 0.002  # Moving closer to important targets

# ============ Penalty Values ============
# Negative rewards for undesirable actions
const
  PenaltyInvalidAction* = -0.01       # Trying invalid actions
  PenaltyIdleTooLong* = -0.005        # Being idle for multiple steps
  PenaltyWasteSpear* = -0.1           # Missing with a spear attack
  PenaltyAgentDeath* = -0.5           # When agent is killed by clippy
  PenaltyAltarDamaged* = -0.3         # When own altar loses a heart

# ============ Reward Shaping Functions ============

proc calculateDistanceReward*(currentDist, previousDist: int, targetReward: float32): float32 =
  ## Calculate reward for moving closer to a target
  if currentDist < previousDist:
    return targetReward * float32(previousDist - currentDist) / float32(previousDist)
  elif currentDist > previousDist:
    return -targetReward * 0.5  # Small penalty for moving away
  else:
    return 0.0

proc calculateInventoryEfficiencyReward*(inventory, maxInventory: int): float32 =
  ## Reward efficient inventory management (not hoarding)
  let usage = float32(inventory) / float32(maxInventory)
  if usage > 0.8:
    return -0.01  # Penalty for nearly full inventory (should use resources)
  elif usage < 0.3:
    return 0.01   # Small reward for keeping inventory available
  else:
    return 0.0

proc calculateTeamworkBonus*(numAlliesNearby: int, baseReward: float32): float32 =
  ## Bonus reward multiplier for actions done near allies
  case numAlliesNearby:
  of 0: return baseReward
  of 1: return baseReward * 1.2
  of 2: return baseReward * 1.5
  else: return baseReward * 2.0

proc calculateUrgencyMultiplier*(altarHearts: int): float32 =
  ## Increase reward urgency when altar is in danger
  case altarHearts:
  of 0..1: return 2.0    # Critical - double all defensive rewards
  of 2..3: return 1.5    # Urgent - 50% boost
  of 4..5: return 1.2    # Concerning - 20% boost
  else: return 1.0       # Normal

# ============ Integrated Reward Calculator ============

proc calculateActionReward*(
  env: Environment,
  agentId: int,
  action: string,
  success: bool = true,
  context: Table[string, int] = initTable[string, int]()
): float32 =
  ## Calculate the total shaped reward for an action
  ## Context can include: distance_to_target, allies_nearby, altar_hearts, etc.
  
  let agent = env.agents[agentId]
  var reward = 0.0'f32
  
  case action:
  of "get_water":
    reward = RewardGetWater
  of "get_wheat":
    reward = RewardGetWheat
  of "get_wood":
    reward = RewardGetWood
  of "mine_ore":
    reward = RewardMineOre
  of "convert_ore":
    reward = RewardConvertOreToBattery
  of "craft_spear":
    reward = RewardCraftSpear
  of "destroy_clippy":
    reward = RewardDestroyClippy
    # Add defense bonuses
    if context.hasKey("near_own_altar"):
      reward += RewardDefendAltar
    if context.hasKey("allies_nearby"):
      reward = calculateTeamworkBonus(context["allies_nearby"], reward)
  of "deposit_battery":
    reward = RewardDepositBatteryAtAltar
    # Apply urgency if altar is low on hearts
    if context.hasKey("altar_hearts"):
      reward *= calculateUrgencyMultiplier(context["altar_hearts"])
  of "invalid_action":
    reward = PenaltyInvalidAction
  of "agent_death":
    reward = PenaltyAgentDeath
  of "altar_damaged":
    reward = PenaltyAltarDamaged
  of "spear_miss":
    reward = PenaltyWasteSpear
  else:
    reward = 0.0
  
  # Apply success modifier
  if not success and reward > 0:
    reward *= 0.1  # Drastically reduce reward for failed attempts
  
  return reward

# ============ Reward Integration Helpers ============

proc applyShapedReward*(env: Environment, agentId: int, action: string, success: bool = true) =
  ## Apply a shaped reward to an agent
  let reward = calculateActionReward(env, agentId, action, success)
  if reward != 0:
    env.agents[agentId].reward += reward

proc applyContextualReward*(
  env: Environment, 
  agentId: int, 
  action: string, 
  context: Table[string, int],
  success: bool = true
) =
  ## Apply a shaped reward with context
  let reward = calculateActionReward(env, agentId, action, success, context)
  if reward != 0:
    env.agents[agentId].reward += reward

# ============ Batch Reward Updates ============

proc updateAllAgentRewards*(env: Environment) =
  ## Update all agents with time-based rewards/penalties
  ## Called at the end of each step
  
  for i, agent in env.agents:
    # Penalty for being dead (encourages respawn mechanics)
    if env.terminated[i] > 0:
      agent.reward += PenaltyAgentDeath * 0.01  # Small continuous penalty
    
    # Check for idle penalty (would need to track action history)
    # This is a placeholder - would need action tracking in Environment
    
    # Exploration rewards could be added here with visited tile tracking

# ============ Reward Scheduling ============

type
  RewardSchedule* = object
    stepCount*: int
    explorationPhase*: bool  # Early game - bonus exploration rewards
    combatPhase*: bool       # Mid game - bonus combat rewards  
    cooperationPhase*: bool  # Late game - bonus cooperation rewards

proc createRewardSchedule*(): RewardSchedule =
  ## Create a new reward schedule
  result.stepCount = 0
  result.explorationPhase = true
  result.combatPhase = false
  result.cooperationPhase = false

proc updateRewardSchedule*(schedule: var RewardSchedule, step: int) =
  ## Update reward schedule based on game progression
  schedule.stepCount = step
  
  # Phase transitions based on step count
  if step < 1000:
    schedule.explorationPhase = true
    schedule.combatPhase = false
    schedule.cooperationPhase = false
  elif step < 5000:
    schedule.explorationPhase = false
    schedule.combatPhase = true
    schedule.cooperationPhase = false
  else:
    schedule.explorationPhase = false
    schedule.combatPhase = false
    schedule.cooperationPhase = true

proc getPhaseMultiplier*(schedule: RewardSchedule, rewardType: string): float32 =
  ## Get multiplier based on current game phase
  if schedule.explorationPhase:
    case rewardType:
    of "exploration": return 2.0
    of "gathering": return 1.5
    else: return 1.0
  elif schedule.combatPhase:
    case rewardType:
    of "combat": return 2.0
    of "crafting": return 1.5
    else: return 1.0
  elif schedule.cooperationPhase:
    case rewardType:
    of "cooperation": return 2.0
    of "combat": return 1.5
    else: return 1.0
  else:
    return 1.0