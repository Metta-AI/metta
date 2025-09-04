## Example Integration of Shaped Rewards into Tribal.nim
## This shows how to add the shaped reward calls to existing actions
## Copy these patterns into the appropriate places in tribal.nim

import shaped_rewards, tables

# ============ Example: Mining Action ============
# In the Mine case of useAction:
proc exampleMineWithReward(env: Environment, id: int, agent: Thing, thing: Thing) =
  if thing.cooldown == 0 and agent.inventoryOre < MapObjectAgentMaxInventory:
    # Mine gives 1 ore
    agent.inventoryOre += 1
    env.updateObservations(AgentInventoryOreLayer, agent.pos, agent.inventoryOre)
    thing.cooldown = MapObjectMineCooldown
    env.updateObservations(MineReadyLayer, thing.pos, thing.cooldown)
    
    # ADD SHAPED REWARD:
    agent.reward += RewardMineOre
    
    inc env.stats[id].actionUseMine
    inc env.stats[id].actionUse

# ============ Example: Converter Action ============
# In the Converter case of useAction:
proc exampleConverterWithReward(env: Environment, id: int, agent: Thing, thing: Thing) =
  if thing.cooldown == 0 and agent.inventoryOre > 0 and agent.inventoryBattery < MapObjectAgentMaxInventory:
    # Convert 1 ore to 1 battery
    agent.inventoryOre -= 1
    agent.inventoryBattery += 1
    env.updateObservations(AgentInventoryOreLayer, agent.pos, agent.inventoryOre)
    env.updateObservations(AgentInventoryBatteryLayer, agent.pos, agent.inventoryBattery)
    thing.cooldown = 0
    env.updateObservations(ConverterReadyLayer, thing.pos, 1)
    
    # ADD SHAPED REWARD:
    agent.reward += RewardConvertOreToBattery
    
    inc env.stats[id].actionUseConverter
    inc env.stats[id].actionUse

# ============ Example: Altar Action ============
# In the Altar case of useAction:
proc exampleAltarWithReward(env: Environment, id: int, agent: Thing, thing: Thing) =
  if thing.cooldown == 0 and agent.inventoryBattery >= 1:
    # Agent deposits a battery as a heart into the altar
    agent.inventoryBattery -= 1
    thing.hearts += 1
    env.updateObservations(AgentInventoryBatteryLayer, agent.pos, agent.inventoryBattery)
    env.updateObservations(AltarHeartsLayer, thing.pos, thing.hearts)
    thing.cooldown = MapObjectAltarCooldown
    env.updateObservations(AltarReadyLayer, thing.pos, thing.cooldown)
    
    # ADD SHAPED REWARD WITH CONTEXT:
    var context = initTable[string, int]()
    context["altar_hearts"] = thing.hearts
    
    # Check if this is agent's home altar for bonus
    if agent.homeAltar == thing.pos:
      agent.reward += RewardDepositBatteryAtAltar
    else:
      # Helping another team's altar (cooperation bonus)
      agent.reward += RewardDepositBatteryAtAltar * 0.8
    
    # Apply urgency multiplier if altar is low
    agent.reward *= calculateUrgencyMultiplier(thing.hearts)
    
    inc env.stats[id].actionUseAltar
    inc env.stats[id].actionUse

# ============ Example: Forge Action (Already in tribal.nim) ============
# Current implementation with reward:
proc exampleForgeWithReward(env: Environment, id: int, agent: Thing, thing: Thing) =
  if thing.cooldown == 0 and agent.inventoryWood > 0 and agent.inventorySpear == 0:
    # Craft spear
    agent.inventoryWood -= 1
    agent.inventorySpear = 1
    thing.cooldown = 5
    env.updateObservations(AgentInventoryWoodLayer, agent.pos, agent.inventoryWood)
    env.updateObservations(AgentInventorySpearLayer, agent.pos, agent.inventorySpear)
    
    # SHAPED REWARD (already added):
    agent.reward += RewardCraftSpear  # 0.15
    
    inc env.stats[id].actionUse

# ============ Example: Get Actions ============
# In getAction proc for different resources:
proc exampleGetActionWithReward(env: Environment, id: int, agent: Thing, targetPos: IVec2) =
  case env.terrain[targetPos.x][targetPos.y]:
  of Water:
    if agent.inventoryWater < 5:
      agent.inventoryWater += 1
      env.terrain[targetPos.x][targetPos.y] = Empty
      env.updateObservations(AgentInventoryWaterLayer, agent.pos, agent.inventoryWater)
      
      # ADD SHAPED REWARD:
      agent.reward += RewardGetWater
      
      inc env.stats[id].actionGetWater
      inc env.stats[id].actionGet
  
  of Wheat:
    if agent.inventoryWheat < 5:
      agent.inventoryWheat += 1
      env.terrain[targetPos.x][targetPos.y] = Empty
      env.updateObservations(AgentInventoryWheatLayer, agent.pos, agent.inventoryWheat)
      
      # ADD SHAPED REWARD:
      agent.reward += RewardGetWheat
      
      inc env.stats[id].actionGetWheat
      inc env.stats[id].actionGet
  
  of Tree:
    if agent.inventoryWood < 5:
      agent.inventoryWood += 1
      env.terrain[targetPos.x][targetPos.y] = Empty
      env.updateObservations(AgentInventoryWoodLayer, agent.pos, agent.inventoryWood)
      
      # ADD SHAPED REWARD:
      agent.reward += RewardGetWood
      
      inc env.stats[id].actionGetWood
      inc env.stats[id].actionGet

# ============ Example: Attack Action ============
# In attackAction proc when destroying a Clippy:
proc exampleAttackWithReward(env: Environment, id: int, agent: Thing, clippyToRemove: Thing) =
  # Remove the Clippy
  env.grid[clippyToRemove.pos.x][clippyToRemove.pos.y] = nil
  let idx = env.things.find(clippyToRemove)
  if idx >= 0:
    env.things.del(idx)
  
  # Consume the spear
  agent.inventorySpear = 0
  env.updateObservations(AgentInventorySpearLayer, agent.pos, agent.inventorySpear)
  
  # ADD CONTEXTUAL SHAPED REWARD:
  var context = initTable[string, int]()
  
  # Check if near own altar (defensive bonus)
  let distToHomeAltar = abs(agent.pos.x - agent.homeAltar.x) + abs(agent.pos.y - agent.homeAltar.y)
  if distToHomeAltar <= 5:
    context["near_own_altar"] = 1
  
  # Check for nearby allies (teamwork bonus)
  var alliesNearby = 0
  for otherAgent in env.agents:
    if otherAgent != agent:
      let dist = abs(agent.pos.x - otherAgent.pos.x) + abs(agent.pos.y - otherAgent.pos.y)
      if dist <= 3:
        alliesNearby += 1
  if alliesNearby > 0:
    context["allies_nearby"] = alliesNearby
  
  env.applyContextualReward(id, "destroy_clippy", context, true)
  
  inc env.stats[id].actionUse

# ============ Example: Invalid Action Penalty ============
# Whenever incrementing actionInvalid:
proc exampleInvalidActionPenalty(env: Environment, id: int, agent: Thing) =
  inc env.stats[id].actionInvalid
  
  # ADD SHAPED PENALTY:
  agent.reward += PenaltyInvalidAction  # -0.01

# ============ Example: Agent Death ============
# When agent dies (in Clippy combat section):
proc exampleAgentDeathPenalty(env: Environment, id: int, agent: Thing) =
  # Agent dies - mark for respawn at altar
  agent.frozen = 999999
  env.terminated[agent.agentId] = 1.0
  
  # Clear the agent from its current position
  env.grid[agent.pos.x][agent.pos.y] = nil
  
  # ADD SHAPED PENALTY:
  agent.reward += PenaltyAgentDeath  # -0.5

# ============ Example: Altar Damage ============
# When Clippy damages an altar:
proc exampleAltarDamagePenalty(env: Environment, altar: Thing) =
  if altar.hearts > 0:
    altar.hearts = max(0, altar.hearts - 1)
    env.updateObservations(AltarHeartsLayer, altar.pos, altar.hearts)
    
    # ADD SHAPED PENALTY to all agents with this as home altar:
    for agent in env.agents:
      if agent.homeAltar == altar.pos:
        agent.reward += PenaltyAltarDamaged  # -0.3

# ============ Integration into Step Function ============
# At the end of step function:
proc exampleStepEndRewards(env: Environment) =
  # Update all time-based rewards
  updateAllAgentRewards(env)
  
  # Optional: Update reward schedule for phased learning
  # if env has rewardSchedule field:
  # env.rewardSchedule.updateRewardSchedule(env.currentStep)