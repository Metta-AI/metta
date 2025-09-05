## Core action implementations for agents
## Contains all the action logic extracted from tribal.nim

import std/[math, random]
import vmath
import environment_core, observations, statistics, rewards, terrain

# Action implementations
proc noopAction*(env: Environment, id: int, agent: Thing) =
  ## Do nothing action
  inc env.stats[id].actionNoop

proc moveAction*(env: Environment, id: int, agent: Thing, direction: int) =
  ## Move the agent in specified direction
  var newPos = agent.pos
  
  # Direction is the orientation to move in
  if direction >= 0 and direction <= 7:
    agent.orientation = Orientation(direction)
    let delta = getOrientationDelta(agent.orientation)
    newPos = agent.pos + ivec2(delta.x, delta.y)
  else:
    inc env.stats[id].actionInvalid
    return
  
  # Check if new position is empty
  if env.isEmpty(newPos):
    # Clear old position in grid
    env.grid[agent.pos.x][agent.pos.y] = nil
    
    # Update observation layers for old position
    env.updateObservations(AgentLayer, agent.pos, 0)
    env.updateObservations(AgentOrientationLayer, agent.pos, 0)
    env.updateObservations(AgentInventoryOreLayer, agent.pos, 0)
    env.updateObservations(AgentInventoryBatteryLayer, agent.pos, 0)
    env.updateObservations(AgentInventoryWaterLayer, agent.pos, 0)
    env.updateObservations(AgentInventoryWheatLayer, agent.pos, 0)
    env.updateObservations(AgentInventoryWoodLayer, agent.pos, 0)
    env.updateObservations(AgentInventorySpearLayer, agent.pos, 0)
    
    # Move agent
    agent.pos = newPos
    
    # Update grid with new position
    env.grid[agent.pos.x][agent.pos.y] = agent
    
    # Update observation layers for new position
    env.updateObservations(AgentLayer, agent.pos, 1)
    env.updateObservations(AgentOrientationLayer, agent.pos, agent.orientation.int)
    env.updateObservations(AgentInventoryOreLayer, agent.pos, agent.inventoryOre)
    env.updateObservations(AgentInventoryBatteryLayer, agent.pos, agent.inventoryBattery)
    env.updateObservations(AgentInventoryWaterLayer, agent.pos, agent.inventoryWater)
    env.updateObservations(AgentInventoryWheatLayer, agent.pos, agent.inventoryWheat)
    env.updateObservations(AgentInventoryWoodLayer, agent.pos, agent.inventoryWood)
    env.updateObservations(AgentInventorySpearLayer, agent.pos, agent.inventorySpear)
    
    # Update full observations for this agent
    env.updateObservations(id)
    
    inc env.stats[id].actionMove
  else:
    inc env.stats[id].actionInvalid

proc rotateAction*(env: Environment, id: int, agent: Thing, argument: int) =
  ## Rotate the agent to face a new direction
  if argument < 0 or argument > 7:
    inc env.stats[id].actionInvalid
    return
  agent.orientation = Orientation(argument)
  env.updateObservations(AgentOrientationLayer, agent.pos, argument)
  inc env.stats[id].actionRotate

proc getAction*(env: Environment, id: int, agent: Thing, direction: int) =
  ## Get resources from terrain in specified direction
  if direction < 0 or direction > 7:
    inc env.stats[id].actionInvalid
    return
  
  let delta = getOrientationDelta(Orientation(direction))
  let targetPos = agent.pos + ivec2(delta.x, delta.y)
  
  # Check bounds
  if targetPos.x < 0 or targetPos.x >= MapWidth or 
     targetPos.y < 0 or targetPos.y >= MapHeight:
    inc env.stats[id].actionInvalid
    return
  
  # Check terrain at target position
  case env.terrain[targetPos.x][targetPos.y]:
  of Water:
    # Get water (max 5 water inventory)
    if agent.inventoryWater < 5:
      agent.inventoryWater += 1
      env.terrain[targetPos.x][targetPos.y] = Empty  # Remove water tile
      env.updateObservations(AgentInventoryWaterLayer, agent.pos, agent.inventoryWater)
      agent.reward += RewardGetWater
      inc env.stats[id].actionGetWater
      inc env.stats[id].actionGet
    else:
      inc env.stats[id].actionInvalid
  
  of Wheat:
    # Get wheat (max 5 wheat inventory)
    if agent.inventoryWheat < 5:
      agent.inventoryWheat += 1
      env.terrain[targetPos.x][targetPos.y] = Empty  # Remove wheat tile
      env.updateObservations(AgentInventoryWheatLayer, agent.pos, agent.inventoryWheat)
      agent.reward += RewardGetWheat
      inc env.stats[id].actionGetWheat
      inc env.stats[id].actionGet
    else:
      inc env.stats[id].actionInvalid
  
  of Tree:
    # Get wood (max 5 wood inventory)
    if agent.inventoryWood < 5:
      agent.inventoryWood += 1
      env.terrain[targetPos.x][targetPos.y] = Empty  # Remove tree tile
      env.updateObservations(AgentInventoryWoodLayer, agent.pos, agent.inventoryWood)
      agent.reward += RewardGetWood
      inc env.stats[id].actionGetWood
      inc env.stats[id].actionGet
    else:
      inc env.stats[id].actionInvalid
  
  else:
    inc env.stats[id].actionInvalid

proc useAction*(env: Environment, id: int, agent: Thing, direction: int) =
  ## Use an object in the specified direction
  if direction < 0 or direction > 7:
    inc env.stats[id].actionInvalid
    return
  
  let delta = getOrientationDelta(Orientation(direction))
  let usePos = agent.pos + ivec2(delta.x, delta.y)
  var thing = env.getThing(usePos)
  
  if thing == nil:
    inc env.stats[id].actionInvalid
    return
  
  case thing.kind:
  of Wall, Agent, Temple, Clippy:
    inc env.stats[id].actionInvalid
  
  of Altar:
    if thing.cooldown == 0 and agent.inventoryBattery >= 1:
      # Agent deposits a battery as a heart into the altar
      agent.reward += RewardDepositBattery
      agent.inventoryBattery -= 1
      thing.hearts += 1  # Add one heart to altar
      env.updateObservations(AgentInventoryBatteryLayer, agent.pos, agent.inventoryBattery)
      env.updateObservations(AltarHeartsLayer, thing.pos, thing.hearts)
      thing.cooldown = MapObjectAltarCooldown
      env.updateObservations(AltarReadyLayer, thing.pos, 0)
      inc env.stats[id].actionUseAltar
      inc env.stats[id].actionUse
    else:
      inc env.stats[id].actionInvalid
  
  of Mine:
    if thing.cooldown == 0 and agent.inventoryOre < MapObjectAgentMaxInventory:
      # Mine gives 1 ore
      agent.inventoryOre += 1
      env.updateObservations(AgentInventoryOreLayer, agent.pos, agent.inventoryOre)
      thing.cooldown = MapObjectMineCooldown
      env.updateObservations(MineReadyLayer, thing.pos, thing.cooldown)
      agent.reward += RewardMineOre
      inc env.stats[id].actionUseMine
      inc env.stats[id].actionUse
  
  of Converter:
    if thing.cooldown == 0 and agent.inventoryOre > 0 and agent.inventoryBattery < MapObjectAgentMaxInventory:
      # Convert 1 ore to 1 battery
      agent.inventoryOre -= 1
      agent.inventoryBattery += 1
      env.updateObservations(AgentInventoryOreLayer, agent.pos, agent.inventoryOre)
      env.updateObservations(AgentInventoryBatteryLayer, agent.pos, agent.inventoryBattery)
      # No cooldown for instant conversion
      agent.reward += RewardConvertOreToBattery
      inc env.stats[id].actionUseConverter
      inc env.stats[id].actionUse
  
  of Forge:
    # Use forge to craft a spear from wood
    if thing.cooldown == 0 and agent.inventoryWood > 0 and agent.inventorySpear == 0:
      # Craft spear
      agent.inventoryWood -= 1
      agent.inventorySpear = 1
      thing.cooldown = 5  # Forge cooldown
      env.updateObservations(AgentInventoryWoodLayer, agent.pos, agent.inventoryWood)
      env.updateObservations(AgentInventorySpearLayer, agent.pos, agent.inventorySpear)
      agent.reward += RewardCraftSpear
      inc env.stats[id].actionUse
    else:
      inc env.stats[id].actionInvalid
  
  of Armory, ClayOven, WeavingLoom:
    # These will be handled by their specific modules
    inc env.stats[id].actionInvalid

proc attackAction*(env: Environment, id: int, agent: Thing, targetDirection: int) =
  ## Attack with a spear in the specified direction
  if targetDirection < 0 or targetDirection > 7:
    inc env.stats[id].actionInvalid
    return
  
  # Check if agent has a spear
  if agent.inventorySpear <= 0:
    inc env.stats[id].actionInvalid
    return
  
  # Calculate target position (spears have range of 2)
  let delta = getOrientationDelta(Orientation(targetDirection))
  let targetPos = agent.pos + ivec2(delta.x * 2, delta.y * 2)
  
  # Check if position is valid
  if targetPos.x < 0 or targetPos.x >= MapWidth or 
     targetPos.y < 0 or targetPos.y >= MapHeight:
    # Missed - still consume spear
    agent.inventorySpear = 0
    env.updateObservations(AgentInventorySpearLayer, agent.pos, agent.inventorySpear)
    return
  
  # Get target at position
  let target = env.getThing(targetPos)
  if target != nil and target.kind == Clippy:
    # Hit the clippy! Remove it
    env.things.del(env.things.find(target))
    env.grid[targetPos.x][targetPos.y] = nil
    
    # Consume the spear
    agent.inventorySpear = 0
    env.updateObservations(AgentInventorySpearLayer, agent.pos, agent.inventorySpear)
    
    # Apply reward
    agent.reward += RewardDestroyClippy
  else:
    # Missed - still consume spear
    agent.inventorySpear = 0
    env.updateObservations(AgentInventorySpearLayer, agent.pos, agent.inventorySpear)

proc swapAction*(env: Environment, id: int, agent: Thing, argument: int) =
  ## Swap positions with a frozen agent
  if argument > 1:
    inc env.stats[id].actionInvalid
    return
  
  let targetPos = agent.pos + orientationToVec(agent.orientation)
  let target = env.getThing(targetPos)
  
  if target == nil:
    inc env.stats[id].actionInvalid
    return
  
  if target.kind == Agent and target.frozen > 0:
    # Swap positions
    var temp = agent.pos
    agent.pos = target.pos
    target.pos = temp
    
    # Update grid
    env.grid[agent.pos.x][agent.pos.y] = agent
    env.grid[target.pos.x][target.pos.y] = target
    
    inc env.stats[id].actionSwap
    env.updateObservations(id)
    env.updateObservations(target.agentId)
  else:
    inc env.stats[id].actionInvalid

# Placeholder actions for future implementation
proc jumpAction*(env: Environment, id: int, agent: Thing) =
  ## Jump action (not implemented)
  discard

proc transferAction*(env: Environment, id: int, agent: Thing) =
  ## Transfer resources (not implemented)
  discard

proc giftAction*(env: Environment, id: int, agent: Thing) =
  ## Gift resources (not implemented)
  discard

proc shieldAction*(env: Environment, id: int, agent: Thing, argument: int) =
  ## Shield action removed - no longer used
  inc env.stats[id].actionInvalid