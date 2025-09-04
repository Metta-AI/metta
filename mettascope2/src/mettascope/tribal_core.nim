## Core tribal environment orchestration
## This is the main module that ties together all the refactored components

import std/[strformat, random, math, strutils], vmath, jsony
import environment_core, observations, statistics, action_handlers, map_generation, 
       rewards, rendering, terrain, clippy
export environment_core, terrain, observations, statistics, rendering

# Re-export important things for backward compatibility
export agentVillageColors, altarColors

proc loadMap*(env: Environment, map: string) =
  ## Load a map from a string representation
  env.currentStep = 0
  env.terminated.clear()
  env.truncated.clear()
  env.things.setLen(0)
  env.agents.setLen(0)
  env.stats.setLen(0)
  env.grid.clear()
  env.observations.clear()
  
  for line in map.split("\n"):
    let parts = line.split(" ")
    if parts.len < 2:
      continue
    let kind = parseEnum[ThingKind](parts[0])
    let id = parts[1].parseInt
    
    case kind:
    of Agent:
      env.add(Thing(
        kind: kind,
        id: id,
        agentId: parts[2].parseInt,
        pos: ivec2(parts[3].parseInt, parts[4].parseInt),
        orientation: N,
        inventoryOre: 0,
        inventoryBattery: 0,
        inventoryWater: 0,
        inventoryWheat: 0,
        inventoryWood: 0,
        inventorySpear: 0
      ))
    of Wall:
      env.add(Thing(
        kind: kind,
        id: id,
        pos: ivec2(parts[2].parseInt, parts[3].parseInt)
      ))
    of Mine:
      env.add(Thing(
        kind: kind,
        id: id,
        pos: ivec2(parts[2].parseInt, parts[3].parseInt),
        resources: 30
      ))
    of Converter:
      env.add(Thing(
        kind: kind,
        id: id,
        pos: ivec2(parts[2].parseInt, parts[3].parseInt)
      ))
    of Altar:
      env.add(Thing(
        kind: kind,
        id: id,
        pos: ivec2(parts[2].parseInt, parts[3].parseInt),
        hearts: MapObjectAltarInitialHearts
      ))
    else:
      discard
  
  # Update all agent observations
  for agentId in 0 ..< MapAgents:
    env.updateObservations(agentId)

proc newEnvironment*(): Environment =
  ## Create a new environment and initialize it
  result = Environment()
  result.stats = @[]
  result.initMapGeneration()

proc step*(env: Environment, actions: ptr array[MapAgents, array[2, uint8]]) =
  ## Step the environment forward by processing all actions
  inc env.currentStep
  
  # Process agent actions
  for id, action in actions[]:
    let agent = env.agents[id]
    if agent.frozen > 0:
      continue
    
    case action[0]:
    of 0: env.noopAction(id, agent)
    of 1: env.moveAction(id, agent, action[1].int)
    of 2: env.rotateAction(id, agent, action[1].int)
    of 3: env.useAction(id, agent, action[1].int)
    of 4: env.attackAction(id, agent, action[1].int)
    of 5: env.getAction(id, agent, action[1].int)
    of 6: env.shieldAction(id, agent, action[1].int)
    of 7: env.giftAction(id, agent)
    of 8: env.swapAction(id, agent, action[1].int)
    else: inc env.stats[id].actionInvalid
  
  # Update objects (mines, converters, altars, temples)
  for thing in env.things:
    case thing.kind:
    of Altar:
      if thing.cooldown > 0:
        thing.cooldown -= 1
        env.updateObservations(AltarReadyLayer, thing.pos, thing.cooldown)
    
    of Converter:
      if thing.cooldown > 0:
        thing.cooldown -= 1
        env.updateObservations(ConverterReadyLayer, thing.pos, thing.cooldown)
    
    of Mine:
      if thing.cooldown > 0:
        thing.cooldown -= 1
        env.updateObservations(MineReadyLayer, thing.pos, thing.cooldown)
    
    of Agent:
      if thing.frozen > 0:
        thing.frozen -= 1
    
    of Temple:
      # Handle clippy spawning
      if thing.cooldown > 0:
        thing.cooldown -= 1
      else:
        # Count nearby clippys
        var nearbyClippyCount = 0
        for other in env.things:
          if other.kind == Clippy:
            let dist = abs(other.pos.x - thing.pos.x) + abs(other.pos.y - thing.pos.y)
            if dist <= 10:
              nearbyClippyCount += 1
        
        # Spawn new clippy if conditions met
        if shouldSpawnClippy(thing.cooldown, nearbyClippyCount):
          let templeCenter = thing.pos
          # Find spawn position near temple
          for dx in -1..1:
            for dy in -1..1:
              if dx == 0 and dy == 0:
                continue
              let spawnPos = templeCenter + ivec2(dx, dy)
              if env.isEmpty(spawnPos):
                env.add(Thing(
                  kind: Clippy,
                  pos: spawnPos,
                  homeTemple: templeCenter,
                  wanderRadius: 5,
                  wanderAngle: 0.0,
                  targetPos: ivec2(-1, -1),
                  wanderStepsRemaining: 0
                ))
                thing.cooldown = TempleCooldown
                break
    
    of Forge, Armory, ClayOven, WeavingLoom:
      # Update cooldowns for crafting buildings
      if thing.cooldown > 0:
        thing.cooldown -= 1
    
    else:
      discard
  
  # Update clippy AI behavior
  var clippysToRemove: seq[Thing] = @[]
  for thing in env.things:
    if thing.kind == Clippy:
      # Get clippy movement direction using the plague-wave expansion
      let moveDir = getOutwardExpansionDirection(
        thing.addr,
        cast[seq[pointer]](env.things.addr),
        var Rand()
      )
      
      if moveDir.x != 0 or moveDir.y != 0:
        let newPos = thing.pos + moveDir
        if env.isEmpty(newPos):
          env.grid[thing.pos.x][thing.pos.y] = nil
          thing.pos = newPos
          env.grid[newPos.x][newPos.y] = thing
  
  # Handle clippy-agent collisions
  for thing in env.things:
    if thing.kind == Clippy:
      # Check adjacent cells for agents
      for dx in -1..1:
        for dy in -1..1:
          if abs(dx) + abs(dy) == 1:  # Only orthogonal adjacency
            let checkPos = thing.pos + ivec2(dx, dy)
            let adjacentThing = env.getThing(checkPos)
            if adjacentThing != nil and adjacentThing.kind == Agent:
              # Combat occurs! 50% chance agent survives
              var r = initRand()
              
              # Clippy always dies in combat
              clippysToRemove.add(thing)
              env.grid[thing.pos.x][thing.pos.y] = nil
              
              # 50% chance agent dies and needs respawning
              if r.rand(1.0) < 0.5:
                # Agent dies - mark for respawn at altar
                adjacentThing.frozen = 999999  # Mark as dead
                adjacentThing.pos = adjacentThing.homeAltar
                adjacentThing.inventoryOre = 0
                adjacentThing.inventoryBattery = 0
                adjacentThing.inventoryWater = 0
                adjacentThing.inventoryWheat = 0
                adjacentThing.inventoryWood = 0
                adjacentThing.inventorySpear = 0
              break
      
      # Check if clippy touches an altar (instant destruction)
      let atPos = env.getThing(thing.pos)
      if atPos != nil and atPos.kind == Altar:
        clippysToRemove.add(thing)
  
  # Remove dead clippys
  for clippy in clippysToRemove:
    env.things.del(env.things.find(clippy))
  
  # Respawn dead agents at their altars
  for agent in env.agents:
    if agent.frozen >= 999999:  # Agent is dead
      if agent.homeAltar.x >= 0 and agent.homeAltar.y >= 0:
        # Find the altar
        var altar: Thing = nil
        for thing in env.things:
          if thing.kind == Altar and thing.pos == agent.homeAltar:
            altar = thing
            break
        
        # Respawn if altar exists and has hearts
        if altar != nil and altar.hearts >= MapObjectAltarRespawnCost:
          # Find empty position near altar
          altar.hearts -= MapObjectAltarRespawnCost
          env.updateObservations(AltarHeartsLayer, altar.pos, altar.hearts)
          
          for dx in -2..2:
            for dy in -2..2:
              if abs(dx) + abs(dy) <= 2 and (dx != 0 or dy != 0):
                let respawnPos = altar.pos + ivec2(dx, dy)
                if env.isEmpty(respawnPos):
                  # Respawn the agent
                  env.grid[agent.pos.x][agent.pos.y] = nil
                  agent.pos = respawnPos
                  agent.frozen = MapObjectAgentFreezeDuration
                  env.grid[respawnPos.x][respawnPos.y] = agent
                  env.updateObservations(agent.agentId)
                  break
  
  # Apply team altar rewards
  env.applyTeamAltarReward()

proc reset*(env: Environment) =
  ## Reset the environment to initial state
  env.currentStep = 0
  env.terminated.clear()
  env.truncated.clear()
  env.things.setLen(0)
  env.agents.setLen(0)
  env.stats.setLen(0)
  env.grid.clear()
  env.observations.clear()
  env.initMapGeneration()