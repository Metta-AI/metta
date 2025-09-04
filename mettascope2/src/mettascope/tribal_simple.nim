## Simplified tribal.nim without HP and Energy systems
## This version focuses on pure tactical movement and positioning

import std/[tables, random, strformat, strutils, sets, jsony, terminal]
import vmath, terrain, clippy, village, placement

# Temple and Clippy constants
const
  TempleInitialClippys* = 1  # Each temple starts with 1 Clippy
  TempleCooldown* = 20  # Spawn a new Clippy every 20 steps
  ClippyDamage* = 1  # Damage dealt to altars
  ClippyVisionRange* = 5

# Map constants
const
  MapLayoutRoomsX* = 1
  MapLayoutRoomsY* = 1
  MapLayoutWidth* = 100  
  MapLayoutHeight* = 50  
  MapBorder* = 5
  MapRoomObjectsAgents* = 15  # Total agents to spawn (will be distributed across villages)
  MapRoomObjectsHouses* = 3  # Number of houses/villages
  MapAgentsPerHouse* = 5  # Agents to spawn per house/village
  MapRoomObjectsMines* = 5
  MapRoomObjectsGenerators* = 3
  MapRoomObjectsWalls* = 30

  MapObjectAgentMaxInventory* = 5
  MapObjectAgentFreezeDuration* = 10
  MapObjectAgentMortal* = false
  MapObjectAgentUpkeepTime* = 0
  MapObjectAgentUseCost* = 0

  MapObjectAltarInitialHearts* = 5  # Altars start with 5 hearts
  MapObjectAltarCooldown* = 10
  MapObjectAltarRespawnCost* = 1  # Cost 1 heart to respawn an agent

  MapObjectGeneratorCooldown* = 0  # No cooldown for instant conversion

  MapObjectMineCooldown* = 5
  MapObjectMineInitialResources* = 30
  MapObjectMineUseCost* = 0

  ObservationLayers* = 16  # Reduced from 24
  ObservationWidth* = 11
  ObservationHeight* = 11

  # Computed
  MapAgents* = MapRoomObjectsAgents * MapLayoutRoomsX * MapLayoutRoomsY

proc toIVec2*(x, y: int): IVec2 =
  result.x = x.int32
  result.y = y.int32

type
  ObservationName* = enum
    AgentLayer = 0
    AgentFrozenLayer = 1
    AgentOrientationLayer = 2
    AgentShieldLayer = 3
    AgentInventory1Layer = 4
    AgentInventory2Layer = 5
    AgentInventory3Layer = 6
    WallLayer = 7
    MineLayer = 8
    MineResourcesLayer = 9
    MineReadyLayer = 10
    GeneratorLayer = 11
    GeneratorReadyLayer = 12
    AltarLayer = 13
    AltarHeartsLayer = 14
    AltarReadyLayer = 15

  Orientation* = enum
    N # Up
    S # Down
    W # Left
    E # Right

  ThingKind* = enum
    Agent
    Wall
    Mine
    Generator
    Altar
    Temple
    Clippy

  Thing* = ref object
    id*: int
    case kind*: ThingKind
    of Agent:
      agentId*: int
      orientation*: Orientation
      frozen*: int
      shield*: bool
      inventory*: int        # Slot 1: Ore (from mines)
      inventoryWater*: int   # Slot 2: Water
      inventoryWheat*: int   # Slot 3: Wheat
      inventoryWood*: int    # Slot 4: Wood
      reward*: float32
      homeAltar*: IVec2      # Position of agent's home altar for respawning
    of Wall:
      discard
    of Mine:
      resources*: int
    of Generator:
      outputResource*: int
    of Altar, Temple, Clippy:
      hearts*: int  # For altars (respawn resources) and temples
    pos*: IVec2
    cooldown*: int

  Stats* = ref object
    # Agent Stats:
    actionInvalid*: int
    actionAttack*: int
    actionAttackAgent*: int
    actionAttackAltar*: int
    actionAttackGenerator*: int
    actionAttackWall*: int
    actionAttackMine*: int
    actionMine*: int
    actionShield*: int
    actionUse*: int
    actionUseAltar*: int
    actionUseGenerator*: int
    actionUseMine*: int
    agentFrozenTime*: int
    collectedOre*: int
    depositedHearts*: int

  Environment* = ref object
    mapWidth*, mapHeight*: int
    currentStep*: int
    agents*: seq[Thing]
    things*: seq[Thing]
    clippys*: seq[Thing]
    stats*: seq[Stats]
    terminated*: seq[float32]
    truncated*: seq[float32]
    observations*: seq[seq[seq[uint8]]]
    grid*: PlacementGrid
    terrain*: TerrainGrid

# Global table for altar colors
var altarColors*: Table[IVec2, Color] = initTable[IVec2, Color]()
var agentVillageColors*: seq[Color] = @[]

const ObservationNames: seq[string] = @[
  "agent",
  "agent:frozen",
  "agent:orientation",
  "agent:shield",
  "agent:inventory:ore",
  "agent:inventory:water",
  "agent:inventory:wheat",
  "wall",
  "mine",
  "mine:resources",
  "mine:ready",
  "generator",
  "generator:ready",
  "altar",
  "altar:hearts",
  "altar:ready",
]

proc updateObservations*(env: Environment, layer: ObservationName, pos: IVec2, value: int) =
  ## Update a specific observation layer at a position
  let layerIdx = layer.int
  if layerIdx < env.observations.len and 
     pos.x >= 0 and pos.x < env.mapWidth and 
     pos.y >= 0 and pos.y < env.mapHeight:
    env.observations[layerIdx][pos.x][pos.y] = value.uint8

proc updateObservations*(env: Environment, agentId: int) =
  ## Update observations for a specific agent
  let agent = env.agents[agentId]
  let x = agent.pos.x
  let y = agent.pos.y
  
  # Clear old position observations
  for layer in 0 ..< ObservationLayers:
    env.observations[layer][x][y] = 0
  
  # Update agent observations
  env.observations[AgentLayer.int][x][y] = 1
  env.observations[AgentFrozenLayer.int][x][y] = agent.frozen.uint8
  env.observations[AgentOrientationLayer.int][x][y] = agent.orientation.int.uint8
  env.observations[AgentShieldLayer.int][x][y] = agent.shield.int.uint8
  env.observations[AgentInventory1Layer.int][x][y] = agent.inventory.uint8
  env.observations[AgentInventory2Layer.int][x][y] = agent.inventoryWater.uint8
  env.observations[AgentInventory3Layer.int][x][y] = agent.inventoryWheat.uint8

proc moveAction*(env: Environment, id: int, agent: Thing, argument: int) =
  ## Move action without energy costs
  if agent.frozen > 0:
    inc env.stats[id].actionInvalid
    return
  
  let oldPos = agent.pos
  var newPos = oldPos
  
  case argument:
  of 0: # North
    newPos.y -= 1
  of 1: # South  
    newPos.y += 1
  of 2: # West
    newPos.x -= 1
  of 3: # East
    newPos.x += 1
  else:
    inc env.stats[id].actionInvalid
    return
  
  # Check bounds
  if newPos.x < 0 or newPos.x >= env.mapWidth or newPos.y < 0 or newPos.y >= env.mapHeight:
    inc env.stats[id].actionInvalid
    return
  
  # Check collision
  if not isNil(env.grid[newPos.x][newPos.y]):
    inc env.stats[id].actionInvalid
    return
  
  # Move agent
  agent.pos = newPos
  env.grid[oldPos.x][oldPos.y] = nil
  env.grid[newPos.x][newPos.y] = agent
  
  # Update observations
  env.updateObservations(AgentLayer, oldPos, 0)
  env.updateObservations(AgentFrozenLayer, oldPos, 0)
  env.updateObservations(AgentOrientationLayer, oldPos, 0)
  env.updateObservations(AgentShieldLayer, oldPos, 0)
  
  env.updateObservations(AgentLayer, newPos, 1)
  env.updateObservations(AgentFrozenLayer, newPos, agent.frozen)
  env.updateObservations(AgentOrientationLayer, newPos, agent.orientation.int)
  env.updateObservations(AgentShieldLayer, newPos, agent.shield.int)

proc shieldAction*(env: Environment, id: int, agent: Thing) =
  ## Toggle shield (no energy cost in simplified version)
  if agent.frozen > 0:
    inc env.stats[id].actionInvalid
    return
  agent.shield = not agent.shield
  inc env.stats[id].actionShield
  env.updateObservations(AgentShieldLayer, agent.pos, agent.shield.int)

proc useAction*(env: Environment, id: int, agent: Thing, argument: int) =
  ## Simplified use action - only works with mines and altars
  if argument > 1:
    inc env.stats[id].actionInvalid
    return
    
  let targetPos = case argument:
    of 0: agent.pos + toIVec2(0, -1)  # North
    of 1: agent.pos + toIVec2(0, 1)   # South
    else: agent.pos
  
  if targetPos.x < 0 or targetPos.x >= env.mapWidth or 
     targetPos.y < 0 or targetPos.y >= env.mapHeight:
    inc env.stats[id].actionInvalid
    return
  
  let thing = env.grid[targetPos.x][targetPos.y]
  if isNil(thing):
    inc env.stats[id].actionInvalid
    return
    
  case thing.kind:
  of Mine:
    if thing.cooldown == 0 and agent.inventory < MapObjectAgentMaxInventory and thing.resources > 0:
      inc agent.inventory
      dec thing.resources
      thing.cooldown = MapObjectMineCooldown
      env.updateObservations(MineResourcesLayer, thing.pos, thing.resources)
      env.updateObservations(MineReadyLayer, thing.pos, thing.cooldown)
      env.updateObservations(AgentInventory1Layer, agent.pos, agent.inventory)
      inc env.stats[id].actionUseMine
      inc env.stats[id].collectedOre
  of Altar:
    if thing.cooldown == 0 and agent.inventory > 0 and thing.hearts < MapObjectAltarInitialHearts * 2:
      # Convert ore to hearts for respawning
      dec agent.inventory
      inc thing.hearts
      thing.cooldown = MapObjectAltarCooldown
      env.updateObservations(AgentInventory1Layer, agent.pos, agent.inventory)
      env.updateObservations(AltarHeartsLayer, thing.pos, thing.hearts)
      env.updateObservations(AltarReadyLayer, thing.pos, thing.cooldown)
      inc env.stats[id].actionUseAltar
      inc env.stats[id].depositedHearts
      agent.reward += 1
  else:
    inc env.stats[id].actionInvalid

proc attackAction*(env: Environment, id: int, agent: Thing, argument: int) =
  ## Simplified attack - freezes target, no energy cost
  if agent.frozen > 0:
    inc env.stats[id].actionInvalid
    return
  
  let targetOffset = case argument:
    of 0: toIVec2(0, -1)  # North
    of 1: toIVec2(0, 1)   # South
    of 2: toIVec2(-1, 0)  # West
    of 3: toIVec2(1, 0)   # East
    else: toIVec2(0, 0)
  
  let targetPos = agent.pos + targetOffset
  if targetPos.x < 0 or targetPos.x >= env.mapWidth or 
     targetPos.y < 0 or targetPos.y >= env.mapHeight:
    inc env.stats[id].actionInvalid
    return
  
  let target = env.grid[targetPos.x][targetPos.y]
  if isNil(target):
    inc env.stats[id].actionInvalid
    return
  
  case target.kind:
  of Agent:
    inc env.stats[id].actionAttackAgent
    inc env.stats[id].actionAttack
    if not target.shield:
      # Freeze the target
      target.frozen = MapObjectAgentFreezeDuration
      env.updateObservations(AgentFrozenLayer, target.pos, target.frozen)
      # Steal inventory
      if target.inventory > 0 and agent.inventory < MapObjectAgentMaxInventory:
        inc agent.inventory
        dec target.inventory
        env.updateObservations(AgentInventory1Layer, agent.pos, agent.inventory)
        env.updateObservations(AgentInventory1Layer, target.pos, target.inventory)
  of Altar:
    if target.hearts > 0:
      dec target.hearts
      env.updateObservations(AltarHeartsLayer, target.pos, target.hearts)
      inc env.stats[id].actionAttackAltar
  else:
    inc env.stats[id].actionInvalid

proc step*(env: Environment, actions: seq[tuple[action: int, argument: int]]) =
  ## Execute one step of the environment
  inc env.currentStep
  
  # Process agent actions
  for id in 0 ..< MapAgents:
    let agent = env.agents[id]
    if env.terminated[id] > 0:
      continue
    
    let (action, argument) = actions[id]
    case action:
    of 0: env.moveAction(id, agent, argument)
    of 1: env.attackAction(id, agent, argument)
    of 2: env.shieldAction(id, agent)
    of 3: env.useAction(id, agent, argument)
    else:
      inc env.stats[id].actionInvalid
  
  # Update frozen timers and shields
  for agent in env.agents:
    if agent.frozen > 0:
      dec agent.frozen
      env.updateObservations(AgentFrozenLayer, agent.pos, agent.frozen)
      inc env.stats[agent.agentId].agentFrozenTime
  
  # Update thing cooldowns
  for thing in env.things:
    if thing.cooldown > 0:
      dec thing.cooldown
      case thing.kind:
      of Mine:
        env.updateObservations(MineReadyLayer, thing.pos, thing.cooldown)
      of Altar:
        env.updateObservations(AltarReadyLayer, thing.pos, thing.cooldown)
      of Generator:
        env.updateObservations(GeneratorReadyLayer, thing.pos, if thing.cooldown == 0: 1 else: 0)
      else:
        discard
  
  # Move and spawn Clippys
  env.updateClippys()
  
  # Respawn dead agents at altars
  env.respawnAgents()

proc updateClippys*(env: Environment) =
  ## Update Clippy movement and spawning
  var r = initRand(env.currentStep)
  var clippysToRemove: seq[Thing] = @[]
  
  # Move existing Clippys
  for clippy in env.clippys:
    let oldPos = clippy.pos
    let direction = getClippyDirection(clippy.pos, cast[seq[pointer]](env.things), ClippyVisionRange, cast[seq[pointer]](env.clippys), r)
    var newPos = clippy.pos
    
    case direction:
    of 0: newPos.y -= 1  # North
    of 1: newPos.y += 1  # South
    of 2: newPos.x -= 1  # West
    of 3: newPos.x += 1  # East
    else: discard
    
    # Check bounds and collision
    if newPos.x >= 0 and newPos.x < env.mapWidth and
       newPos.y >= 0 and newPos.y < env.mapHeight:
      let target = env.grid[newPos.x][newPos.y]
      
      if isNil(target):
        # Move clippy
        clippy.pos = newPos
        env.grid[oldPos.x][oldPos.y] = nil
        env.grid[newPos.x][newPos.y] = clippy
      elif target.kind == Altar:
        # Damage altar and remove clippy
        if target.hearts > 0:
          dec target.hearts
          env.updateObservations(AltarHeartsLayer, target.pos, target.hearts)
        clippysToRemove.add(clippy)
        env.grid[oldPos.x][oldPos.y] = nil
      elif target.kind == Agent:
        # Combat with agent
        if r.rand(1.0) < 0.5:
          # Agent dies
          env.terminated[target.agentId] = 1.0
          env.grid[target.pos.x][target.pos.y] = nil
        clippysToRemove.add(clippy)
        env.grid[oldPos.x][oldPos.y] = nil
  
  # Remove dead clippys
  for clippy in clippysToRemove:
    let idx = env.clippys.find(clippy)
    if idx >= 0:
      env.clippys.del(idx)
  
  # Spawn new clippys from temples
  for thing in env.things:
    if thing.kind == Temple and thing.cooldown == 0:
      # Find empty position near temple
      let emptyPositions = env.findEmptyPositionsAround(thing.pos, 2)
      if emptyPositions.len > 0:
        let newClippy = Thing(
          kind: Clippy,
          pos: emptyPositions[0],
          hearts: 1,
          cooldown: 0
        )
        env.clippys.add(newClippy)
        env.grid[newClippy.pos.x][newClippy.pos.y] = newClippy
        thing.cooldown = TempleCooldown

proc respawnAgents*(env: Environment) =
  ## Respawn dead agents at their home altars
  for agentId in 0 ..< MapAgents:
    if env.terminated[agentId] > 0:
      let agent = env.agents[agentId]
      if agent.homeAltar.x >= 0:
        # Find the home altar
        var altar: Thing = nil
        for thing in env.things:
          if thing.kind == Altar and thing.pos == agent.homeAltar:
            altar = thing
            break
        
        # Respawn if altar has hearts
        if not isNil(altar) and altar.hearts >= MapObjectAltarRespawnCost:
          altar.hearts -= MapObjectAltarRespawnCost
          env.updateObservations(AltarHeartsLayer, altar.pos, altar.hearts)
          
          # Find empty position near altar
          let emptyPositions = env.findEmptyPositionsAround(altar.pos, 2)
          if emptyPositions.len > 0:
            agent.pos = emptyPositions[0]
            agent.frozen = 0
            agent.shield = false
            agent.inventory = 0
            env.terminated[agentId] = 0.0
            env.grid[agent.pos.x][agent.pos.y] = agent
            env.updateObservations(agentId)

proc findEmptyPositionsAround*(env: Environment, center: IVec2, radius: int): seq[IVec2] =
  ## Find empty positions around a center point
  result = @[]
  for dx in -radius .. radius:
    for dy in -radius .. radius:
      if dx == 0 and dy == 0:
        continue
      let pos = center + toIVec2(dx, dy)
      if pos.x >= 0 and pos.x < env.mapWidth and
         pos.y >= 0 and pos.y < env.mapHeight and
         isNil(env.grid[pos.x][pos.y]):
        result.add(pos)

proc init*(env: Environment) =
  ## Initialize the simplified environment
  env.mapWidth = MapLayoutWidth
  env.mapHeight = MapLayoutHeight
  env.currentStep = 0
  
  # Initialize arrays
  env.agents.setLen(MapAgents)
  env.things = @[]
  env.clippys = @[]
  env.stats.setLen(MapAgents)
  env.terminated.setLen(MapAgents)
  env.truncated.setLen(MapAgents)
  env.observations.setLen(ObservationLayers)
  
  for layer in 0 ..< ObservationLayers:
    env.observations[layer].setLen(env.mapWidth)
    for x in 0 ..< env.mapWidth:
      env.observations[layer][x].setLen(env.mapHeight)
  
  env.grid.init(env.mapWidth, env.mapHeight)
  env.terrain.initTerrain(env.mapWidth, env.mapHeight, MapBorder)
  
  # Initialize stats
  for i in 0 ..< MapAgents:
    env.stats[i] = Stats()
  
  # Spawn houses, agents, mines, generators, temples, etc.
  # (Implementation would continue here with simplified spawning logic)

proc reset*(env: Environment) =
  ## Reset the environment
  env.currentStep = 0
  env.terminated.clear()
  env.truncated.clear()
  env.things.setLen(0)
  env.agents.setLen(0)
  env.clippys.setLen(0)
  env.stats.setLen(0)
  env.grid.clear()
  env.terrain.clear()
  env.observations.clear()
  env.init()

proc getEpisodeStats*(env: Environment): string =
  ## Get episode stats as a string
  if env.stats.len == 0:
    return "{}"
  
  var totalStats = Stats()
  for stat in env.stats:
    totalStats.actionInvalid += stat.actionInvalid
    totalStats.actionAttack += stat.actionAttack
    totalStats.actionAttackAgent += stat.actionAttackAgent
    totalStats.actionAttackAltar += stat.actionAttackAltar
    totalStats.actionMine += stat.actionMine
    totalStats.actionShield += stat.actionShield
    totalStats.actionUse += stat.actionUse
    totalStats.agentFrozenTime += stat.agentFrozenTime
    totalStats.collectedOre += stat.collectedOre
    totalStats.depositedHearts += stat.depositedHearts
  
  return fmt"""{{
    "invalid": {totalStats.actionInvalid},
    "attack": {totalStats.actionAttack},
    "attack_agent": {totalStats.actionAttackAgent},
    "attack_altar": {totalStats.actionAttackAltar},
    "mine": {totalStats.actionMine},
    "shield": {totalStats.actionShield},
    "use": {totalStats.actionUse},
    "frozen_time": {totalStats.agentFrozenTime},
    "ore_collected": {totalStats.collectedOre},
    "hearts_deposited": {totalStats.depositedHearts}
  }}"""