import std/[tables, math], vmath, chroma
import rng_compat
import terrain, objects, common
export terrain, objects, common


const
  # Map Layout
  MapLayoutRoomsX* = 1
  MapLayoutRoomsY* = 1
  MapBorder* = 4
  MapRoomWidth* = 192  # 16:9 aspect ratio
  MapRoomHeight* = 108
  MapRoomBorder* = 0

  # World Objects
  MapRoomObjectsHouses* = 12
  MapAgentsPerHouse* = 5
  MapRoomObjectsAgents* = MapRoomObjectsHouses * MapAgentsPerHouse  # 60 total agents
  MapRoomObjectsConverters* = 10
  MapRoomObjectsMines* = 20
  MapRoomObjectsWalls* = 30

  # Agent Parameters
  MapObjectAgentMaxInventory* = 5
  MapObjectAgentFreezeDuration* = 10

  # Building Parameters
  MapObjectAltarInitialHearts* = 5
  MapObjectAltarCooldown* = 10
  MapObjectAltarRespawnCost* = 1
  MapObjectConverterCooldown* = 0
  MapObjectMineCooldown* = 5
  MapObjectMineInitialResources* = 30
  MapObjectMineUseCost* = 0

  # Gameplay
  SpawnerCooldown* = 13
  MinTintEpsilon* = 5

  # Observation System
  ObservationLayers* = 21
  ObservationWidth* = 11
  ObservationHeight* = 11

  # Reward System
  RewardGetWater* = 0.001
  RewardGetWheat* = 0.001
  RewardGetWood* = 0.002
  RewardMineOre* = 0.003
  RewardConvertOreToBattery* = 0.01
  RewardCraftSpear* = 0.01
  RewardCraftArmor* = 0.015
  RewardCraftFood* = 0.012
  RewardCraftCloth* = 0.012
  RewardDestroyTumor* = 0.1

  # Computed Values
  MapAgents* = MapRoomObjectsAgents * MapLayoutRoomsX * MapLayoutRoomsY
  MapWidth* = MapLayoutRoomsX * (MapRoomWidth + MapRoomBorder) + MapBorder
  MapHeight* = MapLayoutRoomsY * (MapRoomHeight + MapRoomBorder) + MapBorder

  # Compile-time optimization constants
  ObservationRadius* = ObservationWidth div 2  # 5 - computed once
  MapAgentsPerHouseFloat* = MapAgentsPerHouse.float32  # Avoid runtime conversion
  MapBorderMinus1* = MapBorder - 1  # For bounds checking

{.push inline.}
proc getTeamId*(agentId: int): int =
  ## Inline team ID calculation - frequently used
  agentId div MapAgentsPerHouse

proc getTeamIdByte*(agentId: int): uint8 =
  ## Inline team ID as uint8 for observations
  (agentId div MapAgentsPerHouse + 1).uint8

template isValidPos*(pos: IVec2): bool =
  ## Inline bounds checking template - very frequently used
  pos.x >= 0 and pos.x < MapWidth and pos.y >= 0 and pos.y < MapHeight

template safeTintAdd*(tintMod: var int16, delta: int): void =
  ## Safe tint accumulation with overflow protection
  tintMod = max(-32000'i16, min(32000'i16, tintMod + delta.int16))
{.pop.}


# Global village color management
var agentVillageColors*: seq[Color] = @[]
var teamColors*: seq[Color] = @[]
var altarColors*: Table[IVec2, Color] = initTable[IVec2, Color]()

const WarmVillagePalette* = [
  color(1.00, 0.20, 0.15, 1.0),  # fire red
  color(0.95, 0.32, 0.10, 1.0),  # vermilion
  color(1.00, 0.48, 0.00, 1.0),  # tangerine
  color(1.00, 0.64, 0.05, 1.0),  # amber
  color(1.00, 0.78, 0.15, 1.0),  # sunflower
  color(1.00, 0.64, 0.38, 1.0),  # peach
  color(0.98, 0.46, 0.34, 1.0),  # coral
  color(0.93, 0.29, 0.28, 1.0),  # crimson rose
  color(0.95, 0.35, 0.45, 1.0),  # watermelon
  color(0.90, 0.23, 0.58, 1.0),  # hot magenta
  color(0.88, 0.40, 0.18, 1.0),  # copper
  color(0.97, 0.54, 0.22, 1.0)   # persimmon
]


type
  ObservationName* = enum
    AgentLayer = 0        # Team-aware: 0=empty, 1=team0, 2=team1, 3=team2, 255=Tumor
    AgentOrientationLayer = 1
    AgentInventoryOreLayer = 2
    AgentInventoryBatteryLayer = 3
    AgentInventoryWaterLayer = 4
    AgentInventoryWheatLayer = 5
    AgentInventoryWoodLayer = 6
    AgentInventorySpearLayer = 7
    AgentInventoryLanternLayer = 8
    AgentInventoryArmorLayer = 9
    WallLayer = 10
    MineLayer = 11
    MineResourceLayer = 12
    MineReadyLayer = 13
    ConverterLayer = 14  # Renamed from Converter
    ConverterReadyLayer = 15
    AltarLayer = 16
    AltarHeartsLayer = 17  # Hearts for respawning
    AltarReadyLayer = 18
    TintLayer = 19        # Unified tint layer for all environmental effects
    AgentInventoryBreadLayer = 20  # Bread baked from clay oven


  ThingKind* = enum
    Agent
    Wall
    Mine
    Converter  # Converts ore to batteries
    Altar
    Spawner
    Tumor
    Armory
    Forge
    ClayOven
    WeavingLoom
    PlantedLantern  # Planted lanterns that spread team colors

  Thing* = ref object
    kind*: ThingKind
    pos*: IVec2
    id*: int
    layer*: int
    hearts*: int  # For altars only - used for respawning agents
    resources*: int  # For mines - remaining ore
    cooldown*: int
    frozen*: int

    # Agent:
    agentId*: int
    orientation*: Orientation
    inventoryOre*: int      # Ore from mines
    inventoryBattery*: int  # Batteries from converters
    inventoryWater*: int    # Water from water tiles
    inventoryWheat*: int    # Wheat from wheat tiles
    inventoryWood*: int     # Wood from tree tiles
    inventorySpear*: int    # Spears crafted from forge
    inventoryLantern*: int  # Lanterns from weaving loom (plantable team markers)
    inventoryArmor*: int    # Armor from armory (5-hit protection, tracks remaining uses)
    inventoryBread*: int    # Bread baked from clay oven
    reward*: float32
    homeAltar*: IVec2      # Position of agent's home altar for respawning
    # Tumor:
    homeSpawner*: IVec2     # Position of tumor's home spawner
    hasClaimedTerritory*: bool  # Whether this tumor has already branched and is now inert
    turnsAlive*: int            # Number of turns this tumor has been alive
    
    # PlantedLantern:
    teamId*: int               # Which team this lantern belongs to (for color spreading)
    lanternHealthy*: bool      # Whether lantern is active (not destroyed by tumor)
    
    # Spawner: (no longer needs altar targeting for new creep spread behavior)

  Stats* = ref object
    # Agent Stats - simplified actions:
    actionInvalid*: int
    actionNoop*: int     # Action 0: NOOP
    actionMove*: int     # Action 1: MOVE  
    actionAttack*: int   # Action 2: ATTACK
    actionUse*: int      # Action 3: USE (terrain/buildings)
    actionSwap*: int     # Action 4: SWAP
    actionPlant*: int    # Action 6: PLANT lantern
    actionPut*: int      # Action 5: GIVE to teammate

  TileColor* = object
    r*, g*, b*: float32      # RGB color components  
    intensity*: float32      # Overall intensity/brightness modifier
  
  # Tint modification layers for efficient batch updates
  TintModification* = object
    r*, g*, b*: int16       # Delta values to add (scaled by 1000)
  
  # Track active tiles for sparse processing
  ActiveTiles* = object
    positions*: seq[IVec2]  # Linear list of active tiles
    flags*: array[MapWidth, array[MapHeight, bool]]  # Dedup mask per tile

  # Configuration structure for environment - ONLY runtime parameters
  # Structural constants (map size, agent count, observation dimensions) remain compile-time constants
  EnvironmentConfig* = object
    # Core game parameters
    maxSteps*: int
    
    # Resource configuration
    orePerBattery*: int
    batteriesPerHeart*: int
    
    # Combat configuration
    enableCombat*: bool
    tumorSpawnRate*: float
    tumorDamage*: int
    
    # Reward configuration
    heartReward*: float
    oreReward*: float
    batteryReward*: float
    woodReward*: float
    waterReward*: float
    wheatReward*: float
    spearReward*: float
    armorReward*: float
    foodReward*: float
    clothReward*: float
    tumorKillReward*: float
    survivalPenalty*: float
    deathPenalty*: float
    
  Environment* = ref object
    currentStep*: int
    config*: EnvironmentConfig  # Configuration for this environment
    shouldReset*: bool  # Track if environment needs reset
    things*: seq[Thing]
    agents*: seq[Thing]
    grid*: array[MapWidth, array[MapHeight, Thing]]
    terrain*: TerrainGrid
    tileColors*: array[MapWidth, array[MapHeight, TileColor]]  # Main color array
    baseTileColors*: array[MapWidth, array[MapHeight, TileColor]]  # Base colors (terrain)
    tintMods*: array[MapWidth, array[MapHeight, TintModification]]  # Unified tint modifications  
    activeTiles*: ActiveTiles  # Sparse list of tiles to process
    observations*: array[
      MapAgents,
      array[ObservationLayers,
        array[ObservationWidth, array[ObservationHeight, uint8]]
      ]
    ]
    terminated*: array[MapAgents, float32]
    truncated*: array[MapAgents, float32]
    stats: seq[Stats]

var
  env*: Environment  # Global environment instance
  selection*: Thing  # Currently selected entity for UI interaction

# Frozen building detection
proc isBuildingFrozen*(pos: IVec2, env: Environment): bool =
  ## Enhanced check if a building is frozen due to tumor creep zone effect
  if pos.x < 0 or pos.x >= MapWidth or pos.y < 0 or pos.y >= MapHeight:
    return false
  
  let color = env.tileColors[pos.x][pos.y]
  # Cool colors have more blue than red+green combined
  let basicCoolCheck = color.b > (color.r + color.g)
  
  # Additional saturation check: blue should be significantly high (close to max saturation)  
  # This indicates prolonged presence in a tumor creep zone
  let highSaturationCheck = color.b >= 1.0  # Blue component near maximum
  
  return basicCoolCheck and highSaturationCheck

proc render*(env: Environment): string =
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
            cell = "A"
          of Wall:
            cell = "#"
          of Mine:
            cell = "m"
          of Converter:
            cell = "g"
          of Altar:
            cell = "a"
          of Spawner:
            cell = "t"
          of Tumor:
            cell = "C"
          of Armory:
            cell = "A"
          of Forge:
            cell = "F"
          of ClayOven:
            cell = "O"
          of WeavingLoom:
            cell = "W"
          of PlantedLantern:
            cell = "L"
          break
      result.add(cell)
    result.add("\n")


proc clear[T](s: var openarray[T]) =
  let p = cast[pointer](s[0].addr)
  zeroMem(p, s.len * sizeof(T))



{.push inline.}
proc updateObservations(
  env: Environment,
  layer: ObservationName,
  pos: IVec2,
  value: int
) =
  ## Ultra-optimized observation update - early bailout and minimal calculations
  let layerId = ord(layer)

  # Ultra-fast observation update with minimal calculations

  # Still need to check all agents but with optimized early exit
  let agentCount = env.agents.len
  for agentId in 0 ..< agentCount:
    let agentPos = env.agents[agentId].pos

    # Ultra-fast bounds check using compile-time constants
    let dx = pos.x - agentPos.x
    let dy = pos.y - agentPos.y
    if dx < -ObservationRadius or dx > ObservationRadius or
       dy < -ObservationRadius or dy > ObservationRadius:
      continue

    let x = dx + ObservationRadius
    let y = dy + ObservationRadius
    var agentLayer = addr env.observations[agentId][layerId]
    agentLayer[][x][y] = value.uint8
{.pop.}


{.push inline.}
proc getThing(env: Environment, pos: IVec2): Thing =
  if not isValidPos(pos):
    return nil
  return env.grid[pos.x][pos.y]

proc isEmpty*(env: Environment, pos: IVec2): bool =
  if not isValidPos(pos):
    return false
  return env.grid[pos.x][pos.y] == nil
{.pop.}



proc createTumor(pos: IVec2, homeSpawner: IVec2, r: var Rand): Thing =
  ## Create a new Tumor seed that can branch once before turning inert
  Thing(
    kind: Tumor,
    pos: pos,
    orientation: Orientation(randIntInclusive(r, 0, 3)),
    homeSpawner: homeSpawner,
    hasClaimedTerritory: false,  # Start mobile, will plant when far enough from others
    turnsAlive: 0                # New tumor hasn't lived any turns yet
  )






proc noopAction(env: Environment, id: int, agent: Thing) =
  inc env.stats[id].actionNoop


proc moveAction(env: Environment, id: int, agent: Thing, argument: int) =
  if argument < 0 or argument > 7:
    inc env.stats[id].actionInvalid
    return
  
  let moveOrientation = Orientation(argument)
  let delta = getOrientationDelta(moveOrientation)
  
  var newPos = agent.pos
  newPos.x += int32(delta.x)
  newPos.y += int32(delta.y)
  
  let newOrientation = moveOrientation
  # Allow walking through planted lanterns by relocating the lantern, preferring push direction (up to 2 tiles ahead)
  var canMove = env.isEmpty(newPos)
  if not canMove:
    let blocker = env.getThing(newPos)
    if not isNil(blocker) and blocker.kind == PlantedLantern:
      var relocated = false
      # Preferred push positions in move direction
      let ahead1 = ivec2(newPos.x + delta.x, newPos.y + delta.y)
      let ahead2 = ivec2(newPos.x + delta.x * 2, newPos.y + delta.y * 2)
      if ahead2.x >= 0 and ahead2.x < MapWidth and ahead2.y >= 0 and ahead2.y < MapHeight and env.isEmpty(ahead2) and env.terrain[ahead2.x][ahead2.y] != Water:
        env.grid[blocker.pos.x][blocker.pos.y] = nil
        blocker.pos = ahead2
        env.grid[blocker.pos.x][blocker.pos.y] = blocker
        relocated = true
      elif ahead1.x >= 0 and ahead1.x < MapWidth and ahead1.y >= 0 and ahead1.y < MapHeight and env.isEmpty(ahead1) and env.terrain[ahead1.x][ahead1.y] != Water:
        env.grid[blocker.pos.x][blocker.pos.y] = nil
        blocker.pos = ahead1
        env.grid[blocker.pos.x][blocker.pos.y] = blocker
        relocated = true
      # Fallback to any adjacent empty tile around the lantern
      if not relocated:
        for dy in -1 .. 1:
          for dx in -1 .. 1:
            if dx == 0 and dy == 0: continue
            let alt = ivec2(newPos.x + dx, newPos.y + dy)
            if alt.x < 0 or alt.y < 0 or alt.x >= MapWidth or alt.y >= MapHeight: continue
            if env.isEmpty(alt) and env.terrain[alt.x][alt.y] != Water:
              env.grid[blocker.pos.x][blocker.pos.y] = nil
              blocker.pos = alt
              env.grid[blocker.pos.x][blocker.pos.y] = blocker
              relocated = true
              break
          if relocated: break
      if relocated:
        canMove = true

  if canMove:
    env.grid[agent.pos.x][agent.pos.y] = nil
    # Clear old position and set new position
    env.updateObservations(AgentLayer, agent.pos, 0)  # Clear old
    agent.pos = newPos
    agent.orientation = newOrientation
    env.grid[agent.pos.x][agent.pos.y] = agent

    # Update observations for new position only
    env.updateObservations(AgentLayer, agent.pos, getTeamId(agent.agentId) + 1)
    env.updateObservations(AgentOrientationLayer, agent.pos, agent.orientation.int)
    inc env.stats[id].actionMove
  else:
    inc env.stats[id].actionInvalid

proc transferAgentInventory(env: Environment, killer, victim: Thing) =
  ## Move the victim's inventory to the killer before the victim dies
  template moveAll(field: untyped, layer: ObservationName) =
    if victim.field > 0:
      killer.field += victim.field
      victim.field = 0
      env.updateObservations(layer, killer.pos, killer.field)

  moveAll(inventoryOre, AgentInventoryOreLayer)
  moveAll(inventoryBattery, AgentInventoryBatteryLayer)
  moveAll(inventoryWater, AgentInventoryWaterLayer)
  moveAll(inventoryWheat, AgentInventoryWheatLayer)
  moveAll(inventoryWood, AgentInventoryWoodLayer)
  moveAll(inventorySpear, AgentInventorySpearLayer)
  moveAll(inventoryLantern, AgentInventoryLanternLayer)
  moveAll(inventoryArmor, AgentInventoryArmorLayer)
  moveAll(inventoryBread, AgentInventoryBreadLayer)

proc killAgent(env: Environment, victim: Thing) =
  ## Remove an agent from the board and mark for respawn
  if victim.frozen >= 999999:
    return

  env.grid[victim.pos.x][victim.pos.y] = nil
  env.updateObservations(AgentLayer, victim.pos, 0)
  env.updateObservations(AgentOrientationLayer, victim.pos, 0)
  env.updateObservations(AgentInventoryOreLayer, victim.pos, 0)
  env.updateObservations(AgentInventoryBatteryLayer, victim.pos, 0)
  env.updateObservations(AgentInventoryWaterLayer, victim.pos, 0)
  env.updateObservations(AgentInventoryWheatLayer, victim.pos, 0)
  env.updateObservations(AgentInventoryWoodLayer, victim.pos, 0)
  env.updateObservations(AgentInventorySpearLayer, victim.pos, 0)
  env.updateObservations(AgentInventoryLanternLayer, victim.pos, 0)
  env.updateObservations(AgentInventoryArmorLayer, victim.pos, 0)
  env.updateObservations(AgentInventoryBreadLayer, victim.pos, 0)

  env.terminated[victim.agentId] = 1.0
  victim.frozen = 999999
  victim.reward += env.config.deathPenalty

  victim.inventoryOre = 0
  victim.inventoryBattery = 0
  victim.inventoryWater = 0
  victim.inventoryWheat = 0
  victim.inventoryWood = 0
  victim.inventorySpear = 0
  victim.inventoryLantern = 0
  victim.inventoryArmor = 0
  victim.inventoryBread = 0

proc attackAction(env: Environment, id: int, agent: Thing, argument: int) =
  ## Attack an entity in one of eight directions. Spears extend range to 2 tiles.
  if argument < 0 or argument > 7:
    inc env.stats[id].actionInvalid
    return

  let attackOrientation = Orientation(argument)
  let delta = getOrientationDelta(attackOrientation)
  let hasSpear = agent.inventorySpear > 0
  let maxRange = if hasSpear: 2 else: 1

  var attackHit = false

  for distance in 1 .. maxRange:
    let attackPos = agent.pos + ivec2(delta.x * distance, delta.y * distance)
    if attackPos.x < 0 or attackPos.x >= MapWidth or attackPos.y < 0 or attackPos.y >= MapHeight:
      continue

    let target = env.getThing(attackPos)
    if isNil(target):
      continue

    case target.kind
    of Tumor:
      env.grid[attackPos.x][attackPos.y] = nil
      env.updateObservations(AgentLayer, attackPos, 0)
      env.updateObservations(AgentOrientationLayer, attackPos, 0)
      let idx = env.things.find(target)
      if idx >= 0:
        env.things.del(idx)
      agent.reward += env.config.tumorKillReward
      attackHit = true
    of Spawner:
      env.grid[attackPos.x][attackPos.y] = nil
      let idx = env.things.find(target)
      if idx >= 0:
        env.things.del(idx)
      attackHit = true
    of Agent:
      if target.agentId == agent.agentId:
        continue
      if getTeamId(target.agentId) == getTeamId(agent.agentId):
        continue
      env.transferAgentInventory(agent, target)
      env.killAgent(target)
      attackHit = true
    else:
      discard

    if attackHit:
      break

  if attackHit:
    if hasSpear:
      agent.inventorySpear = max(0, agent.inventorySpear - 1)
      env.updateObservations(AgentInventorySpearLayer, agent.pos, agent.inventorySpear)
    inc env.stats[id].actionAttack
  else:
    inc env.stats[id].actionInvalid

proc useAction(env: Environment, id: int, agent: Thing, argument: int) =
  ## Use terrain or building with a single action (requires holding needed resource if any)
  if argument > 7:
    inc env.stats[id].actionInvalid
    return
  
  # Calculate target position based on orientation argument
  let useOrientation = Orientation(argument)
  let delta = getOrientationDelta(useOrientation)
  var targetPos = agent.pos
  targetPos.x += int32(delta.x)
  targetPos.y += int32(delta.y)
  
  # Check bounds
  if targetPos.x < 0 or targetPos.x >= MapWidth or targetPos.y < 0 or targetPos.y >= MapHeight:
    inc env.stats[id].actionInvalid
    return

  # Terrain use first
  case env.terrain[targetPos.x][targetPos.y]:
  of Water:
    if agent.inventoryWater < MapObjectAgentMaxInventory:
      agent.inventoryWater += 1
      env.terrain[targetPos.x][targetPos.y] = Empty
      agent.reward += env.config.waterReward
      inc env.stats[id].actionUse
      return
    else:
      inc env.stats[id].actionInvalid
      return
  of Wheat:
    if agent.inventoryWheat < MapObjectAgentMaxInventory:
      agent.inventoryWheat += 1
      env.terrain[targetPos.x][targetPos.y] = Empty
      agent.reward += env.config.wheatReward
      inc env.stats[id].actionUse
      return
    else:
      inc env.stats[id].actionInvalid
      return
  of Tree:
    if agent.inventoryWood < MapObjectAgentMaxInventory:
      agent.inventoryWood += 1
      env.terrain[targetPos.x][targetPos.y] = Empty
      agent.reward += env.config.woodReward
      inc env.stats[id].actionUse
      return
    else:
      inc env.stats[id].actionInvalid
      return
  of Empty:
    discard

  # Building use
  let thing = env.getThing(targetPos)
  if isNil(thing):
    inc env.stats[id].actionInvalid
    return
  # Prevent using frozen buildings
  if isBuildingFrozen(targetPos, env):
    inc env.stats[id].actionInvalid
    return

  case thing.kind:
  of Mine:
    if thing.cooldown == 0 and agent.inventoryOre < MapObjectAgentMaxInventory:
      agent.inventoryOre += 1
      env.updateObservations(AgentInventoryOreLayer, agent.pos, agent.inventoryOre)
      thing.cooldown = MapObjectMineCooldown
      env.updateObservations(MineReadyLayer, thing.pos, thing.cooldown)
      if agent.inventoryOre == 1: agent.reward += env.config.oreReward
      inc env.stats[id].actionUse
    else:
      inc env.stats[id].actionInvalid
  of Converter:
    if thing.cooldown == 0 and agent.inventoryOre > 0 and agent.inventoryBattery < MapObjectAgentMaxInventory:
      agent.inventoryOre -= 1
      agent.inventoryBattery += 1
      env.updateObservations(AgentInventoryOreLayer, agent.pos, agent.inventoryOre)
      env.updateObservations(AgentInventoryBatteryLayer, agent.pos, agent.inventoryBattery)
      thing.cooldown = 0
      env.updateObservations(ConverterReadyLayer, thing.pos, 1)
      if agent.inventoryBattery == 1: agent.reward += env.config.batteryReward
      inc env.stats[id].actionUse
    else:
      inc env.stats[id].actionInvalid
  of Forge:
    if thing.cooldown == 0 and agent.inventoryWood > 0 and agent.inventorySpear == 0:
      agent.inventoryWood -= 1
      agent.inventorySpear = 5
      thing.cooldown = 5
      agent.reward += env.config.spearReward
      inc env.stats[id].actionUse
    else:
      inc env.stats[id].actionInvalid
  of WeavingLoom:
    if thing.cooldown == 0 and agent.inventoryWheat > 0 and agent.inventoryLantern == 0:
      agent.inventoryWheat -= 1
      agent.inventoryLantern = 1
      thing.cooldown = 15
      agent.reward += env.config.clothReward
      inc env.stats[id].actionUse
    else:
      inc env.stats[id].actionInvalid
  of Armory:
    if thing.cooldown == 0 and agent.inventoryWood > 0 and agent.inventoryArmor == 0:
      agent.inventoryWood -= 1
      agent.inventoryArmor = 5
      thing.cooldown = 20
      agent.reward += env.config.armorReward
      inc env.stats[id].actionUse
    else:
      inc env.stats[id].actionInvalid
  of ClayOven:
    if thing.cooldown == 0 and agent.inventoryWheat > 0:
      agent.inventoryWheat -= 1
      agent.inventoryBread += 1
      thing.cooldown = 10
      # No observation layer for bread; optional for UI later
      agent.reward += env.config.foodReward
      inc env.stats[id].actionUse
    else:
      inc env.stats[id].actionInvalid
  of Altar:
    if thing.cooldown == 0 and agent.inventoryBattery >= 1:
      agent.inventoryBattery -= 1
      env.updateObservations(AgentInventoryBatteryLayer, agent.pos, agent.inventoryBattery)
      thing.hearts += 1
      thing.cooldown = MapObjectAltarCooldown
      env.updateObservations(AltarHeartsLayer, thing.pos, thing.hearts)
      env.updateObservations(AltarReadyLayer, thing.pos, thing.cooldown)
      agent.reward += env.config.heartReward
      inc env.stats[id].actionUse
    else:
      inc env.stats[id].actionInvalid
  else:
    inc env.stats[id].actionInvalid

proc swapAction(env: Environment, id: int, agent: Thing, argument: int) =
  ## Swap
  if argument > 1:
    inc env.stats[id].actionInvalid
    return
  let
    targetPos = agent.pos + orientationToVec(agent.orientation)
    target = env.getThing(targetPos)
  if target == nil:
    inc env.stats[id].actionInvalid
    return
  if target.kind == Agent and target.frozen > 0:
    var temp = agent.pos
    agent.pos = target.pos
    target.pos = temp
    inc env.stats[id].actionSwap
    # REMOVED: expensive per-agent full grid rebuilds
  else:
    inc env.stats[id].actionInvalid



proc putAction(env: Environment, id: int, agent: Thing, argument: int) =
  ## Give items to adjacent teammate. Argument is direction (0..7)
  if argument > 7:
    inc env.stats[id].actionInvalid
    return
  let dir = Orientation(argument)
  let delta = getOrientationDelta(dir)
  let targetPos = ivec2(agent.pos.x + delta.x.int32, agent.pos.y + delta.y.int32)
  if targetPos.x < 0 or targetPos.x >= MapWidth or targetPos.y < 0 or targetPos.y >= MapHeight:
    inc env.stats[id].actionInvalid
    return
  let target = env.getThing(targetPos)
  if isNil(target) or target.kind != Agent:
    inc env.stats[id].actionInvalid
    return
  var transferred = false
  # Give armor if we have any and target has none
  if agent.inventoryArmor > 0 and target.inventoryArmor == 0:
    target.inventoryArmor = agent.inventoryArmor
    agent.inventoryArmor = 0
    transferred = true
  # Otherwise give food if possible (no obs layer yet)
  elif agent.inventoryBread > 0 and target.inventoryBread < MapObjectAgentMaxInventory:
    let giveAmt = min(agent.inventoryBread, MapObjectAgentMaxInventory - target.inventoryBread)
    agent.inventoryBread -= giveAmt
    target.inventoryBread += giveAmt
    transferred = true
  if transferred:
    inc env.stats[id].actionPut
    # Update observations for changed inventories
  else:
    inc env.stats[id].actionInvalid
# ============== CLIPPY AI ==============




{.push inline.}
proc isValidEmptyPosition(env: Environment, pos: IVec2): bool =
  ## Check if a position is within map bounds, empty, and not water
  pos.x >= MapBorder and pos.x < MapWidth - MapBorder and
  pos.y >= MapBorder and pos.y < MapHeight - MapBorder and
  env.isEmpty(pos) and env.terrain[pos.x][pos.y] != Water

proc generateRandomMapPosition(r: var Rand): IVec2 =
  ## Generate a random position within map boundaries
  ivec2(
    int32(randIntExclusive(r, MapBorder, MapWidth - MapBorder)),
    int32(randIntExclusive(r, MapBorder, MapHeight - MapBorder))
  )
{.pop.}

proc findEmptyPositionsAround*(env: Environment, center: IVec2, radius: int): seq[IVec2] =
  ## Find empty positions around a center point within a given radius
  result = @[]
  for dx in -radius .. radius:
    for dy in -radius .. radius:
      if dx == 0 and dy == 0:
        continue  # Skip the center position
      let pos = ivec2(center.x + dx, center.y + dy)
      if env.isValidEmptyPosition(pos):
        result.add(pos)

proc findFirstEmptyPositionAround*(env: Environment, center: IVec2, radius: int): IVec2 =
  ## Find first empty position around center (no allocation)
  for dx in -radius .. radius:
    for dy in -radius .. radius:
      if dx == 0 and dy == 0:
        continue  # Skip the center position
      let pos = ivec2(center.x + dx, center.y + dy)
      if env.isValidEmptyPosition(pos):
        return pos
  return ivec2(-1, -1)  # No empty position found

# ============== LANTERN PLACEMENT ==============

proc findLanternPlacementSpot*(env: Environment, agent: Thing, controller: pointer): IVec2 =
  ## Find a good spot to place a lantern (7+ tiles from altar, 2+ tiles from other lanterns)
  let homeAltar = agent.homeAltar
  
  # Search in expanding rings around the agent
  for radius in 1 .. 15:
    for dx in -radius .. radius:
      for dy in -radius .. radius:
        if abs(dx) != radius and abs(dy) != radius:
          continue  # Only check perimeter of ring
        
        let candidate = agent.pos + ivec2(dx, dy)
        
        # Must be valid empty position
        if not env.isValidEmptyPosition(candidate):
          continue
          
        # Must be 3+ tiles from home altar (reduced for testing)
        let distToAltar = manhattanDistance(candidate, homeAltar)
        if distToAltar < 3:
          continue
          
        # Fast grid-based lantern proximity check (avoid scanning all things)
        var tooCloseToLantern = false
        for dx in -1..1:
          for dy in -1..1:
            let checkPos = candidate + ivec2(dx, dy)
            if isValidPos(checkPos):
              let thing = env.getThing(checkPos)
              if not isNil(thing) and thing.kind == PlantedLantern:
                tooCloseToLantern = true
                break
          if tooCloseToLantern: break
        
        if not tooCloseToLantern:
          return candidate  # Found a good spot!
  
  # No good spot found
  return ivec2(-1, -1)

# ============== CLIPPY CREEP SPREAD AI ==============


const
  TumorBranchRange = 5
  TumorBranchMinAge = 2
  TumorBranchChance = 0.1
  TumorAdjacencyDeathChance = 1.0 / 3.0

proc findTumorBranchTarget(tumor: Thing, env: Environment, r: var Rand): IVec2 =
  ## Pick a random empty tile within the tumor's branching range
  var candidates: seq[IVec2] = @[]

  for dx in -TumorBranchRange .. TumorBranchRange:
    for dy in -TumorBranchRange .. TumorBranchRange:
      if dx == 0 and dy == 0:
        continue
      if max(abs(dx), abs(dy)) > TumorBranchRange:
        continue
      let candidate = ivec2(tumor.pos.x + dx, tumor.pos.y + dy)
      if not env.isValidEmptyPosition(candidate):
        continue

      var adjacentTumor = false
      for adj in [ivec2(0, -1), ivec2(1, 0), ivec2(0, 1), ivec2(-1, 0)]:
        let checkPos = candidate + adj
        if not isValidPos(checkPos):
          continue
        let occupant = env.getThing(checkPos)
        if not isNil(occupant) and occupant.kind == Tumor:
          adjacentTumor = true
          break
      if not adjacentTumor:
        candidates.add(candidate)

  if candidates.len == 0:
    return ivec2(-1, -1)

  return candidates[randIntExclusive(r, 0, candidates.len)]

proc randomEmptyPos(r: var Rand, env: Environment): IVec2 =
  # Try with moderate attempts first
  for i in 0 ..< 100:
    let pos = r.generateRandomMapPosition()
    if env.isValidEmptyPosition(pos):
      return pos
  # Try harder with more attempts
  for i in 0 ..< 1000:
    let pos = r.generateRandomMapPosition()
    if env.isValidEmptyPosition(pos):
      return pos
  quit("Failed to find an empty position, map too full!")

proc clearTintModifications(env: Environment) =
  ## Clear only active tile modifications for performance
  for pos in env.activeTiles.positions:
    let tileX = pos.x.int
    let tileY = pos.y.int
    if tileX >= 0 and tileX < MapWidth and tileY >= 0 and tileY < MapHeight:
      env.tintMods[tileX][tileY] = TintModification(r: 0, g: 0, b: 0)
      env.activeTiles.flags[tileX][tileY] = false

  # Clear the active list for next frame
  env.activeTiles.positions.setLen(0)

proc updateTintModifications(env: Environment) =
  ## Update unified tint modification array based on entity positions - runs every frame
  # Clear previous frame's modifications
  env.clearTintModifications()
  
  template markActiveTile(tileX, tileY: int) =
    if tileX >= 0 and tileX < MapWidth and tileY >= 0 and tileY < MapHeight:
      if not env.activeTiles.flags[tileX][tileY]:
        env.activeTiles.flags[tileX][tileY] = true
        env.activeTiles.positions.add(ivec2(tileX, tileY))
  
  # Process all entities and mark their affected positions as active
  for thing in env.things:
    let pos = thing.pos
    if pos.x < 0 or pos.x >= MapWidth or pos.y < 0 or pos.y >= MapHeight:
      continue
    let baseX = pos.x.int
    let baseY = pos.y.int
    
    case thing.kind
    of Tumor:
      # Tumors create creep spread in 5x5 area (active seeds glow brighter)
      let creepIntensity = if thing.hasClaimedTerritory: 2 else: 1
      
      for dx in -2 .. 2:
        for dy in -2 .. 2:
          let tileX = baseX + dx
          let tileY = baseY + dy
          if tileX >= 0 and tileX < MapWidth and tileY >= 0 and tileY < MapHeight:
            # Distance-based falloff for more organic look
            let distance = abs(dx) + abs(dy)  # Manhattan distance
            let falloff = max(1, 5 - distance)  # Stronger at center, weaker at edges (5x5 grid)
            markActiveTile(tileX, tileY)
            
            # Tumor creep effect with overflow protection
            safeTintAdd(env.tintMods[tileX][tileY].r, -15 * creepIntensity * falloff)
            safeTintAdd(env.tintMods[tileX][tileY].g, -8 * creepIntensity * falloff)
            safeTintAdd(env.tintMods[tileX][tileY].b, 20 * creepIntensity * falloff)
      
    of Agent:
      # Agents create 5x stronger warmth in 3x3 area based on their tribe color
      let tribeId = thing.agentId
      if tribeId < agentVillageColors.len:
        let tribeColor = agentVillageColors[tribeId]
        
        for dx in -1 .. 1:
          for dy in -1 .. 1:
            let tileX = baseX + dx
            let tileY = baseY + dy
            if tileX >= 0 and tileX < MapWidth and tileY >= 0 and tileY < MapHeight:
              # Distance-based falloff
              let distance = abs(dx) + abs(dy)
              let falloff = max(1, 3 - distance)  # Stronger at center, weaker at edges
              markActiveTile(tileX, tileY)

              # Agent warmth effect with overflow protection
              safeTintAdd(env.tintMods[tileX][tileY].r, int((tribeColor.r - 0.7) * 63 * falloff.float32))
              safeTintAdd(env.tintMods[tileX][tileY].g, int((tribeColor.g - 0.65) * 63 * falloff.float32))
              safeTintAdd(env.tintMods[tileX][tileY].b, int((tribeColor.b - 0.6) * 63 * falloff.float32))
        
    of Altar:
      # Reduce altar tint effect by 10x (minimal warm glow)
      # No HashSet tracking needed
      markActiveTile(baseX, baseY)
      safeTintAdd(env.tintMods[baseX][baseY].r, 5)
      safeTintAdd(env.tintMods[baseX][baseY].g, 5)
      safeTintAdd(env.tintMods[baseX][baseY].b, 2)
    
    of PlantedLantern:
      # Lanterns spread team colors in 5x5 area (similar to clippies but warm colors)
      if thing.lanternHealthy and thing.teamId >= 0 and thing.teamId < teamColors.len:
        let teamColor = teamColors[thing.teamId]
        
        for dx in -2 .. 2:
          for dy in -2 .. 2:
            let tileX = baseX + dx
            let tileY = baseY + dy
            if tileX >= 0 and tileX < MapWidth and tileY >= 0 and tileY < MapHeight:
              # Distance-based falloff for more organic look
              let distance = abs(dx) + abs(dy)  # Manhattan distance
              let falloff = max(1, 5 - distance)  # Stronger at center, weaker at edges (5x5 grid)
              markActiveTile(tileX, tileY)
              
              # Lantern warm effect with overflow protection
              safeTintAdd(env.tintMods[tileX][tileY].r, int((teamColor.r - 0.7) * 50 * falloff.float32))
              safeTintAdd(env.tintMods[tileX][tileY].g, int((teamColor.g - 0.65) * 50 * falloff.float32))
              safeTintAdd(env.tintMods[tileX][tileY].b, int((teamColor.b - 0.6) * 50 * falloff.float32))
    
    else:
      discard

proc applyTintModifications(env: Environment) =
  ## Apply tint modifications to entity positions and their surrounding areas
  
  # Apply modifications only to tiles touched this frame
  for pos in env.activeTiles.positions:
    let tileX = pos.x.int
    let tileY = pos.y.int
    if tileX < 0 or tileX >= MapWidth or tileY < 0 or tileY >= MapHeight:
      continue

    # Skip if tint modifications are below minimum threshold
    let tint = env.tintMods[tileX][tileY]
    if abs(tint.r) < MinTintEpsilon and abs(tint.g) < MinTintEpsilon and abs(tint.b) < MinTintEpsilon:
      continue
    
    # Skip tinting on water tiles (rivers should remain clean)
    if env.terrain[tileX][tileY] == Water:
      continue
  
    # Get current color as integers (scaled by 1000 for precision)
    var r = int(env.tileColors[tileX][tileY].r * 1000)
    var g = int(env.tileColors[tileX][tileY].g * 1000)
    var b = int(env.tileColors[tileX][tileY].b * 1000)
    
    # Apply unified tint modifications
    r += tint.r div 10  # 10% of the modification
    g += tint.g div 10
    b += tint.b div 10
    
    # Convert back to float with clamping
    env.tileColors[tileX][tileY].r = min(max(r.float32 / 1000.0, 0.3), 1.2)
    env.tileColors[tileX][tileY].g = min(max(g.float32 / 1000.0, 0.3), 1.2)
    env.tileColors[tileX][tileY].b = min(max(b.float32 / 1000.0, 0.3), 1.2)
  
  # Apply global decay to ALL tiles (but infrequently for performance)
  if env.currentStep mod 30 == 0 and env.currentStep > 0:
    let decay = 0.98'f32  # 2% decay every 30 steps
    
    for x in 0 ..< MapWidth:
      for y in 0 ..< MapHeight:
        # Get the base color for this tile (could be team color for houses)
        let baseR = env.baseTileColors[x][y].r
        let baseG = env.baseTileColors[x][y].g
        let baseB = env.baseTileColors[x][y].b
        
        # Only decay if color differs from base (avoid floating point errors)
        # Lowered threshold to allow subtle creep effects to be balanced by decay
        if abs(env.tileColors[x][y].r - baseR) > 0.001 or 
           abs(env.tileColors[x][y].g - baseG) > 0.001 or 
           abs(env.tileColors[x][y].b - baseB) > 0.001:
          env.tileColors[x][y].r = env.tileColors[x][y].r * decay + baseR * (1.0 - decay)
          env.tileColors[x][y].g = env.tileColors[x][y].g * decay + baseG * (1.0 - decay)
          env.tileColors[x][y].b = env.tileColors[x][y].b * decay + baseB * (1.0 - decay)
        
        # Also decay intensity back to base intensity
        let baseIntensity = env.baseTileColors[x][y].intensity
        if abs(env.tileColors[x][y].intensity - baseIntensity) > 0.01:
          env.tileColors[x][y].intensity = env.tileColors[x][y].intensity * decay + baseIntensity * (1.0 - decay)

proc add(env: Environment, thing: Thing) =
  env.things.add(thing)
  if thing.kind == Agent:
    env.agents.add(thing)
    env.stats.add(Stats())
  env.grid[thing.pos.x][thing.pos.y] = thing

proc plantAction(env: Environment, id: int, agent: Thing, argument: int) =
  ## Plant lantern at agent's current position - argument specifies direction (0=N, 1=S, 2=W, 3=E, 4=NW, 5=NE, 6=SW, 7=SE)
  if argument > 7:
    inc env.stats[id].actionInvalid
    return
  
  # Check if agent has a lantern
  if agent.inventoryLantern <= 0:
    inc env.stats[id].actionInvalid
    return
  
  # Calculate target position based on orientation argument
  let plantOrientation = Orientation(argument)
  let delta = getOrientationDelta(plantOrientation)
  var targetPos = agent.pos
  targetPos.x += int32(delta.x)
  targetPos.y += int32(delta.y)
  
  # Check bounds
  if targetPos.x < 0 or targetPos.x >= MapWidth or targetPos.y < 0 or targetPos.y >= MapHeight:
    inc env.stats[id].actionInvalid
    return
  
  # Check if position is empty and not water
  if not env.isEmpty(targetPos) or env.terrain[targetPos.x][targetPos.y] == Water:
    inc env.stats[id].actionInvalid
    return
  
  # Calculate team ID directly from the planting agent's ID
  var teamId = agent.agentId div 5
  
  # Plant the lantern
  let lantern = Thing(
    kind: PlantedLantern,
    pos: targetPos,
    teamId: teamId,
    lanternHealthy: true
  )
  
  env.add(lantern)
  
  # Consume the lantern from agent's inventory
  agent.inventoryLantern = 0
  
  # Give reward for planting
  agent.reward += env.config.clothReward * 0.5  # Half reward for planting
  
  inc env.stats[id].actionPlant

proc init(env: Environment) =
  # Use current time for random seed to get different maps each time
  let seed = int(nowSeconds() * 1000)
  var r = initRand(seed)
  
  # Initialize tile colors to base terrain colors (neutral gray-brown)
  for x in 0 ..< MapWidth:
    for y in 0 ..< MapHeight:
      env.tileColors[x][y] = TileColor(r: 0.7, g: 0.65, b: 0.6, intensity: 1.0)
      env.baseTileColors[x][y] = TileColor(r: 0.7, g: 0.65, b: 0.6, intensity: 1.0)
  
  # Initialize active tiles tracking
  env.activeTiles.positions.setLen(0)
  env.activeTiles.flags = default(array[MapWidth, array[MapHeight, bool]])
  
  # Initialize terrain with all features
  initTerrain(env.terrain, MapWidth, MapHeight, MapBorder, seed)

  if MapBorder > 0:
    for x in 0 ..< MapWidth:
      for j in 0 ..< MapBorder:
        env.add(Thing(kind: Wall, pos: ivec2(x, j)))
        env.add(Thing(kind: Wall, pos: ivec2(x, MapHeight - j - 1)))
    for y in 0 ..< MapHeight:
      for j in 0 ..< MapBorder:
        env.add(Thing(kind: Wall, pos: ivec2(j, y)))
        env.add(Thing(kind: Wall, pos: ivec2(MapWidth - j - 1, y)))

  # Agents will now spawn with their villages/houses below
  # Clear and prepare village colors arrays
  agentVillageColors.setLen(MapRoomObjectsAgents)  # Allocate space for all agents
  teamColors.setLen(0)  # Clear team colors
  altarColors.clear()  # Clear altar colors from previous game
  # Spawn houses with their altars, walls, and associated agents (tribes)
  let numHouses = MapRoomObjectsHouses
  var totalAgentsSpawned = 0
  for i in 0 ..< numHouses:
    let houseStruct = createHouse()
    var placed = false
    var placementPosition: IVec2

    # Simple random placement with collision avoidance
    for attempt in 0 ..< 200:
      let candidatePos = r.randomEmptyPos(env)
      # Check if position has enough space for the 5x5 house
      var canPlace = true
      for dy in 0 ..< houseStruct.height:
        for dx in 0 ..< houseStruct.width:
          let checkX = candidatePos.x + dx
          let checkY = candidatePos.y + dy
          if checkX >= MapWidth or checkY >= MapHeight or
             not env.isEmpty(ivec2(checkX, checkY)) or
             env.terrain[checkX][checkY] == Water:
            canPlace = false
            break
        if not canPlace: break

      if canPlace:
        placementPosition = candidatePos
        placed = true
        break

    if placed:
      let elements = getStructureElements(houseStruct, placementPosition)
      
      # Clear terrain within the house area to create a clearing
      for dy in 0 ..< houseStruct.height:
        for dx in 0 ..< houseStruct.width:
          let clearX = placementPosition.x + dx
          let clearY = placementPosition.y + dy
          if clearX >= 0 and clearX < MapWidth and clearY >= 0 and clearY < MapHeight:
            # Clear any terrain features (wheat, trees) but keep water
            if env.terrain[clearX][clearY] != Water:
              env.terrain[clearX][clearY] = Empty
      
      # Generate a distinct warm color for this village (avoid cool/blue hues)
      let paletteIndex = i mod WarmVillagePalette.len
      let villageColor = WarmVillagePalette[paletteIndex]

      # If we loop past the palette (more than 12 houses), nudge the hue slightly
      # so repeats still look distinct while staying on the warm side.
      var finalVillageColor = villageColor
      if i >= WarmVillagePalette.len:
        let tweak = min(0.15, float(i div WarmVillagePalette.len) * 0.05)
        finalVillageColor = color(
          clamp(villageColor.r + tweak, 0.0, 1.0),
          clamp(villageColor.g + tweak * 0.3, 0.0, 1.0),
          clamp(villageColor.b, 0.0, 1.0),
          1.0
        )

      teamColors.add(finalVillageColor)

      # Spawn agents around this house
      let agentsForThisHouse = min(MapAgentsPerHouse, MapRoomObjectsAgents - totalAgentsSpawned)
      
      # Add the altar with initial hearts and house bounds
      env.add(Thing(
        kind: Altar,
        pos: elements.center,
        hearts: MapObjectAltarInitialHearts  # Altar starts with default hearts
      ))
      altarColors[elements.center] = finalVillageColor  # Associate altar position with village color
      
      # Initialize base colors for house tiles to team color
      for dx in 0 ..< houseStruct.width:
        for dy in 0 ..< houseStruct.height:
          let tileX = placementPosition.x + dx
          let tileY = placementPosition.y + dy
          if tileX >= 0 and tileX < MapWidth and tileY >= 0 and tileY < MapHeight:
            env.baseTileColors[tileX][tileY] = TileColor(
              r: finalVillageColor.r,
              g: finalVillageColor.g,
              b: finalVillageColor.b,
              intensity: 1.0
            )
            env.tileColors[tileX][tileY] = env.baseTileColors[tileX][tileY]
      
      # Add the walls
      for wallPos in elements.walls:
        env.add(Thing(
          kind: Wall,
          pos: wallPos,
        ))
      
      # Add the corner buildings from the house layout
      # Parse the house structure to find corner buildings
      for y in 0 ..< houseStruct.height:
        for x in 0 ..< houseStruct.width:
          if y < houseStruct.layout.len and x < houseStruct.layout[y].len:
            let worldPos = placementPosition + ivec2(x.int32, y.int32)
            case houseStruct.layout[y][x]:
            of 'A':  # Armory at top-left
              env.add(Thing(
                kind: Armory,
                pos: worldPos,
              ))
            of 'F':  # Forge at top-right  
              env.add(Thing(
                kind: Forge,
                pos: worldPos,
              ))
            of 'C':  # Clay Oven at bottom-left
              env.add(Thing(
                kind: ClayOven,
                pos: worldPos,
              ))
            of 'W':  # Weaving Loom at bottom-right
              env.add(Thing(
                kind: WeavingLoom,
                pos: worldPos,
              ))
            else:
              discard
      if agentsForThisHouse > 0:
        # Get nearby positions around the altar
        let nearbyPositions = env.findEmptyPositionsAround(elements.center, 3)
        
        for j in 0 ..< agentsForThisHouse:
          var agentPos: IVec2
          if j < nearbyPositions.len:
            # Use nearby positions
            agentPos = nearbyPositions[j]
          else:
            # Fallback to random
            agentPos = r.randomEmptyPos(env)
          
          let agentId = totalAgentsSpawned
          
          # Store the village color for this agent
          agentVillageColors[agentId] = finalVillageColor
          
          # Create the agent
          env.add(Thing(
            kind: Agent,
            agentId: agentId,
            pos: agentPos,
            orientation: Orientation(randIntInclusive(r, 0, 3)),
            homeAltar: elements.center,  # Link agent to their home altar
            inventoryOre: 0,
            inventoryBattery: 0,
            inventoryWater: 0,
            inventoryWheat: 0,
            inventoryWood: 0,
            inventorySpear: 0,
            inventoryLantern: 0,
            inventoryArmor: 0,
            frozen: 0,
          ))
          
          totalAgentsSpawned += 1
          if totalAgentsSpawned >= MapRoomObjectsAgents:
            break
      
      # Note: Entrances are left empty (no walls placed there)
  
  # Now place additional random walls after villages to avoid blocking corner placement
  for i in 0 ..< MapRoomObjectsWalls:
    let pos = r.randomEmptyPos(env)
    env.add(Thing(kind: Wall, pos: pos))
  
  # If there are still agents to spawn (e.g., if not enough houses), spawn them randomly
  # They will get a neutral color
  let neutralColor = color(0.5, 0.5, 0.5, 1.0)  # Gray for unaffiliated agents
  while totalAgentsSpawned < MapRoomObjectsAgents:
    let agentPos = r.randomEmptyPos(env)
    let agentId = totalAgentsSpawned
    
    # Store neutral color for agents without a village
    agentVillageColors[agentId] = neutralColor
    
    env.add(Thing(
      kind: Agent,
      agentId: agentId,
      pos: agentPos,
      orientation: Orientation(randIntInclusive(r, 0, 3)),
      homeAltar: ivec2(-1, -1),  # No home altar for unaffiliated agents
      inventoryOre: 0,
      inventoryBattery: 0,
      inventoryWater: 0,
      inventoryWheat: 0,
      inventoryWood: 0,
      inventorySpear: 0,
      inventoryLantern: 0,
      inventoryArmor: 0,
      frozen: 0,
    ))
    
    totalAgentsSpawned += 1

  # Random spawner placement with minimum distance from villages and other spawners
  # Gather altar positions for distance checks
  var altarPositionsNow: seq[IVec2] = @[]
  var spawnerPositions: seq[IVec2] = @[]
  for thing in env.things:
    if thing.kind == Altar:
      altarPositionsNow.add(thing.pos)

  let numSpawners = numHouses
  let minDist = 20  # tiles; simple guard so spawner isn't extremely close to a village
  let minDist2 = minDist * minDist

  for i in 0 ..< numSpawners:
    let spawnerStruct = createSpawner()
    var placed = false
    var targetPos: IVec2
    
    for attempt in 0 ..< 200:
      targetPos = r.randomEmptyPos(env)
      # Keep within borders allowing spawner bounds
      if targetPos.x < MapBorder + spawnerStruct.width div 2 or
         targetPos.x >= MapWidth - MapBorder - spawnerStruct.width div 2 or
         targetPos.y < MapBorder + spawnerStruct.height div 2 or
         targetPos.y >= MapHeight - MapBorder - spawnerStruct.height div 2:
        continue
      
      # Check simple area clear (3x3)
      var areaValid = true
      for dx in -(spawnerStruct.width div 2) .. (spawnerStruct.width div 2):
        for dy in -(spawnerStruct.height div 2) .. (spawnerStruct.height div 2):
          let checkPos = targetPos + ivec2(dx, dy)
          if checkPos.x < 0 or checkPos.x >= MapWidth or checkPos.y < 0 or checkPos.y >= MapHeight:
            areaValid = false
            break
          if not env.isEmpty(checkPos) or env.terrain[checkPos.x][checkPos.y] == Water:
            areaValid = false
            break
        if not areaValid:
          break

      if not areaValid:
        continue

      # Enforce min distance from any altar and other spawners
      var okDistance = true
      # Check distance from villages (altars)
      for ap in altarPositionsNow:
        let dx = int(targetPos.x) - int(ap.x)
        let dy = int(targetPos.y) - int(ap.y)
        if dx*dx + dy*dy < minDist2:
          okDistance = false
          break
      # Check distance from other spawners
      for sp in spawnerPositions:
        let dx = int(targetPos.x) - int(sp.x)
        let dy = int(targetPos.y) - int(sp.y)
        if dx*dx + dy*dy < minDist2:
          okDistance = false
          break
      if not okDistance:
        continue

      # Clear terrain and place spawner
      for dx in -(spawnerStruct.width div 2) .. (spawnerStruct.width div 2):
        for dy in -(spawnerStruct.height div 2) .. (spawnerStruct.height div 2):
          let clearPos = targetPos + ivec2(dx, dy)
          if clearPos.x >= 0 and clearPos.x < MapWidth and clearPos.y >= 0 and clearPos.y < MapHeight:
            if env.terrain[clearPos.x][clearPos.y] != Water:
              env.terrain[clearPos.x][clearPos.y] = Empty

      env.add(Thing(
        kind: Spawner,
        pos: targetPos,
        cooldown: 0,
        homeSpawner: targetPos
      ))

      # Add this spawner position for future collision checks
      spawnerPositions.add(targetPos)

      let nearbyPositions = env.findEmptyPositionsAround(targetPos, 1)
      if nearbyPositions.len > 0:
        let spawnCount = min(3, nearbyPositions.len)
        for i in 0 ..< spawnCount:
          env.add(createTumor(nearbyPositions[i], targetPos, r))
      placed = true
      break

    # If we fail to satisfy distance after attempts, place anywhere random
    if not placed:
      targetPos = r.randomEmptyPos(env)
      env.add(Thing(
        kind: Spawner,
        pos: targetPos,
        cooldown: 0,
        homeSpawner: targetPos
      ))
      let nearbyPositions = env.findEmptyPositionsAround(targetPos, 1)
      if nearbyPositions.len > 0:
        let spawnCount = min(3, nearbyPositions.len)
        for i in 0 ..< spawnCount:
          env.add(createTumor(nearbyPositions[i], targetPos, r))

  for i in 0 ..< MapRoomObjectsConverters:
    let pos = r.randomEmptyPos(env)
    env.add(Thing(
      kind: Converter,
      pos: pos,
    ))

  for i in 0 ..< MapRoomObjectsMines:
    let pos = r.randomEmptyPos(env)
    env.add(Thing(
      kind: Mine,
      pos: pos,
      resources: MapObjectMineInitialResources,
    ))

  # Initialize altar locations for all spawners
  var altarPositions: seq[IVec2] = @[]
  for thing in env.things:
    if thing.kind == Altar:
      altarPositions.add(thing.pos)

  # Initialize observations only when first needed (lazy approach)
  # Individual action updates will populate observations as needed


proc defaultEnvironmentConfig*(): EnvironmentConfig =
  ## Create default environment configuration
  EnvironmentConfig(
    # Core game parameters
    maxSteps: 1000,
    
    # Resource configuration
    orePerBattery: 1,
    batteriesPerHeart: 1,
    
    # Combat configuration
    enableCombat: true,
    tumorSpawnRate: 0.1,
    tumorDamage: 1,
    
    # Reward configuration (only arena_basic_easy_shaped rewards active)
    heartReward: 1.0,      # Arena: heart reward
    oreReward: 0.1,        # Arena: ore mining reward  
    batteryReward: 0.8,    # Arena: battery crafting reward
    woodReward: 0.0,       # Disabled - not in arena
    waterReward: 0.0,      # Disabled - not in arena
    wheatReward: 0.0,      # Disabled - not in arena
    spearReward: 0.0,      # Disabled - not in arena
    armorReward: 0.0,      # Disabled - not in arena
    foodReward: 0.0,       # Disabled - not in arena
    clothReward: 0.0,      # Disabled - not in arena
    tumorKillReward: 0.0, # Disabled - not in arena
    survivalPenalty: -0.01,
    deathPenalty: -5.0
  )

proc newEnvironment*(): Environment =
  ## Create a new environment with default configuration
  result = Environment(config: defaultEnvironmentConfig())
  result.init()

proc newEnvironment*(config: EnvironmentConfig): Environment =
  ## Create a new environment with custom configuration
  result = Environment(config: config)
  result.init()

# Initialize the global environment
env = newEnvironment()


proc step*(env: Environment, actions: ptr array[MapAgents, array[2, uint8]]) =
  ## Step the environment
  inc env.currentStep
  # Single RNG for entire step - more efficient than multiple initRand calls
  var stepRng = initRand(env.currentStep)

  for id, action in actions[]:
    let agent = env.agents[id]
    if agent.frozen > 0:
      continue

    case action[0]:
    of 0: env.noopAction(id, agent)
    of 1: env.moveAction(id, agent, action[1].int)
    of 2: env.attackAction(id, agent, action[1].int)
    of 3: env.useAction(id, agent, action[1].int)  # Use terrain/buildings
    of 4: env.swapAction(id, agent, action[1].int)
    of 5: env.putAction(id, agent, action[1].int)  # Give to teammate
    of 6: env.plantAction(id, agent, action[1].int)  # Plant lantern
    else: inc env.stats[id].actionInvalid

  # Combined single-pass object updates and tumor collection
  var newTumorsToSpawn: seq[Thing] = @[]
  var tumorsToProcess: seq[Thing] = @[]

  for thing in env.things:
    if thing.kind == Altar:
      if thing.cooldown > 0:
        thing.cooldown -= 1
        env.updateObservations(AltarReadyLayer, thing.pos, thing.cooldown)
      # Combine altar heart reward calculation here
      if env.currentStep >= env.config.maxSteps:  # Only at episode end
        let altarHearts = thing.hearts.float32
        for agent in env.agents:
          if agent.homeAltar == thing.pos:
            agent.reward += altarHearts / MapAgentsPerHouseFloat
    elif thing.kind == Converter:
      if thing.cooldown > 0:
        thing.cooldown -= 1
        env.updateObservations(ConverterReadyLayer, thing.pos, thing.cooldown)
    elif thing.kind == Mine:
      if thing.cooldown > 0:
        thing.cooldown -= 1
        env.updateObservations(MineReadyLayer, thing.pos, thing.cooldown)
    elif thing.kind in {Forge, Armory, ClayOven, WeavingLoom}:
      # All production buildings have simple cooldown
      if thing.cooldown > 0:
        thing.cooldown -= 1
    elif thing.kind == Spawner:
      if thing.cooldown > 0:
        thing.cooldown -= 1
      else:
        # Spawner is ready to spawn a Tumor
        # Fast grid-based nearby Tumor count (5-tile radius)
        var nearbyTumorCount = 0
        for dx in -5..5:
          for dy in -5..5:
            let checkPos = thing.pos + ivec2(dx, dy)
            if isValidPos(checkPos):
              let other = env.getThing(checkPos)
              if not isNil(other) and other.kind == Tumor and not other.hasClaimedTerritory:
                inc nearbyTumorCount
        
        # Spawn a new Tumor with reasonable limits to prevent unbounded growth
        let maxTumorsPerSpawner = 3  # Keep only a few active tumors near the spawner
        if nearbyTumorCount < maxTumorsPerSpawner:
          # Find first empty position (no allocation)
          let spawnPos = env.findFirstEmptyPositionAround(thing.pos, 2)
          if spawnPos.x >= 0:
            
            let newTumor = createTumor(spawnPos, thing.pos, stepRng)
            # Don't add immediately - collect for later
            newTumorsToSpawn.add(newTumor)
            
            # Reset spawner cooldown based on spawn rate
            # Convert spawn rate (0.0-1.0) to cooldown steps (higher rate = lower cooldown)
            let cooldown = if env.config.tumorSpawnRate > 0.0:
              max(1, int(20.0 / env.config.tumorSpawnRate))  # Base 20 steps, scaled by rate
            else:
              1000  # Very long cooldown if spawn disabled
            thing.cooldown = cooldown
    elif thing.kind == Agent:
      if thing.frozen > 0:
        thing.frozen -= 1
    elif thing.kind == Tumor:
      # Only collect mobile clippies for processing (planted ones are static)
      if not thing.hasClaimedTerritory:
        tumorsToProcess.add(thing)

  # ============== TUMOR PROCESSING ==============
  var newTumorBranches: seq[Thing] = @[]

  for tumor in tumorsToProcess:
    tumor.turnsAlive += 1
    if tumor.turnsAlive < TumorBranchMinAge:
      continue

    if randFloat(stepRng) >= TumorBranchChance:
      continue

    let branchPos = findTumorBranchTarget(tumor, env, stepRng)
    if branchPos.x < 0:
      continue

    let newTumor = createTumor(branchPos, tumor.homeSpawner, stepRng)

    # Face both clippies toward the new branch direction for clarity
    let dx = branchPos.x - tumor.pos.x
    let dy = branchPos.y - tumor.pos.y
    var branchOrientation: Orientation
    if abs(dx) >= abs(dy):
      branchOrientation = (if dx >= 0: Orientation.E else: Orientation.W)
    else:
      branchOrientation = (if dy >= 0: Orientation.S else: Orientation.N)

    newTumor.orientation = branchOrientation
    tumor.orientation = branchOrientation

    # Queue the new tumor for insertion and mark parent as inert
    newTumorBranches.add(newTumor)
    tumor.hasClaimedTerritory = true
    tumor.turnsAlive = 0

  # Add newly spawned tumors from spawners and branching this step
  for newTumor in newTumorsToSpawn:
    env.add(newTumor)
  for newTumor in newTumorBranches:
    env.add(newTumor)

  # Resolve agent contact: agents adjacent to tumors risk lethal creep
  var tumorsToRemove: seq[Thing] = @[]

  for thing in env.things:
    if thing.kind != Tumor:
      continue

    let tumor = thing
    let adjacentPositions = [
      tumor.pos + ivec2(0, -1),
      tumor.pos + ivec2(1, 0),
      tumor.pos + ivec2(0, 1),
      tumor.pos + ivec2(-1, 0)
    ]

    for adjPos in adjacentPositions:
      if not isValidPos(adjPos):
        continue

      let occupant = env.getThing(adjPos)
      if isNil(occupant) or occupant.kind != Agent:
        continue

      if randFloat(stepRng) < TumorAdjacencyDeathChance:
        if tumor notin tumorsToRemove:
          tumorsToRemove.add(tumor)
          env.grid[tumor.pos.x][tumor.pos.y] = nil
          env.updateObservations(AgentLayer, tumor.pos, 0)
          env.updateObservations(AgentOrientationLayer, tumor.pos, 0)

        env.killAgent(occupant)
        break

  # Remove tumors cleared by lethal contact this step
  if tumorsToRemove.len > 0:
    for i in countdown(env.things.len - 1, 0):
      if env.things[i] in tumorsToRemove:
        env.things.del(i)

  # Respawn dead agents at their altars
  for agentId in 0 ..< MapAgents:
    let agent = env.agents[agentId]
    
    # Check if agent is dead and has a home altar
    if env.terminated[agentId] == 1.0 and agent.homeAltar.x >= 0:
      # Find the altar
      var altar: Thing = nil
      for thing in env.things:
        if thing.kind == Altar and thing.pos == agent.homeAltar:
          altar = thing
          break
      
      # Respawn if altar exists and has hearts
      if not isNil(altar) and altar.hearts > 0:
        # Deduct a heart from the altar
        altar.hearts -= MapObjectAltarRespawnCost
        env.updateObservations(AltarHeartsLayer, altar.pos, altar.hearts)
        
        # Find first empty position around altar (no allocation)
        let respawnPos = env.findFirstEmptyPositionAround(altar.pos, 2)
        if respawnPos.x >= 0:
          # Respawn the agent
          agent.pos = respawnPos
          agent.inventoryOre = 0
          agent.inventoryBattery = 0
          agent.inventoryWater = 0
          agent.inventoryWheat = 0
          agent.inventoryWood = 0
          agent.inventorySpear = 0
          agent.frozen = 0
          env.terminated[agentId] = 0.0
          
          # Update grid
          env.grid[agent.pos.x][agent.pos.y] = agent
          
          # Update observations
          # REMOVED: expensive per-agent full grid rebuild
  
  # Apply per-step survival penalty to all living agents
  if env.config.survivalPenalty != 0.0:
    for agent in env.agents:
      if agent.frozen == 0:  # Only alive agents
        agent.reward += env.config.survivalPenalty
  
  # Update heatmap using batch tint modification system
  # This is much more efficient than updating during each entity move
  env.updateTintModifications()  # Collect all entity contributions
  env.applyTintModifications()   # Apply them to the main color array in one pass

  
  # Check if episode should end
  if env.currentStep >= env.config.maxSteps:
    # Team altar rewards already applied in main loop above
    # Mark all living agents as truncated (episode ended due to time limit)
    for i in 0..<MapAgents:
      if env.terminated[i] == 0.0:
        env.truncated[i] = 1.0
    env.shouldReset = true
  
  # Check if all agents are terminated/truncated
  var allDone = true
  for i in 0..<MapAgents:
    if env.terminated[i] == 0.0 and env.truncated[i] == 0.0:
      allDone = false
      break
  if allDone:
    # Team altar rewards already applied in main loop if needed
    env.shouldReset = true

proc reset*(env: Environment) =
  env.currentStep = 0
  env.shouldReset = false
  env.terminated.clear()
  env.truncated.clear()
  env.things.setLen(0)
  env.agents.setLen(0)
  env.stats.setLen(0)
  env.grid.clear()
  env.observations.clear()
  # Clear the massive tintMods array to prevent accumulation
  env.tintMods.clear()
  env.activeTiles.positions.setLen(0)
  env.activeTiles.flags = default(array[MapWidth, array[MapHeight, bool]])
  # Clear global colors that could accumulate
  agentVillageColors.setLen(0)
  teamColors.setLen(0)
  altarColors.clear()
  # Clear UI selection to prevent stale references
  selection = nil
  env.init()  # init() handles terrain, activeTiles, and tile colors

# ============== COLOR MANAGEMENT ==============

proc generateEntityColor*(entityType: string, id: int, fallbackColor: Color = color(0.5, 0.5, 0.5, 1.0)): Color =
  ## Unified color generation for all entity types
  ## Uses golden angle for optimal color distribution
  case entityType:
  of "agent":
    if id >= 0 and id < agentVillageColors.len:
      return agentVillageColors[id]
    # Fallback using mathematical constants for variety
    let f = id.float32
    return color(
      f * PI mod 1.0,
      f * math.E mod 1.0,
      f * sqrt(2.0) mod 1.0,
      1.0
    )
  of "village":
    # Warm colors for villages using golden angle
    let hue = (id.float32 * 137.5) mod 360.0 / 360.0
    let saturation = 0.7 + (id.float32 * 0.13) mod 0.3
    let lightness = 0.5 + (id.float32 * 0.17) mod 0.2
    return color(hue, saturation, lightness, 1.0)
  else:
    return fallbackColor

proc getAltarColor*(pos: IVec2): Color =
  ## Get altar color by position, with white fallback
  altarColors.getOrDefault(pos, color(1.0, 1.0, 1.0, 1.0))
