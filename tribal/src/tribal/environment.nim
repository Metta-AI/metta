import std/[random, tables, times, math, os], vmath, chroma
import terrain, objects, common
export terrain, objects, common


const
  # Map layout constants
  MapLayoutRoomsX* = 1
  MapLayoutRoomsY* = 1
  MapBorder* = 4
  # Increase map size and target ~16:9 aspect by using 192x108 rooms
  MapRoomWidth* = 192
  MapRoomHeight* = 108
  MapRoomBorder* = 0
  
  # Reward constants
  RewardGetWater* = 0.001      # Collecting water from tiles
  RewardGetWheat* = 0.001      # Harvesting wheat 
  RewardGetWood* = 0.002       # Chopping wood (slightly higher as it's needed for spears)
  RewardMineOre* = 0.003       # Mining ore (first step in battery chain)
  RewardConvertOreToBattery* = 0.01   # Using converter to make batteries
  RewardCraftSpear* = 0.01            # Using forge to craft spear
  RewardCraftArmor* = 0.015           # Using armory to craft armor  
  RewardCraftFood* = 0.012            # Using clay oven to craft food
  RewardCraftCloth* = 0.012           # Using weaving loom to craft cloth
  RewardDestroyClippy* = 0.1          # Destroying a clippy with spear
  MapRoomObjectsAgents* = 15  # Total agents to spawn (will be distributed across villages)
  MapRoomObjectsHouses* = 3  # Number of villages/houses to spawn
  MapAgentsPerHouse* = 5  # Agents to spawn per house/village
  MapRoomObjectsConverters* = 10  # Converters to process ore into batteries
  MapRoomObjectsMines* = 20  # Mines to extract ore (2x generators)
  MapRoomObjectsWalls* = 30

  MapObjectAgentMaxInventory* = 5
  MapObjectAgentFreezeDuration* = 10

  MapObjectAltarInitialHearts* = 5
  MapObjectAltarCooldown* = 10
  MapObjectAltarRespawnCost* = 1
  MapObjectConverterCooldown* = 0

  MapObjectMineCooldown* = 5
  MapObjectMineInitialResources* = 30
  MapObjectMineUseCost* = 0
  SpawnerCooldown* = 13  # Steps between Clippy spawns (1/3 of original 40)
  MinTintEpsilon* = 5   # Minimum tint threshold for visual effects
  ObservationLayers* = 21
  ObservationWidth* = 11
  ObservationHeight* = 11
  # Computed
  MapAgents* = MapRoomObjectsAgents * MapLayoutRoomsX * MapLayoutRoomsY
  MapWidth* = MapLayoutRoomsX * (MapRoomWidth + MapRoomBorder) + MapBorder
  MapHeight* = MapLayoutRoomsY * (MapRoomHeight + MapRoomBorder) + MapBorder


# Global village color management
var agentVillageColors*: seq[Color] = @[]
var teamColors*: seq[Color] = @[]
var altarColors*: Table[IVec2, Color] = initTable[IVec2, Color]()

type
  ObservationName* = enum
    AgentLayer = 0
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
    Clippy
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
    # Clippy:
    homeSpawner*: IVec2     # Position of clippy's home spawner
    hasClaimedTerritory*: bool  # Whether this clippy has claimed territory and is stationary
    turnsAlive*: int            # Number of turns this clippy has been alive
    
    # PlantedLantern:
    teamId*: int               # Which team this lantern belongs to (for color spreading)
    lanternHealthy*: bool      # Whether lantern is active (not destroyed by clippy)
    
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
    intensity*: int16       # Intensity delta (scaled by 1000)
  
  # Track active tiles for sparse processing
  ActiveTiles* = object
    positions*: seq[IVec2]  # List of tiles with entities
    count*: int             # Number of active tiles

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
    clippySpawnRate*: float
    clippyDamage*: int
    
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
    clippyKillReward*: float
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
  ## Enhanced check if a building is frozen due to clippy creep zone effect
  if pos.x < 0 or pos.x >= MapWidth or pos.y < 0 or pos.y >= MapHeight:
    return false
  
  let color = env.tileColors[pos.x][pos.y]
  # Cool colors have more blue than red+green combined
  let basicCoolCheck = color.b > (color.r + color.g)
  
  # Additional saturation check: blue should be significantly high (close to max saturation)  
  # This indicates prolonged presence in a clippy creep zone
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
          of Clippy:
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

proc clear[N: int, T](s: ptr array[N, T]) =
  let p = cast[pointer](s[][0].addr)
  zeroMem(p, s[].len * sizeof(T))

proc updateObservations(env: Environment, agentId: int) =
  var obs = env.observations[agentId].addr
  obs.clear()

  let agent = env.agents[agentId]
  var
    gridOffset = agent.pos - ivec2(ObservationWidth div 2, ObservationHeight div 2)
    gridStart = gridOffset
    gridEnd = gridOffset + ivec2(ObservationWidth, ObservationHeight)
  if gridStart.x < 0:
    gridStart.x = 0
  if gridStart.y < 0:
    gridStart.y = 0
  if gridEnd.x > MapWidth: gridEnd.x = MapWidth
  if gridEnd.y > MapHeight: gridEnd.y = MapHeight

  for gy in gridStart.y ..< gridEnd.y:
    for gx in gridStart.x ..< gridEnd.x:
      let thing = env.grid[gx][gy]
      if thing == nil:
        continue
      let x = gx - gridOffset.x
      let y = gy - gridOffset.y

      case thing.kind
      of Agent:
        # Layer 0: AgentLayer
        obs[0][x][y] = 1
        # Layer 1: AgentOrientationLayer
        obs[1][x][y] = thing.orientation.uint8
        # Layer 2: AgentInventoryOreLayer
        obs[2][x][y] = thing.inventoryOre.uint8
        # Layer 3: AgentInventoryBatteryLayer
        obs[3][x][y] = thing.inventoryBattery.uint8
        # Layer 4: AgentInventoryWaterLayer
        obs[4][x][y] = thing.inventoryWater.uint8
        # Layer 5: AgentInventoryWheatLayer
        obs[5][x][y] = thing.inventoryWheat.uint8
        # Layer 6: AgentInventoryWoodLayer
        obs[6][x][y] = thing.inventoryWood.uint8
        # Layer 7: AgentInventorySpearLayer
        obs[7][x][y] = thing.inventorySpear.uint8
        # Layer 8: AgentInventoryLanternLayer
        obs[8][x][y] = thing.inventoryLantern.uint8
        # Layer 9: AgentInventoryArmorLayer
        obs[9][x][y] = thing.inventoryArmor.uint8
        # Layer 20: AgentInventoryBreadLayer
        obs[20][x][y] = thing.inventoryBread.uint8

      of Wall:
        # Layer 10: WallLayer (shifted due to lantern layer)
        obs[10][x][y] = 1

      of Mine:
        # Layer 11: MineLayer
        obs[11][x][y] = 1
        # Layer 12: MineResourceLayer
        obs[12][x][y] = thing.resources.uint8
        # Layer 13: MineReadyLayer
        obs[13][x][y] = (thing.cooldown == 0).uint8

      of Converter:
        # Layer 14: ConverterLayer
        obs[14][x][y] = 1
        # Layer 15: ConverterReadyLayer
        obs[15][x][y] = (thing.cooldown == 0).uint8

      of Altar:
        # Layer 16: AltarLayer
        obs[16][x][y] = 1
        # Layer 17: AltarHeartsLayer
        obs[17][x][y] = thing.hearts.uint8
        # Layer 18: AltarReadyLayer
        obs[18][x][y] = (thing.cooldown == 0).uint8
      
      of Spawner:
        # Spawner acts similar to altar for observations
        obs[16][x][y] = 1
        obs[17][x][y] = thing.hearts.uint8
        obs[18][x][y] = (thing.cooldown == 0).uint8
      
      of PlantedLantern:
        # PlantedLantern uses wall layer for now (visible as obstacle)
        obs[10][x][y] = 1
      
      of Clippy:
        # Clippy acts similar to agent for observations
        obs[0][x][y] = 1
        obs[1][x][y] = thing.orientation.uint8  # Clippy orientation
        obs[2][x][y] = 0  # Clippys don't carry ore
        obs[3][x][y] = 0  # Clippys don't carry batteries
        obs[4][x][y] = 0  # No water
        obs[5][x][y] = 0  # No wheat
        obs[6][x][y] = 0  # No wood
      
      of Armory, Forge, ClayOven, WeavingLoom:
        # Corner buildings act like walls for observations
        obs[10][x][y] = 1  # Use the wall layer for now

  # Add tint data to new observation layers
  for gy in gridStart.y ..< gridEnd.y:
    for gx in gridStart.x ..< gridEnd.x:
      let x = gx - gridOffset.x
      let y = gy - gridOffset.y
      
      # Layer 19: TintLayer - Unified tint intensity from all sources
      let tint = env.tintMods[gx][gy]
      let tintIntensity = abs(tint.r) + abs(tint.g) + abs(tint.b)
      obs[19][x][y] = min(255, tintIntensity div 4).uint8  # Scale down for uint8 range

proc updateObservations(
  env: Environment,
  layer: ObservationName,
  pos: IVec2,
  value: int
) =
  let layerId = ord(layer)
  for agentId in 0 ..< MapAgents:
    let x = pos.x - env.agents[agentId].pos.x + ObservationWidth div 2
    let y = pos.y - env.agents[agentId].pos.y + ObservationHeight div 2
    if x < 0 or x >= ObservationWidth or y < 0 or y >= ObservationHeight:
      continue
    env.observations[agentId][layerId][x][y] = value.uint8


proc getThing(env: Environment, pos: IVec2): Thing =
  if pos.x < 0 or pos.x >= MapWidth or pos.y < 0 or pos.y >= MapHeight:
    return nil
  return env.grid[pos.x][pos.y]


proc isEmpty*(env: Environment, pos: IVec2): bool =
  if pos.x < 0 or pos.x >= MapWidth or pos.y < 0 or pos.y >= MapHeight:
    return false
  return env.grid[pos.x][pos.y] == nil



proc createClippy(pos: IVec2, homeSpawner: IVec2, r: var Rand): Thing =
  ## Create a new Clippy for creep spread behavior
  Thing(
    kind: Clippy,
    pos: pos,
    orientation: Orientation(r.rand(0..3)),
    homeSpawner: homeSpawner,
    hasClaimedTerritory: false,  # Start mobile, will plant when far enough from others
    turnsAlive: 0                # New clippy hasn't lived any turns yet
  )






proc noopAction(env: Environment, id: int, agent: Thing) =
  inc env.stats[id].actionNoop

proc clearAgentObservations(env: Environment, pos: IVec2) =
  env.updateObservations(AgentLayer, pos, 0)
  env.updateObservations(AgentOrientationLayer, pos, 0)
  env.updateObservations(AgentInventoryOreLayer, pos, 0)
  env.updateObservations(AgentInventoryBatteryLayer, pos, 0)
  env.updateObservations(AgentInventoryWaterLayer, pos, 0)
  env.updateObservations(AgentInventoryWheatLayer, pos, 0)
  env.updateObservations(AgentInventoryWoodLayer, pos, 0)

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
    env.clearAgentObservations(agent.pos)
    agent.pos = newPos
    agent.orientation = newOrientation
    env.grid[agent.pos.x][agent.pos.y] = agent
    
    env.updateObservations(AgentLayer, agent.pos, 1)
    env.updateObservations(AgentOrientationLayer, agent.pos, agent.orientation.int)
    env.updateObservations(AgentInventoryOreLayer, agent.pos, agent.inventoryOre)
    env.updateObservations(AgentInventoryBatteryLayer, agent.pos, agent.inventoryBattery)
    env.updateObservations(AgentInventoryWaterLayer, agent.pos, agent.inventoryWater)
    env.updateObservations(AgentInventoryWheatLayer, agent.pos, agent.inventoryWheat)
    env.updateObservations(AgentInventoryWoodLayer, agent.pos, agent.inventoryWood)
    env.updateObservations(id)
    inc env.stats[id].actionMove
  else:
    inc env.stats[id].actionInvalid

proc attackAction(env: Environment, id: int, agent: Thing, argument: int) =
  ## Attack with a spear if agent has one
  ## argument: 0=N, 1=S, 2=W, 3=E, 4=NW, 5=NE, 6=SW, 7=SE (direction to attack)
  
  # Check if agent has a spear
  if agent.inventorySpear <= 0:
    inc env.stats[id].actionInvalid
    return
  
  # Validate orientation argument
  if argument < 0 or argument > 7:
    inc env.stats[id].actionInvalid
    return
  
  # Calculate attack positions (range of 2 tiles in given direction)
  let attackOrientation = Orientation(argument)
  let delta = getOrientationDelta(attackOrientation)
  var attackPositions: seq[IVec2] = @[]
  
  # Add positions at range 1 and 2 in the given direction
  attackPositions.add(agent.pos + ivec2(delta.x, delta.y))
  attackPositions.add(agent.pos + ivec2(delta.x * 2, delta.y * 2))
  
  # Check for Clippys at attack positions
  var hitClippy = false
  var clippyToRemove: Thing = nil
  
  for attackPos in attackPositions:
    # Check bounds
    if attackPos.x < 0 or attackPos.x >= MapWidth or 
       attackPos.y < 0 or attackPos.y >= MapHeight:
      continue
    
    # Check for Clippy at this position
    let target = env.getThing(attackPos)
    if not isNil(target) and target.kind == Clippy:
      clippyToRemove = target
      hitClippy = true
      break
  
  if hitClippy and not isNil(clippyToRemove):
    # Remove the Clippy
    env.grid[clippyToRemove.pos.x][clippyToRemove.pos.y] = nil
    let idx = env.things.find(clippyToRemove)
    if idx >= 0:
      env.things.del(idx)
    
    # Consume one use of the spear
    agent.inventorySpear -= 1
    env.updateObservations(AgentInventorySpearLayer, agent.pos, agent.inventorySpear)
    
    # Give reward for destroying Clippy
    agent.reward += env.config.clippyKillReward  # Moderate reward for defense
    
    inc env.stats[id].actionAttack
  else:
    # Attack missed or no valid target
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
      env.updateObservations(AgentInventoryWaterLayer, agent.pos, agent.inventoryWater)
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
      env.updateObservations(AgentInventoryWheatLayer, agent.pos, agent.inventoryWheat)
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
      env.updateObservations(AgentInventoryWoodLayer, agent.pos, agent.inventoryWood)
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
      env.updateObservations(AgentInventoryWoodLayer, agent.pos, agent.inventoryWood)
      env.updateObservations(AgentInventorySpearLayer, agent.pos, agent.inventorySpear)
      agent.reward += env.config.spearReward
      inc env.stats[id].actionUse
    else:
      inc env.stats[id].actionInvalid
  of WeavingLoom:
    if thing.cooldown == 0 and agent.inventoryWheat > 0 and agent.inventoryLantern == 0:
      agent.inventoryWheat -= 1
      agent.inventoryLantern = 1
      thing.cooldown = 15
      env.updateObservations(AgentInventoryWheatLayer, agent.pos, agent.inventoryWheat)
      env.updateObservations(AgentInventoryLanternLayer, agent.pos, agent.inventoryLantern)
      agent.reward += env.config.clothReward
      inc env.stats[id].actionUse
    else:
      inc env.stats[id].actionInvalid
  of Armory:
    if thing.cooldown == 0 and agent.inventoryWood > 0 and agent.inventoryArmor == 0:
      agent.inventoryWood -= 1
      agent.inventoryArmor = 5
      thing.cooldown = 20
      env.updateObservations(AgentInventoryWoodLayer, agent.pos, agent.inventoryWood)
      env.updateObservations(AgentInventoryArmorLayer, agent.pos, agent.inventoryArmor)
      agent.reward += env.config.armorReward
      inc env.stats[id].actionUse
    else:
      inc env.stats[id].actionInvalid
  of ClayOven:
    if thing.cooldown == 0 and agent.inventoryWheat > 0:
      agent.inventoryWheat -= 1
      agent.inventoryBread += 1
      thing.cooldown = 10
      env.updateObservations(AgentInventoryWheatLayer, agent.pos, agent.inventoryWheat)
      env.updateObservations(AgentInventoryBreadLayer, agent.pos, agent.inventoryBread)
      # No observation layer for bread; optional for UI later
      agent.reward += env.config.foodReward
      inc env.stats[id].actionUse
    else:
      inc env.stats[id].actionInvalid
  of Altar:
    if thing.cooldown == 0 and agent.inventoryBattery >= 1:
      agent.inventoryBattery -= 1
      thing.hearts += 1
      thing.cooldown = MapObjectAltarCooldown
      env.updateObservations(AgentInventoryBatteryLayer, agent.pos, agent.inventoryBattery)
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
    env.updateObservations(id)
    env.updateObservations(target.agentId)
  else:
    inc env.stats[id].actionInvalid



proc sameTeam(a, b: Thing): bool =
  ## Agents are teammates if they share the same house index (agentId div MapAgentsPerHouse)
  (a.agentId div MapAgentsPerHouse) == (b.agentId div MapAgentsPerHouse)

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
  if not sameTeam(agent, target):
    inc env.stats[id].actionInvalid
    return
  var transferred = false
  # Give armor if we have any and target has none
  if agent.inventoryArmor > 0 and target.inventoryArmor == 0:
    target.inventoryArmor = agent.inventoryArmor
    agent.inventoryArmor = 0
    env.updateObservations(AgentInventoryArmorLayer, target.pos, target.inventoryArmor)
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
    env.updateObservations(AgentInventoryArmorLayer, agent.pos, agent.inventoryArmor)
    env.updateObservations(AgentInventoryArmorLayer, target.pos, target.inventoryArmor)
    env.updateObservations(AgentInventoryBreadLayer, agent.pos, agent.inventoryBread)
    env.updateObservations(AgentInventoryBreadLayer, target.pos, target.inventoryBread)
  else:
    inc env.stats[id].actionInvalid
# ============== CLIPPY AI ==============




proc isValidEmptyPosition(env: Environment, pos: IVec2): bool =
  ## Check if a position is within map bounds, empty, and not water
  pos.x >= MapBorder and pos.x < MapWidth - MapBorder and
  pos.y >= MapBorder and pos.y < MapHeight - MapBorder and
  env.isEmpty(pos) and env.terrain[pos.x][pos.y] != Water

proc generateRandomMapPosition(r: var Rand): IVec2 =
  ## Generate a random position within map boundaries
  ivec2(r.rand(MapBorder ..< MapWidth - MapBorder), r.rand(MapBorder ..< MapHeight - MapBorder))

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
          
        # Must be 2+ tiles from any existing lantern
        var tooCloseToLantern = false
        for thing in env.things:
          if thing.kind == PlantedLantern:
            let distToLantern = manhattanDistance(candidate, thing.pos)
            if distToLantern < 2:
              tooCloseToLantern = true
              break
        
        if not tooCloseToLantern:
          return candidate  # Found a good spot!
  
  # No good spot found
  return ivec2(-1, -1)

# ============== CLIPPY CREEP SPREAD AI ==============

proc hasNearbyClippies(env: Environment, pos: IVec2, radius: int, excludeClippy: Thing = nil): bool =
  ## Check if there are any clippies within radius tiles (observation window)
  for dx in -radius .. radius:
    for dy in -radius .. radius:
      if dx == 0 and dy == 0:
        continue  # Skip center position
      
      let checkPos = pos + ivec2(dx, dy)
      if checkPos.x >= 0 and checkPos.x < MapWidth and checkPos.y >= 0 and checkPos.y < MapHeight:
        let thing = env.getThing(checkPos)
        if not isNil(thing) and thing.kind == Clippy and thing != excludeClippy:
          return true
  return false

proc getClippyMoveDirection(clippy: Thing, env: Environment, r: var Rand): IVec2 =
  ## Simple clippy behavior: stay put if planted, otherwise move randomly
  
  # If planted, don't move
  if clippy.hasClaimedTerritory:
    return ivec2(0, 0)
  
  # Move randomly to explore
  const dirs = [ivec2(0, -1), ivec2(0, 1), ivec2(-1, 0), ivec2(1, 0)]
  return dirs[r.rand(0..<4)]

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
    if pos.x >= 0 and pos.x < MapWidth and pos.y >= 0 and pos.y < MapHeight:
      env.tintMods[pos.x][pos.y] = TintModification(r: 0, g: 0, b: 0, intensity: 0)
  
  # Clear the active list for next frame
  env.activeTiles.positions.setLen(0)
  env.activeTiles.count = 0

proc updateTintModifications(env: Environment) =
  ## Update unified tint modification array based on entity positions - runs every frame
  # Clear previous frame's modifications
  env.clearTintModifications()
  
  # Process all entities and mark their affected positions as active
  for thing in env.things:
    let pos = thing.pos
    if pos.x < 0 or pos.x >= MapWidth or pos.y < 0 or pos.y >= MapHeight:
      continue
    
    case thing.kind
    of Clippy:
      # Clippies create creep spread in 5x5 area (stronger effect when planted)
      let creepIntensity = if thing.hasClaimedTerritory: 2 else: 1
      
      for dx in -2 .. 2:
        for dy in -2 .. 2:
          let creepPos = ivec2(pos.x + dx, pos.y + dy)
          if creepPos.x >= 0 and creepPos.x < MapWidth and creepPos.y >= 0 and creepPos.y < MapHeight:
            # Distance-based falloff for more organic look
            let distance = abs(dx) + abs(dy)  # Manhattan distance
            let falloff = max(1, 5 - distance)  # Stronger at center, weaker at edges (5x5 grid)
            
            env.activeTiles.positions.add(creepPos)
            env.activeTiles.count += 1
            
            # Clippy creep effect (cool colors - half speed)
            env.tintMods[creepPos.x][creepPos.y].r += int16(-15 * creepIntensity * falloff)  # Reduce red (halved)
            env.tintMods[creepPos.x][creepPos.y].g += int16(-8 * creepIntensity * falloff)   # Reduce green (halved)  
            env.tintMods[creepPos.x][creepPos.y].b += int16(20 * creepIntensity * falloff)   # Increase blue (halved)
      
    of Agent:
      # Agents create 5x stronger warmth in 3x3 area based on their tribe color
      let tribeId = thing.agentId
      if tribeId < agentVillageColors.len:
        let tribeColor = agentVillageColors[tribeId]
        
        for dx in -1 .. 1:
          for dy in -1 .. 1:
            let agentPos = ivec2(pos.x + dx, pos.y + dy)
            if agentPos.x >= 0 and agentPos.x < MapWidth and agentPos.y >= 0 and agentPos.y < MapHeight:
              # Distance-based falloff
              let distance = abs(dx) + abs(dy)
              let falloff = max(1, 3 - distance)  # Stronger at center, weaker at edges
              
              env.activeTiles.positions.add(agentPos)
              env.activeTiles.count += 1
              
              # Agent warmth effect (1.25x stronger than original)
              env.tintMods[agentPos.x][agentPos.y].r += int16((tribeColor.r - 0.7) * 63 * falloff.float32)   # Quarter of 5x = 1.25x
              env.tintMods[agentPos.x][agentPos.y].g += int16((tribeColor.g - 0.65) * 63 * falloff.float32)  # Quarter of 5x = 1.25x 
              env.tintMods[agentPos.x][agentPos.y].b += int16((tribeColor.b - 0.6) * 63 * falloff.float32)   # Quarter of 5x = 1.25x
        
    of Altar:
      # Reduce altar tint effect by 10x (minimal warm glow)
      env.activeTiles.positions.add(pos)
      env.activeTiles.count += 1
      env.tintMods[pos.x][pos.y].r += int16(5)   # Very minimal warm glow (10x reduction)
      env.tintMods[pos.x][pos.y].g += int16(5)
      env.tintMods[pos.x][pos.y].b += int16(2)
    
    of PlantedLantern:
      # Lanterns spread team colors in 5x5 area (similar to clippies but warm colors)
      if thing.lanternHealthy and thing.teamId >= 0 and thing.teamId < teamColors.len:
        let teamColor = teamColors[thing.teamId]
        
        for dx in -2 .. 2:
          for dy in -2 .. 2:
            let tintPos = ivec2(pos.x + dx, pos.y + dy)
            if tintPos.x >= 0 and tintPos.x < MapWidth and tintPos.y >= 0 and tintPos.y < MapHeight:
              # Distance-based falloff for more organic look
              let distance = abs(dx) + abs(dy)  # Manhattan distance
              let falloff = max(1, 5 - distance)  # Stronger at center, weaker at edges (5x5 grid)
              
              env.activeTiles.positions.add(tintPos)
              env.activeTiles.count += 1
              
              # Lantern warm effect (spread team colors)
              env.tintMods[tintPos.x][tintPos.y].r += int16((teamColor.r - 0.7) * 50 * falloff.float32)
              env.tintMods[tintPos.x][tintPos.y].g += int16((teamColor.g - 0.65) * 50 * falloff.float32)  
              env.tintMods[tintPos.x][tintPos.y].b += int16((teamColor.b - 0.6) * 50 * falloff.float32)
    
    else:
      discard

proc applyTintModifications(env: Environment) =
  ## Apply tint modifications to entity positions and their surrounding areas
  
  # First, apply modifications to all tiles that have tint modifications
  for tileX in 0 ..< MapWidth:
    for tileY in 0 ..< MapHeight:
      # Skip if tint modifications are below minimum threshold
      if abs(env.tintMods[tileX][tileY].r) < MinTintEpsilon and abs(env.tintMods[tileX][tileY].g) < MinTintEpsilon and abs(env.tintMods[tileX][tileY].b) < MinTintEpsilon:
        continue
      
      # Skip tinting on water tiles (rivers should remain clean)
      if env.terrain[tileX][tileY] == Water:
        continue
    
      # Get current color as integers (scaled by 1000 for precision)
      var r = int(env.tileColors[tileX][tileY].r * 1000)
      var g = int(env.tileColors[tileX][tileY].g * 1000)  
      var b = int(env.tileColors[tileX][tileY].b * 1000)
      
      # Apply unified tint modifications
      r += env.tintMods[tileX][tileY].r div 10  # 10% of the modification
      g += env.tintMods[tileX][tileY].g div 10
      b += env.tintMods[tileX][tileY].b div 10
      
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
  env.updateObservations(AgentInventoryLanternLayer, agent.pos, agent.inventoryLantern)
  
  # Give reward for planting
  agent.reward += env.config.clothReward * 0.5  # Half reward for planting
  
  inc env.stats[id].actionPlant

proc init(env: Environment) =
  # Ensure placeholder sprites for missing inventory icons (wood, spear)
  try:
    let woodPath = "tribal/data/resources/wood.png"
    let woodSrc = "tribal/data/resources/palm_fiber.png"
    if (not fileExists(woodPath)) and fileExists(woodSrc):
      copyFile(woodSrc, woodPath)
    let spearPath = "tribal/data/resources/spear.png"
    let spearSrc = "tribal/data/resources/laser.png"
    if (not fileExists(spearPath)) and fileExists(spearSrc):
      copyFile(spearSrc, spearPath)
  except:
    discard
  # Use current time for random seed to get different maps each time
  let seed = int(epochTime() * 1000)
  var r = initRand(seed)
  echo "Generating map with seed: ", seed
  
  # Initialize tile colors to base terrain colors (neutral gray-brown)
  for x in 0 ..< MapWidth:
    for y in 0 ..< MapHeight:
      env.tileColors[x][y] = TileColor(r: 0.7, g: 0.65, b: 0.6, intensity: 1.0)
      env.baseTileColors[x][y] = TileColor(r: 0.7, g: 0.65, b: 0.6, intensity: 1.0)
  
  # Initialize active tiles tracking
  env.activeTiles.positions = @[]
  env.activeTiles.count = 0
  
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
    # Use the new unified placement system
    let houseStruct = createHouse()
    var gridPtr = cast[PlacementGrid](env.grid.addr)
    var terrainPtr = env.terrain.addr
    # Simplify: random placement anywhere valid (no corner preference)
    let placementResult = findPlacement(gridPtr, terrainPtr, houseStruct, MapWidth, MapHeight, MapBorder, r, preferCorners = false)
    
    if placementResult.success:  # Valid location found
      let elements = getStructureElements(houseStruct, placementResult.position)
      
      # Clear terrain within the house area to create a clearing
      for dy in 0 ..< houseStruct.height:
        for dx in 0 ..< houseStruct.width:
          let clearX = placementResult.position.x + dx
          let clearY = placementResult.position.y + dy
          if clearX >= 0 and clearX < MapWidth and clearY >= 0 and clearY < MapHeight:
            # Clear any terrain features (wheat, trees) but keep water
            if env.terrain[clearX][clearY] != Water:
              env.terrain[clearX][clearY] = Empty
      
      # Generate a unique warm color for this village (reds, oranges, yellows)
      # Create warm colors using RGB values directly
      var villageColor: Color
      case i mod 6:
      of 0: villageColor = color(1.0, 0.4, 0.2, 1.0)    # Red-orange
      of 1: villageColor = color(1.0, 0.6, 0.2, 1.0)    # Orange
      of 2: villageColor = color(1.0, 0.8, 0.2, 1.0)    # Yellow-orange
      of 3: villageColor = color(0.9, 0.3, 0.3, 1.0)    # Crimson
      of 4: villageColor = color(1.0, 0.5, 0.4, 1.0)    # Coral
      of 5: villageColor = color(0.8, 0.4, 0.6, 1.0)    # Warm pink
      else: villageColor = color(1.0, 0.5, 0.3, 1.0)    # Default warm orange

      # Store team color for lanterns
      teamColors.add(villageColor)

      # Spawn agents around this house
      let agentsForThisHouse = min(MapAgentsPerHouse, MapRoomObjectsAgents - totalAgentsSpawned)
      
      # Add the altar with initial hearts and house bounds
      env.add(Thing(
        kind: Altar,
        pos: elements.center,
        hearts: MapObjectAltarInitialHearts  # Altar starts with default hearts
      ))
      altarColors[elements.center] = villageColor  # Associate altar position with village color
      
      # Initialize base colors for house tiles to team color
      for dx in 0 ..< houseStruct.width:
        for dy in 0 ..< houseStruct.height:
          let tileX = placementResult.position.x + dx
          let tileY = placementResult.position.y + dy
          if tileX >= 0 and tileX < MapWidth and tileY >= 0 and tileY < MapHeight:
            env.baseTileColors[tileX][tileY] = TileColor(
              r: villageColor.r,
              g: villageColor.g,
              b: villageColor.b,
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
            let worldPos = placementResult.position + ivec2(x.int32, y.int32)
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
          agentVillageColors[agentId] = villageColor
          
          # Create the agent
          env.add(Thing(
            kind: Agent,
            agentId: agentId,
            pos: agentPos,
            orientation: Orientation(r.rand(0..3)),
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
      orientation: Orientation(r.rand(0..3)),
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

  # Random spawner placement with a simple minimum distance from villages
  # Gather altar positions for distance checks
  var altarPositionsNow: seq[IVec2] = @[]
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

      # Enforce min distance from any altar (Euclidean squared)
      var okDistance = true
      for ap in altarPositionsNow:
        let dx = int(targetPos.x) - int(ap.x)
        let dy = int(targetPos.y) - int(ap.y)
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

      let nearbyPositions = env.findEmptyPositionsAround(targetPos, 1)
      if nearbyPositions.len > 0:
        env.add(createClippy(nearbyPositions[0], targetPos, r))
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
        env.add(createClippy(nearbyPositions[0], targetPos, r))

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

  for agentId in 0 ..< MapAgents:
    env.updateObservations(agentId)


proc defaultEnvironmentConfig*(): EnvironmentConfig =
  ## Create default environment configuration
  EnvironmentConfig(
    # Core game parameters
    maxSteps: 2000,
    
    # Resource configuration
    orePerBattery: 3,
    batteriesPerHeart: 2,
    
    # Combat configuration
    enableCombat: true,
    clippySpawnRate: 1.0,
    clippyDamage: 1,
    
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
    clippyKillReward: 0.0, # Disabled - not in arena
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

  # Update objects and collect new clippys to spawn
  var newClippysToSpawn: seq[Thing] = @[]
  
  for thing in env.things:
    if thing.kind == Altar:
      if thing.cooldown > 0:
        thing.cooldown -= 1
        env.updateObservations(AltarReadyLayer, thing.pos, thing.cooldown)
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
        # Spawner is ready to spawn a Clippy
        # Count nearby Clippys
        var nearbyClippyCount = 0
        for other in env.things:
          if other.kind == Clippy:
            let dist = manhattanDistance(other.pos, thing.pos)
            if dist <= 5:  # Within 5 tiles of spawner
              nearbyClippyCount += 1
        
        # Spawn a new Clippy (no limit for now)
        if true:
          # Find empty positions around spawner (simple approach)
          let emptyPositions = env.findEmptyPositionsAround(thing.pos, 2)
          if emptyPositions.len > 0:
            var r = initRand(env.currentStep)
            let spawnPos = r.sample(emptyPositions)
            
            let newClippy = createClippy(spawnPos, thing.pos, r)
            # Don't add immediately - collect for later
            newClippysToSpawn.add(newClippy)
            
            # Reset spawner cooldown based on spawn rate
            # Convert spawn rate (0.0-1.0) to cooldown steps (higher rate = lower cooldown)
            let cooldown = if env.config.clippySpawnRate > 0.0:
              max(1, int(20.0 / env.config.clippySpawnRate))  # Base 20 steps, scaled by rate
            else:
              1000  # Very long cooldown if spawn disabled
            thing.cooldown = cooldown
    elif thing.kind == Agent:
      if thing.frozen > 0:
        thing.frozen -= 1
        # Note: frozen status is visible in observations through updateObservations(id)

  # ============== CLIPPY PROCESSING ==============
  # Add newly spawned clippys from spawners
  for newClippy in newClippysToSpawn:
    env.add(newClippy)
  
  # Collect all clippys to process (to avoid modifying collection while iterating)
  var clippysToProcess: seq[Thing] = @[]
  for thing in env.things:
    if thing.kind == Clippy:
      clippysToProcess.add(thing)
  
  var clippysToRemove: seq[Thing] = @[]
  var r = initRand(env.currentStep)
  
  # First pass: Calculate movements for all clippies
  var clippyMoves: seq[tuple[clippy: Thing, moveDir: IVec2]] = @[]
  for clippy in clippysToProcess:
    let moveDir = getClippyMoveDirection(clippy, env, r)
    clippyMoves.add((clippy: clippy, moveDir: moveDir))
  
  # Second pass: Apply movements
  for move in clippyMoves:
    let clippy = move.clippy
    let moveDir = move.moveDir
    let newPos = clippy.pos + moveDir
    
    # Update clippy orientation based on movement direction
    if moveDir != ivec2(0, 0):
      clippy.orientation = 
        if moveDir.x > 0: E
        elif moveDir.x < 0: W
        elif moveDir.y > 0: S
        else: N
    
    # Check if new position is valid and empty
    if env.isEmpty(newPos):
      # Move the clippy
      env.grid[clippy.pos.x][clippy.pos.y] = nil
      clippy.pos = newPos
      env.grid[clippy.pos.x][clippy.pos.y] = clippy
      
      # Heatmap is now updated in batch at end of step function
    else:
      # Check if we're trying to move onto an altar
      let target = env.getThing(newPos)
      if not isNil(target) and target.kind == Altar:
        # Clippy reached an altar - damage it and disappear
        if target.hearts > 0:
          target.hearts = max(0, target.hearts - 1)  # Decrement altar's hearts but don't go below 0
          env.updateObservations(AltarHeartsLayer, target.pos, target.hearts)
        clippysToRemove.add(clippy)
        env.grid[clippy.pos.x][clippy.pos.y] = nil
  
  # Third pass: Handle planting decisions and age tracking
  for clippy in clippysToProcess:
    # Skip if clippy is already marked for removal
    if clippy in clippysToRemove:
      continue
    
    # Increment age counter
    clippy.turnsAlive += 1
    
    # Skip planting if already planted
    if clippy.hasClaimedTerritory:
      continue
    
    # Skip planting on first turn (turnsAlive == 1 since we just incremented)
    if clippy.turnsAlive == 1:
      continue
    
    # Plant if no nearby clippies in observation window (2-tile radius) and not on water
    if not env.hasNearbyClippies(clippy.pos, 2, clippy) and env.terrain[clippy.pos.x][clippy.pos.y] != Water:
      clippy.hasClaimedTerritory = true
  
  # ============== CLIPPY COMBAT ==============
  # Process combat between clippys and adjacent agents
  for clippy in clippysToProcess:
    # Skip if clippy is already marked for removal
    if clippy in clippysToRemove:
      continue
    
    # Check all 4 adjacent positions for agents
    let adjacentPositions = @[
      clippy.pos + ivec2(0, -1),  # North
      clippy.pos + ivec2(0, 1),   # South
      clippy.pos + ivec2(-1, 0),  # West
      clippy.pos + ivec2(1, 0)    # East
    ]
    
    for adjPos in adjacentPositions:
      # Skip if position is out of bounds
      if adjPos.x < 0 or adjPos.x >= MapWidth or adjPos.y < 0 or adjPos.y >= MapHeight:
        continue
      
      let adjacentThing = env.getThing(adjPos)
      if not isNil(adjacentThing) and adjacentThing.kind == Agent:
        # Combat occurs! 50% chance agent dies, 100% chance clippy dies
        let combatRoll = r.rand(0.0 .. 1.0)
        
        # Clippy always dies in combat
        if clippy notin clippysToRemove:
          clippysToRemove.add(clippy)
          env.grid[clippy.pos.x][clippy.pos.y] = nil
        
        # Check if agent can defend against the attack
        let agentWouldDie = combatRoll < 0.5
        var agentSurvived = false
        
        if agentWouldDie:
          # Check defense items - only armor now (3 uses)
          if adjacentThing.inventoryArmor > 0:
            adjacentThing.inventoryArmor -= 1
            env.updateObservations(AgentInventoryArmorLayer, adjacentThing.pos, adjacentThing.inventoryArmor)
            agentSurvived = true
        
        # Only kill agent if they would die, have no defense, and combat is enabled
        if agentWouldDie and not agentSurvived and env.config.enableCombat:
          # Agent dies - mark for respawn at altar
          adjacentThing.frozen = 999999  # Mark as dead (will be respawned)
          env.terminated[adjacentThing.agentId] = 1.0
          # Apply death penalty
          adjacentThing.reward += env.config.deathPenalty
          
          # Clear the agent from its current position
          env.grid[adjacentThing.pos.x][adjacentThing.pos.y] = nil
          
          # Clear inventory when agent dies
          env.updateObservations(AgentInventoryBatteryLayer, adjacentThing.pos, 0)
          env.updateObservations(AgentInventoryLanternLayer, adjacentThing.pos, 0)
          env.updateObservations(AgentInventoryArmorLayer, adjacentThing.pos, 0)
          env.updateObservations(AgentInventoryBreadLayer, adjacentThing.pos, 0)
        
        # Break after first combat (clippy is already dead)
        break
      
      # Check for lanterns adjacent to this clippy
      elif not isNil(adjacentThing) and adjacentThing.kind == PlantedLantern:
        # Lantern vs Clippy combat: 100% clippy dies, 25% lantern dies
        if clippy notin clippysToRemove:
          clippysToRemove.add(clippy)
          env.grid[clippy.pos.x][clippy.pos.y] = nil
        
        # 25% chance lantern dies
        let lanternRoll = r.rand(0.0 .. 1.0)
        if lanternRoll < 0.25:
          # Lantern is destroyed - mark for respawn if possible
          adjacentThing.lanternHealthy = false
          
          # Find the team's altar to consume a heart for respawn
          if adjacentThing.teamId >= 0:
            var teamAltar: Thing = nil
            # Find altar by matching team agents' home altars
            for agent in env.agents:
              if agent.agentId div 5 == adjacentThing.teamId and agent.homeAltar.x >= 0:
                for thing in env.things:
                  if thing.kind == Altar and thing.pos == agent.homeAltar:
                    teamAltar = thing
                    break
                break
            
            if not isNil(teamAltar) and teamAltar.hearts > 0:
              # Consume a heart to respawn lantern
              teamAltar.hearts -= 1
              env.updateObservations(AltarHeartsLayer, teamAltar.pos, teamAltar.hearts)
              adjacentThing.lanternHealthy = true  # Respawn the lantern
            else:
              # No hearts available, remove the lantern permanently
              let idx = env.things.find(adjacentThing)
              if idx >= 0:
                env.things.del(idx)
              env.grid[adjacentThing.pos.x][adjacentThing.pos.y] = nil
        
        # Break after processing this lantern
        break
  
  # ============== CLIPPY CLEANUP ==============
  # Remove clippys that died in combat or touched altars
  for clippy in clippysToRemove:
    let idx = env.things.find(clippy)
    if idx >= 0:
      env.things.del(idx)

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
        
        # Find an empty position around the altar
        let emptyPositions = env.findEmptyPositionsAround(altar.pos, 2)
        if emptyPositions.len > 0:
          # Respawn the agent
          agent.pos = emptyPositions[0]
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
          env.updateObservations(AgentLayer, agent.pos, 1)
          env.updateObservations(AgentInventoryOreLayer, agent.pos, 0)
          env.updateObservations(AgentInventoryBatteryLayer, agent.pos, 0)
          env.updateObservations(AgentInventoryWaterLayer, agent.pos, 0)
          env.updateObservations(AgentInventoryWheatLayer, agent.pos, 0)
          env.updateObservations(AgentInventoryWoodLayer, agent.pos, 0)
          env.updateObservations(AgentInventorySpearLayer, agent.pos, 0)
          env.updateObservations(AgentOrientationLayer, agent.pos, agent.orientation.int)
          env.updateObservations(agentId)
  
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
    env.shouldReset = true
  
  # Check if all agents are terminated/truncated
  var allDone = true
  for i in 0..<MapAgents:
    if env.terminated[i] == 0.0 and env.truncated[i] == 0.0:
      allDone = false
      break
  if allDone:
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
  env.terrain.clear()
  env.observations.clear()
  env.init()

proc applyTeamAltarReward*(env: Environment) =
  # Find all altars and their heart counts
  for thing in env.things:
    if thing.kind == Altar:
      let altarHearts = thing.hearts.float32
      # Find all agents with this altar as their home
      for agent in env.agents:
        if agent.homeAltar == thing.pos:
          # Each agent gets reward equal to altar hearts
          agent.reward += altarHearts
          # Optional: Extra bonus if altar is well-defended (>5 hearts)
          if altarHearts > 5:
            agent.reward += (altarHearts - 5) * 0.5  # Bonus for surplus



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
