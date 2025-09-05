import std/[strformat, random, strutils, tables, times, math], vmath, chroma
import terrain, objects
export terrain, objects

const
  # Map layout constants
  MapLayoutRoomsX* = 1
  MapLayoutRoomsY* = 1
  MapBorder* = 4
  MapRoomWidth* = 96  # 100 - 4 border = 96
  MapRoomHeight* = 46  # 50 - 4 border = 46
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
  MapObjectAgentFreezeDuration* = 10  # Temporary freeze when caught by clippy

  MapObjectAltarInitialHearts* = 5
  MapObjectAltarCooldown* = 10
  MapObjectAltarRespawnCost* = 1
  MapObjectConverterCooldown* = 0

  MapObjectMineCooldown* = 5
  MapObjectMineInitialResources* = 30
  MapObjectMineUseCost* = 0
  SpawnerCooldown* = 30  # Steps between Clippy spawns
  ObservationLayers* = 19
  ObservationWidth* = 11
  ObservationHeight* = 11
  # Computed
  MapAgents* = MapRoomObjectsAgents * MapLayoutRoomsX * MapLayoutRoomsY
  MapWidth* = MapLayoutRoomsX * (MapRoomWidth + MapRoomBorder) + MapBorder
  MapHeight* = MapLayoutRoomsY * (MapRoomHeight + MapRoomBorder) + MapBorder

proc ivec2*(x, y: int): IVec2 =
  result.x = x.int32
  result.y = y.int32

# Global village color management
var agentVillageColors*: seq[Color] = @[]
var altarColors*: Table[IVec2, Color] = initTable[IVec2, Color]()

type
  OrientationDelta* = tuple[x, y: int]
  ObservationName* = enum
    AgentLayer = 0
    AgentOrientationLayer = 1
    AgentInventoryOreLayer = 2
    AgentInventoryBatteryLayer = 3
    AgentInventoryWaterLayer = 4
    AgentInventoryWheatLayer = 5
    AgentInventoryWoodLayer = 6
    AgentInventorySpearLayer = 7
    AgentInventoryHatLayer = 8
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

  Orientation* = enum
    N = 0  # North (Up)
    S = 1  # South (Down) 
    W = 2  # West (Left)
    E = 3  # East (Right)
    NW = 4 # Northwest (Up-Left)
    NE = 5 # Northeast (Up-Right)
    SW = 6 # Southwest (Down-Left)
    SE = 7 # Southeast (Down-Right)

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
    inventoryHat*: int      # Hats from weaving loom (1-hit protection)
    inventoryArmor*: int    # Armor from armory (3-hit protection, tracks remaining uses)
    reward*: float32
    homeAltar*: IVec2      # Position of agent's home altar for respawning
    # Clippy:
    homeSpawner*: IVec2     # Position of clippy's home spawner
    wanderRadius*: int     # Current radius for concentric circle wandering
    wanderAngle*: float    # Current angle in the circle pattern
    targetPos*: IVec2      # Current target (agent or altar)
    wanderStepsRemaining*: int  # Steps to wander before checking for targets

  Stats* = ref object
    # Agent Stats:
    actionInvalid*: int
    actionMove*: int
    actionNoop*: int
    actionRotate*: int
    actionSwap*: int
    actionUse*: int
    actionUseMine*: int
    actionUseConverter*: int
    actionUseAltar*: int
    actionGet*: int
    actionGetWater*: int
    actionGetWheat*: int
    actionGetWood*: int

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
    
  Environment* = ref object
    currentStep*: int
    things*: seq[Thing]
    agents*: seq[Thing]
    grid*: array[MapWidth, array[MapHeight, Thing]]
    terrain*: TerrainGrid
    tileColors*: array[MapWidth, array[MapHeight, TileColor]]  # Main color array
    baseTileColors*: array[MapWidth, array[MapHeight, TileColor]]  # Base colors (terrain)
    agentTintMods*: array[MapWidth, array[MapHeight, TintModification]]  # Agent heat contributions
    clippyTintMods*: array[MapWidth, array[MapHeight, TintModification]]  # Clippy cold contributions  
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

# Global variables for backward compatibility
# Consider using gamestate module for better encapsulation
var
  env*: Environment  # Global environment instance
  selection*: Thing  # Currently selected entity for UI interaction

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
          break
      result.add(cell)
    result.add("\n")

proc renderObservations*(env: Environment): string =
  const featureNames = [
    "agent",
    "agent:orientation",
    "agent:inv:ore",
    "agent:inv:battery",
    "agent:inv:water",
    "agent:inv:wheat",
    "agent:inv:wood",
    "agent:inv:spear",
    "agent:inv:hat",
    "agent:inv:armor",
    "wall",
    "mine",
    "mine:resources",
    "mine:ready",
    "converter",
    "converter:ready",
    "altar",
    "altar:hearts",
    "altar:ready",
  ]
  for id, obs in env.observations:
    result.add "Agent: " & $id & "\n"
    for layer in 0 ..< ObservationLayers:
      result.add "Feature " & $featureNames[layer] & " " & $layer & "\n"
      for y in 0 ..< ObservationHeight:
        for x in 0 ..< ObservationWidth:
          result.formatValue(obs[layer][x][y], "4d")
        result.add "\n"

proc clear[T](s: var openarray[T]) =
  let p = cast[pointer](s[0].addr)
  zeroMem(p, s.len * sizeof(T))

proc clear[N: int, T](s: ptr array[N, T]) =
  let p = cast[pointer](s[][0].addr)
  zeroMem(p, s[].len * sizeof(T))

proc updateObservations*(env: Environment, agentId: int) =
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

      of Wall:
        # Layer 8: WallLayer
        obs[8][x][y] = 1

      of Mine:
        # Layer 9: MineLayer
        obs[9][x][y] = 1
        # Layer 10: MineResourceLayer
        obs[10][x][y] = thing.resources.uint8
        # Layer 11: MineReadyLayer
        obs[11][x][y] = (thing.cooldown == 0).uint8

      of Converter:
        # Layer 12: ConverterLayer
        obs[12][x][y] = 1
        # Layer 13: ConverterReadyLayer
        obs[13][x][y] = (thing.cooldown == 0).uint8

      of Altar:
        # Layer 14: AltarLayer
        obs[14][x][y] = 1
        # Layer 15: AltarHeartsLayer
        obs[15][x][y] = thing.hearts.uint8
        # Layer 16: AltarReadyLayer
        obs[16][x][y] = (thing.cooldown == 0).uint8
      
      of Spawner:
        # Spawner acts similar to altar for observations
        obs[14][x][y] = 1
        obs[15][x][y] = thing.hearts.uint8
        obs[16][x][y] = (thing.cooldown == 0).uint8
      
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
        obs[8][x][y] = 1  # Use the wall layer for now

proc updateObservations*(
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


proc getThing*(env: Environment, pos: IVec2): Thing =
  if pos.x < 0 or pos.x >= MapWidth or pos.y < 0 or pos.y >= MapHeight:
    return nil
  return env.grid[pos.x][pos.y]

# Orientation deltas for each direction
const OrientationDeltas*: array[8, OrientationDelta] = [
  (x: 0, y: -1),   # N (North)
  (x: 0, y: 1),    # S (South)
  (x: -1, y: 0),   # W (West)
  (x: 1, y: 0),    # E (East)
  (x: -1, y: -1),  # NW (Northwest)
  (x: 1, y: -1),   # NE (Northeast)
  (x: -1, y: 1),   # SW (Southwest)
  (x: 1, y: 1)     # SE (Southeast)
]

proc getOrientationDelta*(orient: Orientation): OrientationDelta =
  OrientationDeltas[ord(orient)]

proc isDiagonal*(orient: Orientation): bool =
  ord(orient) >= ord(NW)

proc getOpposite*(orient: Orientation): Orientation =
  case orient
  of N: S
  of S: N
  of W: E
  of E: W
  of NW: SE
  of NE: SW
  of SW: NE
  of SE: NW

proc isEmpty*(env: Environment, pos: IVec2): bool =
  if pos.x < 0 or pos.x >= MapWidth or pos.y < 0 or pos.y >= MapHeight:
    return false
  return env.grid[pos.x][pos.y] == nil

proc findNearestAltar*(env: Environment, fromPos: IVec2): IVec2 =
  ## Find the position of the nearest altar to a given position
  var nearestAltar = ivec2(-1, -1)
  var minDist = int.high
  
  for thing in env.things:
    if thing.kind == Altar:
      let dist = abs(thing.pos.x - fromPos.x) + abs(thing.pos.y - fromPos.y)
      if dist < minDist:
        minDist = dist
        nearestAltar = thing.pos
  
  return nearestAltar

proc createClippy*(pos: IVec2, homeSpawner: IVec2, targetAltar: IVec2, r: var Rand): Thing =
  ## Create a new Clippy that moves toward the nearest altar
  Thing(
    kind: Clippy,
    pos: pos,
    orientation: Orientation(r.rand(0..3)),
    homeSpawner: homeSpawner,
    wanderRadius: 0,  # Unused - we use direct movement now
    wanderAngle: 0.0,  # Unused - we use direct movement now
    targetPos: ivec2(-1, -1),  # Unused - we find nearest altar dynamically
    wanderStepsRemaining: 0,  # Unused - we move every step
  )

proc orientationToVec*(orientation: Orientation): IVec2 =
  case orientation
  of N: result = ivec2(0, -1)
  of S: result = ivec2(0, 1)
  of E: result = ivec2(1, 0)
  of W: result = ivec2(-1, 0)
  of NW: result = ivec2(-1, -1)
  of NE: result = ivec2(1, -1)
  of SW: result = ivec2(-1, 1)
  of SE: result = ivec2(1, 1)

proc relativeLocation*(orientation: Orientation, distance, offset: int): IVec2 =
  case orientation
  of N: ivec2(-offset, -distance)
  of S: ivec2(offset, distance)
  of E: ivec2(distance, -offset)
  of W: ivec2(-distance, offset)
  of NW: ivec2(-distance - offset, -distance + offset)
  of NE: ivec2(distance - offset, -distance - offset)
  of SW: ivec2(-distance + offset, distance + offset)
  of SE: ivec2(distance + offset, distance - offset)

proc noopAction(env: Environment, id: int, agent: Thing) =
  inc env.stats[id].actionNoop

proc moveAction(env: Environment, id: int, agent: Thing, argument: int) =
  # Validate orientation argument
  if argument < 0 or argument > 7:
    inc env.stats[id].actionInvalid
    return
  
  let moveOrientation = Orientation(argument)
  let delta = getOrientationDelta(moveOrientation)
  
  var newPos = agent.pos
  newPos.x += int32(delta.x)
  newPos.y += int32(delta.y)
  
  # Update orientation to face movement direction
  let newOrientation = moveOrientation
  if env.isEmpty(newPos):
    env.grid[agent.pos.x][agent.pos.y] = nil
    env.updateObservations(AgentLayer, agent.pos, 0)
    env.updateObservations(AgentOrientationLayer, agent.pos, 0)
    env.updateObservations(AgentInventoryOreLayer, agent.pos, 0)
    env.updateObservations(AgentInventoryBatteryLayer, agent.pos, 0)
    env.updateObservations(AgentInventoryWaterLayer, agent.pos, 0)
    env.updateObservations(AgentInventoryWheatLayer, agent.pos, 0)
    env.updateObservations(AgentInventoryWoodLayer, agent.pos, 0)
    agent.pos = newPos
    agent.orientation = newOrientation
    env.grid[agent.pos.x][agent.pos.y] = agent
    
    # Heatmap is now updated in batch at end of step function
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

proc rotateAction(env: Environment, id: int, agent: Thing, argument: int) =
  if argument < 0 or argument > 7:
    inc env.stats[id].actionInvalid
    return
  agent.orientation = Orientation(argument)
  env.updateObservations(AgentOrientationLayer, agent.pos, argument)
  inc env.stats[id].actionRotate


proc useAction(env: Environment, id: int, agent: Thing, argument: int) =
  ## Use resources - argument specifies direction (0=N, 1=S, 2=W, 3=E, 4=NW, 5=NE, 6=SW, 7=SE)
  if argument > 7:
    inc env.stats[id].actionInvalid
    return
  
  # Calculate target position based on orientation argument
  let useOrientation = Orientation(argument)
  let delta = getOrientationDelta(useOrientation)
  var usePos = agent.pos
  usePos.x += int32(delta.x)
  usePos.y += int32(delta.y)
  var thing = env.getThing(usePos)
  if thing == nil:
    inc env.stats[id].actionInvalid
    return
  case thing.kind
  of Wall:
    inc env.stats[id].actionInvalid
  of Agent:
    inc env.stats[id].actionInvalid
  of Altar:
    if thing.cooldown == 0 and agent.inventoryBattery >= 1:
      # Agent deposits a battery as a heart into the altar (no max capacity)
      agent.reward += 1
      agent.inventoryBattery -= 1
      thing.hearts += 1  # Add one heart to altar
      env.updateObservations(AgentInventoryBatteryLayer, agent.pos, agent.inventoryBattery)
      env.updateObservations(AltarHeartsLayer, thing.pos, thing.hearts)
      thing.cooldown = MapObjectAltarCooldown
      env.updateObservations(AltarReadyLayer, thing.pos, thing.cooldown)
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
      agent.reward += RewardMineOre  # Small shaped reward
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
      thing.cooldown = 0
      env.updateObservations(ConverterReadyLayer, thing.pos, 1)  # Always ready
      agent.reward += RewardConvertOreToBattery  # Small shaped reward
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
      agent.reward += RewardCraftSpear  # Small shaped reward
      inc env.stats[id].actionUse
    else:
      inc env.stats[id].actionInvalid
  of Armory, ClayOven, WeavingLoom:
    # Production building crafting logic
    let canUse = thing.cooldown == 0 and (
      case thing.kind:
      of Armory: agent.inventoryOre >= 2
      of ClayOven: agent.inventoryWheat >= 1  
      of WeavingLoom: agent.inventoryWheat >= 1
      else: false
    )
    
    if canUse:
      # Consume resources and apply rewards based on building type
      case thing.kind:
      of Armory:
        agent.inventoryOre -= 2
        agent.reward += RewardCraftArmor
        thing.cooldown = 20
        env.updateObservations(AgentInventoryOreLayer, agent.pos, agent.inventoryOre)
      of ClayOven:
        agent.inventoryWheat -= 1
        agent.reward += RewardCraftFood
        thing.cooldown = 10
        env.updateObservations(AgentInventoryWheatLayer, agent.pos, agent.inventoryWheat)
      of WeavingLoom:
        agent.inventoryWheat -= 1
        agent.reward += RewardCraftCloth
        thing.cooldown = 15
        env.updateObservations(AgentInventoryWheatLayer, agent.pos, agent.inventoryWheat)
      else: discard
      inc env.stats[id].actionUse
    else:
      inc env.stats[id].actionInvalid
  
  of Spawner, Clippy:
    # Can't use spawners or Clippys
    inc env.stats[id].actionInvalid

proc attackAction*(env: Environment, id: int, agent: Thing, argument: int) =
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
    
    # Consume the spear
    agent.inventorySpear = 0
    env.updateObservations(AgentInventorySpearLayer, agent.pos, agent.inventorySpear)
    
    # Give reward for destroying Clippy
    agent.reward += RewardDestroyClippy  # Moderate reward for defense
    
    inc env.stats[id].actionUse
  else:
    # Attack missed or no valid target
    inc env.stats[id].actionInvalid

proc getAction(env: Environment, id: int, agent: Thing, argument: int) =
  ## Get resources from terrain (water, wheat, wood) - argument specifies direction (0=N, 1=S, 2=W, 3=E, 4=NW, 5=NE, 6=SW, 7=SE)
  if argument > 7:
    inc env.stats[id].actionInvalid
    return
  
  # Calculate target position based on orientation argument
  let getOrientation = Orientation(argument)
  let delta = getOrientationDelta(getOrientation)
  var targetPos = agent.pos
  targetPos.x += int32(delta.x)
  targetPos.y += int32(delta.y)
  
  # Check bounds
  if targetPos.x < 0 or targetPos.x >= MapWidth or targetPos.y < 0 or targetPos.y >= MapHeight:
    inc env.stats[id].actionInvalid
    return
  
  # Check what terrain is at target position
  case env.terrain[targetPos.x][targetPos.y]:
  of Water:
    # Get water (max 5 water inventory)
    if agent.inventoryWater < 5:
      agent.inventoryWater += 1
      env.terrain[targetPos.x][targetPos.y] = Empty  # Remove water tile
      env.updateObservations(AgentInventoryWaterLayer, agent.pos, agent.inventoryWater)
      agent.reward += RewardGetWater  # Small shaped reward
      inc env.stats[id].actionGetWater
      inc env.stats[id].actionGet
    else:
      inc env.stats[id].actionInvalid  # Inventory full
  
  of Wheat:
    # Get wheat (max 5 wheat inventory)
    if agent.inventoryWheat < 5:
      agent.inventoryWheat += 1
      env.terrain[targetPos.x][targetPos.y] = Empty  # Remove wheat tile
      env.updateObservations(AgentInventoryWheatLayer, agent.pos, agent.inventoryWheat)
      env.updateObservations(AgentInventoryWoodLayer, agent.pos, agent.inventoryWood)
      agent.reward += RewardGetWheat  # Small shaped reward
      inc env.stats[id].actionGetWheat
      inc env.stats[id].actionGet
    else:
      inc env.stats[id].actionInvalid  # Inventory full
  
  of Tree:
    # Get wood (max 5 wood inventory)
    if agent.inventoryWood < 5:
      agent.inventoryWood += 1
      env.terrain[targetPos.x][targetPos.y] = Empty  # Remove tree tile
      env.updateObservations(AgentInventoryWheatLayer, agent.pos, agent.inventoryWheat)
      env.updateObservations(AgentInventoryWoodLayer, agent.pos, agent.inventoryWood)
      agent.reward += RewardGetWood  # Small shaped reward (slightly higher for spear path)
      inc env.stats[id].actionGetWood
      inc env.stats[id].actionGet
    else:
      inc env.stats[id].actionInvalid  # Inventory full
  
  of Empty:
    inc env.stats[id].actionInvalid  # Nothing to get


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

proc shieldAction(env: Environment, id: int, agent: Thing, argument: int) =
  ## Shield action
  inc env.stats[id].actionInvalid

proc giftAction(env: Environment, id: int, agent: Thing) =
  ## Gift action
  inc env.stats[id].actionInvalid

# ============== CLIPPY AI ==============

proc manhattanDistance*(a, b: IVec2): int =
  ## Calculate Manhattan distance between two points
  return abs(a.x - b.x) + abs(a.y - b.y)

proc getDirectionToward*(fromPos, toPos: IVec2): IVec2 =
  ## Calculate optimal unit direction from one position toward another
  ## Allows diagonal movement for more efficient pathfinding
  let dx = toPos.x - fromPos.x
  let dy = toPos.y - fromPos.y
  
  if dx == 0 and dy == 0:
    return ivec2(0, 0)
  
  # Return normalized direction vector (allows diagonal movement)
  result.x = 
    if dx > 0: 1
    elif dx < 0: -1
    else: 0
  
  result.y = 
    if dy > 0: 1
    elif dy < 0: -1
    else: 0

proc getClippyMoveDirection*(clippyPos: IVec2, things: seq[Thing], r: var Rand): IVec2 =
  ## Simple Clippy AI: Move directly toward the nearest altar using oracle knowledge
  
  var nearestAltar = ivec2(-1, -1)
  var minDist = int.high
  
  # Oracle knowledge: scan all entities to find closest altar
  for thing in things:
    if isNil(thing) or thing.kind != Altar:
      continue
    
    let dist = manhattanDistance(clippyPos, thing.pos)
    if dist < minDist:
      minDist = dist
      nearestAltar = thing.pos
  
  # Move directly toward nearest altar if found
  if nearestAltar.x >= 0:
    return getDirectionToward(clippyPos, nearestAltar)
  
  # No altar found - shouldn't happen in normal gameplay
  return ivec2(0, 0)


# proc updateGrid(env: Environment) =
#   ## Update the grid
#   for x in 0 ..< MapWidth:
#     for y in 0 ..< MapHeight:
#       env.grid[x][y] = nil
#   for thing in env.things:
#     env.grid[thing.pos.x][thing.pos.y] = thing

proc isValidEmptyPosition(env: Environment, pos: IVec2): bool =
  ## Check if a position is within map bounds, empty, and not water
  pos.x >= MapBorder and pos.x < MapWidth - MapBorder and
  pos.y >= MapBorder and pos.y < MapHeight - MapBorder and
  env.isEmpty(pos) and env.terrain[pos.x][pos.y] != Water

proc generateRandomMapPosition(r: var Rand): IVec2 =
  ## Generate a random position within map boundaries
  ivec2(r.rand(MapBorder ..< MapWidth - MapBorder), r.rand(MapBorder ..< MapHeight - MapBorder))

proc findEmptyPositionsAround(env: Environment, center: IVec2, radius: int): seq[IVec2] =
  ## Find empty positions around a center point within a given radius
  result = @[]
  for dx in -radius .. radius:
    for dy in -radius .. radius:
      if dx == 0 and dy == 0:
        continue  # Skip the center position
      let pos = ivec2(center.x + dx, center.y + dy)
      if env.isValidEmptyPosition(pos):
        result.add(pos)


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

proc clearTintModifications*(env: Environment) =
  ## Clear only active tile modifications for performance
  for pos in env.activeTiles.positions:
    if pos.x >= 0 and pos.x < MapWidth and pos.y >= 0 and pos.y < MapHeight:
      env.agentTintMods[pos.x][pos.y] = TintModification(r: 0, g: 0, b: 0, intensity: 0)
      env.clippyTintMods[pos.x][pos.y] = TintModification(r: 0, g: 0, b: 0, intensity: 0)
  
  # Clear the active list for next frame
  env.activeTiles.positions.setLen(0)
  env.activeTiles.count = 0

proc updateTintModifications*(env: Environment) =
  ## Update tint modification arrays based on entity positions - runs every frame
  # Clear previous frame's modifications
  env.clearTintModifications()
  
  # Process all entities and mark their current positions as active
  for thing in env.things:
    let pos = thing.pos
    if pos.x < 0 or pos.x >= MapWidth or pos.y < 0 or pos.y >= MapHeight:
      continue
    
    # Add to active tiles list
    env.activeTiles.positions.add(pos)
    env.activeTiles.count += 1
      
    case thing.kind
    of Clippy:
      # Clippies make tiles colder (stronger effect)
      env.clippyTintMods[pos.x][pos.y].r = -30'i16  # Reduce red
      env.clippyTintMods[pos.x][pos.y].g = -15'i16  # Reduce green
      env.clippyTintMods[pos.x][pos.y].b = 40'i16   # Increase blue
      
    of Agent:
      # Agents add warmth based on their tribe color
      let tribeId = thing.agentId
      if tribeId < agentVillageColors.len:
        let tribeColor = agentVillageColors[tribeId]
        # Stronger tribe color effect
        env.agentTintMods[pos.x][pos.y].r = int16((tribeColor.r - 0.7) * 50)
        env.agentTintMods[pos.x][pos.y].g = int16((tribeColor.g - 0.65) * 50)
        env.agentTintMods[pos.x][pos.y].b = int16((tribeColor.b - 0.6) * 50)
        
    of Altar:
      discard
    else:
      discard

proc applyTintModifications*(env: Environment) =
  ## Apply tint modifications only to active tiles - very fast
  
  # Process only tiles that have entities on them (sparse processing)
  for pos in env.activeTiles.positions:
    if pos.x < 0 or pos.x >= MapWidth or pos.y < 0 or pos.y >= MapHeight:
      continue
    
    let x = pos.x
    let y = pos.y
    
    # Get current color as integers (scaled by 1000 for precision)
    var r = int(env.tileColors[x][y].r * 1000)
    var g = int(env.tileColors[x][y].g * 1000)  
    var b = int(env.tileColors[x][y].b * 1000)
    
    # Apply modifications with stronger effects
    if env.agentTintMods[x][y].r != 0 or env.agentTintMods[x][y].g != 0 or env.agentTintMods[x][y].b != 0:
      # Agent effect - warm shift
      r += env.agentTintMods[x][y].r div 10  # 10% of the modification
      g += env.agentTintMods[x][y].g div 10
      b += env.agentTintMods[x][y].b div 10
    
    if env.clippyTintMods[x][y].r != 0 or env.clippyTintMods[x][y].g != 0 or env.clippyTintMods[x][y].b != 0:
      # Clippy effect - cold shift (stronger)
      r += env.clippyTintMods[x][y].r div 8  # 12.5% of the modification
      g += env.clippyTintMods[x][y].g div 8
      b += env.clippyTintMods[x][y].b div 8
    
    
    # Convert back to float with clamping
    env.tileColors[x][y].r = min(max(r.float32 / 1000.0, 0.3), 1.2)
    env.tileColors[x][y].g = min(max(g.float32 / 1000.0, 0.3), 1.2)
    env.tileColors[x][y].b = min(max(b.float32 / 1000.0, 0.3), 1.2)
  
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
        if abs(env.tileColors[x][y].r - baseR) > 0.01 or 
           abs(env.tileColors[x][y].g - baseG) > 0.01 or 
           abs(env.tileColors[x][y].b - baseB) > 0.01:
          env.tileColors[x][y].r = env.tileColors[x][y].r * decay + baseR * (1.0 - decay)
          env.tileColors[x][y].g = env.tileColors[x][y].g * decay + baseG * (1.0 - decay)
          env.tileColors[x][y].b = env.tileColors[x][y].b * decay + baseB * (1.0 - decay)
        
        # Also decay intensity back to base intensity
        let baseIntensity = env.baseTileColors[x][y].intensity
        if abs(env.tileColors[x][y].intensity - baseIntensity) > 0.01:
          env.tileColors[x][y].intensity = env.tileColors[x][y].intensity * decay + baseIntensity * (1.0 - decay)

proc add*(env: Environment, thing: Thing) =
  env.things.add(thing)
  if thing.kind == Agent:
    env.agents.add(thing)
    env.stats.add(Stats())
  env.grid[thing.pos.x][thing.pos.y] = thing

proc init(env: Environment) =
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

  for i in 0 ..< MapRoomObjectsWalls:
    let pos = r.randomEmptyPos(env)
    env.add(Thing(kind: Wall, pos: pos))

  # Agents will now spawn with their villages/houses below
  # Clear and prepare village colors arrays
  agentVillageColors.setLen(MapRoomObjectsAgents)  # Allocate space for all agents
  altarColors.clear()  # Clear altar colors from previous game
  # Spawn houses with their altars, walls, and associated agents (tribes)
  let numHouses = MapRoomObjectsHouses
  var totalAgentsSpawned = 0
  var usedCorners: seq[int] = @[]  # Track which corners have been used
  for i in 0 ..< numHouses:
    # Use the new unified placement system
    let houseStruct = createHouse()
    var gridPtr = cast[PlacementGrid](env.grid.addr)
    var terrainPtr = env.terrain.addr
    let placementResult = findPlacement(gridPtr, terrainPtr, houseStruct, MapWidth, MapHeight, MapBorder, r, preferCorners = true, excludedCorners = usedCorners)
    
    if placementResult.success:  # Valid location found
      # Track which corner was used (if any)
      if placementResult.cornerUsed >= 0:
        usedCorners.add(placementResult.cornerUsed)
      
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
            frozen: 0,
          ))
          
          totalAgentsSpawned += 1
          if totalAgentsSpawned >= MapRoomObjectsAgents:
            break
      
      # Note: Entrances are left empty (no walls placed there)
  
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
      frozen: 0,
    ))
    
    totalAgentsSpawned += 1

  # Calculate spawner positions in the middle area of the map
  # Spawners are distributed to provide equal access from corners
  
  let centerX = MapWidth div 2
  let centerY = MapHeight div 2
  
  # Define spawner positions based on number of houses
  var spawnerPositions: seq[IVec2] = @[]
  
  if numHouses == 3:
    # For 3 houses/altars, create a triangular spawner arrangement
    # Place spawners in positions that are equidistant from center but closer to their respective corners
    spawnerPositions = @[
      ivec2(centerX - 15, centerY - 8),  # Upper-left spawner
      ivec2(centerX + 15, centerY - 8),  # Upper-right spawner
      ivec2(centerX, centerY + 10)       # Bottom-center spawner
    ]
  elif numHouses == 4:
    # For 4 houses, create a square pattern
    spawnerPositions = @[
      ivec2(centerX - 15, centerY - 8),
      ivec2(centerX + 15, centerY - 8),
      ivec2(centerX - 15, centerY + 8),
      ivec2(centerX + 15, centerY + 8)
    ]
  else:
    # Fallback: distribute spawners evenly in a circle around center
    for i in 0 ..< numHouses:
      let angle = (2.0 * PI * i.float) / numHouses.float
      let radius = 15.0
      let x = centerX + (radius * cos(angle)).int
      let y = centerY + (radius * sin(angle)).int
      spawnerPositions.add(ivec2(x, y))
  
  # Spawn spawners with Clippys at calculated positions
  for i in 0 ..< spawnerPositions.len:
    let spawnerStruct = createSpawner()
    var targetPos = spawnerPositions[i]
    
    # Try to place spawner at or near the target position
    var placed = false
    for attempt in 0 ..< 10:
      # Add some random offset if initial position is blocked
      if attempt > 0:
        targetPos = spawnerPositions[i] + ivec2(
          r.rand(-5 .. 5),
          r.rand(-5 .. 5)
        )
      
      # Check if position is valid (within bounds and not on water/walls)
      if targetPos.x >= MapBorder + spawnerStruct.width div 2 and 
         targetPos.x < MapWidth - MapBorder - spawnerStruct.width div 2 and
         targetPos.y >= MapBorder + spawnerStruct.height div 2 and 
         targetPos.y < MapHeight - MapBorder - spawnerStruct.height div 2:
        
        # Check if area is clear
        var areaValid = true
        for dx in -(spawnerStruct.width div 2) .. (spawnerStruct.width div 2):
          for dy in -(spawnerStruct.height div 2) .. (spawnerStruct.height div 2):
            let checkPos = targetPos + ivec2(dx, dy)
            if checkPos.x >= 0 and checkPos.x < MapWidth and 
               checkPos.y >= 0 and checkPos.y < MapHeight:
              if not env.isEmpty(checkPos) or env.terrain[checkPos.x][checkPos.y] == Water:
                areaValid = false
                break
          if not areaValid:
            break
        
        if areaValid:
          # Clear terrain within the spawner area
          for dx in -(spawnerStruct.width div 2) .. (spawnerStruct.width div 2):
            for dy in -(spawnerStruct.height div 2) .. (spawnerStruct.height div 2):
              let clearPos = targetPos + ivec2(dx, dy)
              if clearPos.x >= 0 and clearPos.x < MapWidth and 
                 clearPos.y >= 0 and clearPos.y < MapHeight:
                if env.terrain[clearPos.x][clearPos.y] != Water:
                  env.terrain[clearPos.x][clearPos.y] = Empty
          
          # Add the spawner
          env.add(Thing(
            kind: Spawner,
            pos: targetPos,
            cooldown: 0,
            homeSpawner: targetPos  # A spawner is its own home
          ))
          
          # Find an empty position adjacent to the spawner for initial clippy
          let nearbyPositions = env.findEmptyPositionsAround(targetPos, 1)
          if nearbyPositions.len > 0:
            let nearestAltar = env.findNearestAltar(targetPos)
            env.add(createClippy(nearbyPositions[0], targetPos, nearestAltar, r))
          
          placed = true
          break
    
    # Fall back to random placement if position is blocked
    if not placed:
      # Simple random placement fallback
      var fallbackPos = r.randomEmptyPos(env)
      
      # Try to find a clear area for the spawner
      for attempt in 0 ..< 20:
        fallbackPos = r.randomEmptyPos(env)
        # Check if we have enough space around this position
        var hasSpace = true
        for dx in -1 .. 1:
          for dy in -1 .. 1:
            let checkPos = fallbackPos + ivec2(dx, dy)
            if checkPos.x < MapBorder or checkPos.x >= MapWidth - MapBorder or
               checkPos.y < MapBorder or checkPos.y >= MapHeight - MapBorder or
               not env.isEmpty(checkPos) or env.terrain[checkPos.x][checkPos.y] == Water:
              hasSpace = false
              break
          if not hasSpace:
            break
        
        if hasSpace:
          # Clear terrain around the spawner
          for dx in -1 .. 1:
            for dy in -1 .. 1:
              let clearPos = fallbackPos + ivec2(dx, dy)
              if clearPos.x >= 0 and clearPos.x < MapWidth and 
                 clearPos.y >= 0 and clearPos.y < MapHeight:
                if env.terrain[clearPos.x][clearPos.y] != Water:
                  env.terrain[clearPos.x][clearPos.y] = Empty
          
          env.add(Thing(
            kind: Spawner,
            pos: fallbackPos,
            cooldown: 0,
            homeSpawner: fallbackPos
          ))
          
          let nearbyPositions = env.findEmptyPositionsAround(fallbackPos, 1)
          if nearbyPositions.len > 0:
            let nearestAltar = env.findNearestAltar(fallbackPos)
            env.add(createClippy(nearbyPositions[0], fallbackPos, nearestAltar, r))
          break

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

  for agentId in 0 ..< MapAgents:
    env.updateObservations(agentId)

proc loadMap*(env: Environment, map: string) =
  ## Load a map from a string

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
    let
      kind = parseEnum[ThingKind](parts[0])
      id = parts[1].parseInt
    case kind:
    of Agent:
      env.add(Thing(
        kind: kind,
        id: id,
        agentId: parts[2].parseInt,
        pos: ivec2(parts[3].parseInt, parts[4].parseInt),
        inventoryOre: 0,
        inventoryBattery: 0,
        inventorySpear: 0,
        frozen: 0,
      ))
    of Wall:
      env.add(Thing(
        kind: kind,
        id: id,
        pos: ivec2(parts[2].parseInt, parts[3].parseInt),
      ))
    of Mine:
      env.add(Thing(
        kind: kind,
        id: id,
        pos: ivec2(parts[2].parseInt, parts[3].parseInt),
        resources: MapObjectMineInitialResources,
      ))
    of Converter:
      env.add(Thing(
        kind: kind,
        id: id,
        pos: ivec2(parts[2].parseInt, parts[3].parseInt),
      ))
    of Altar:
      env.add(Thing(
        kind: kind,
        id: id,
        pos: ivec2(parts[2].parseInt, parts[3].parseInt),
        hearts: MapObjectAltarInitialHearts,
      ))
    of Spawner:
      env.add(Thing(
        kind: kind,
        id: id,
        pos: ivec2(parts[2].parseInt, parts[3].parseInt),
      ))
    of Clippy:
      env.add(Thing(
        kind: kind,
        id: id,
        agentId: parts[2].parseInt,
        pos: ivec2(parts[3].parseInt, parts[4].parseInt),
      ))
    of Armory, Forge, ClayOven, WeavingLoom:
      env.add(Thing(
        kind: kind,
        id: id,
        pos: ivec2(parts[2].parseInt, parts[3].parseInt),
      ))

  for agentId in 0 ..< MapAgents:
    env.updateObservations(agentId)

proc dumpMap*(env: Environment): string =
  ## Dump the map
  for thing in env.things:
    if thing.kind == Agent:
      result.add fmt"{thing.kind} {thing.id} {thing.agentId} {thing.pos.x} {thing.pos.y}" & "\n"
    else:
      result.add fmt"{thing.kind} {thing.id} {thing.pos.x} {thing.pos.y}" & "\n"

proc newEnvironment*(): Environment =
  ## Create a new environment
  result = Environment()
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
    of 2: env.rotateAction(id, agent, action[1].int)
    of 3: env.useAction(id, agent, action[1].int)
    of 4: env.attackAction(id, agent, action[1].int)
    of 5: env.getAction(id, agent, action[1].int)  # New get action
    of 6: env.shieldAction(id, agent, action[1].int)
    of 7: env.giftAction(id, agent)
    of 8: env.swapAction(id, agent, action[1].int)
    #of: env.jumpAction(id, agent)
    #of: env.transferAction(id, agent)
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
            let dist = abs(other.pos.x - thing.pos.x) + abs(other.pos.y - thing.pos.y)
            if dist <= 5:  # Within 5 tiles of spawner
              nearbyClippyCount += 1
        
        # Spawn a new Clippy (no limit for now)
        if true:
          # Find empty positions around spawner
          let emptyPositions = env.findEmptyPositionsAround(thing.pos, 2)
          if emptyPositions.len > 0:
            var r = initRand(env.currentStep)
            let spawnPos = r.sample(emptyPositions)
            
            # Create new Clippy with nearest altar as target
            let nearestAltar = env.findNearestAltar(thing.pos)
            let newClippy = createClippy(spawnPos, thing.pos, nearestAltar, r)
            # Don't add immediately - collect for later
            newClippysToSpawn.add(newClippy)
            
            # Reset spawner cooldown
            thing.cooldown = SpawnerCooldown
    elif thing.kind == Agent:
      if thing.frozen > 0:
        thing.frozen -= 1
        # Note: frozen status is visible in observations through updateObservations(id)

  # Add newly spawned clippys from spawners
  for newClippy in newClippysToSpawn:
    env.add(newClippy)
  
  # Update Clippys - they move and interact
  # First collect all clippys to process (to avoid modifying collection while iterating)
  var clippysToProcess: seq[Thing] = @[]
  for thing in env.things:
    if thing.kind == Clippy:
      clippysToProcess.add(thing)
  
  var clippysToRemove: seq[Thing] = @[]
  var r = initRand(env.currentStep)
  
  for clippy in clippysToProcess:
    # Get movement direction from clippy AI (pass things directly)
    let moveDir = getClippyMoveDirection(clippy.pos, env.things, r)
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
  
  # Check for clippy vs agent combat
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
          # Check defense items - armor first (3 uses), then hat (1 use)
          if adjacentThing.inventoryArmor > 0:
            adjacentThing.inventoryArmor -= 1
            env.updateObservations(AgentInventoryArmorLayer, adjacentThing.pos, adjacentThing.inventoryArmor)
            agentSurvived = true
          elif adjacentThing.inventoryHat > 0:
            adjacentThing.inventoryHat = 0
            env.updateObservations(AgentInventoryHatLayer, adjacentThing.pos, adjacentThing.inventoryHat)
            agentSurvived = true
        
        # Only kill agent if they would die and have no defense
        if agentWouldDie and not agentSurvived:
          # Agent dies - mark for respawn at altar
          adjacentThing.frozen = 999999  # Mark as dead (will be respawned)
          env.terminated[adjacentThing.agentId] = 1.0
          
          # Clear the agent from its current position
          env.grid[adjacentThing.pos.x][adjacentThing.pos.y] = nil
          
          # Clear inventory when agent dies
          env.updateObservations(AgentInventoryBatteryLayer, adjacentThing.pos, 0)
          env.updateObservations(AgentInventoryHatLayer, adjacentThing.pos, 0)
          env.updateObservations(AgentInventoryArmorLayer, adjacentThing.pos, 0)
        
        # Break after first combat (clippy is already dead)
        break
  
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
  
  # Update heatmap using batch tint modification system
  # This is much more efficient than updating during each entity move
  env.updateTintModifications()  # Collect all entity contributions
  env.applyTintModifications()   # Apply them to the main color array in one pass

proc reset*(env: Environment) =
  env.currentStep = 0
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

proc getEpisodeStats*(env: Environment): string =
  if env.stats.len == 0:
    return
  template display(name: string, statName) =
    var
      total = 0
      min = int.high
      max = 0
    for stat in env.stats:
      total += stat.statName
      if stat.statName < min:
        min = stat.statName
      if stat.statName > max:
        max = stat.statName
    let avg = total.float32 / env.stats.len.float32
    result.formatValue(name, ">26")
    result.formatValue(total, "10d")
    result.add " "
    result.formatValue(avg, "10.2f")
    result.add " "
    result.formatValue(min, "8d")
    result.add " "
    result.formatValue(max, "8d")
    result.add "\n"

  result = "                      Stat     Total    Average      Min      Max\n"
  display "action.invalid", actionInvalid
  display "action.move", actionMove
  display "action.noop", actionNoop
  display "action.rotate", actionRotate
  display "action.swap", actionSwap
  display "action.use", actionUse
  display "action.use.altar", actionUseAltar
  display "action.use.converter", actionUseConverter
  display "action.use.mine", actionUseMine
  display "action.get", actionGet
  display "action.get.water", actionGetWater
  display "action.get.wheat", actionGetWheat
  display "action.get.wood", actionGetWood

  return result

# ============== GAME STATE MANAGEMENT ==============

type
  GameState* = ref object
    env*: Environment
    selection*: Thing

# Global game state instance
var gameState* = GameState(
  env: nil,
  selection: nil
)

proc initGameState*() =
  ## Initialize the game state with a new environment
  gameState.env = newEnvironment()
  gameState.selection = nil

proc getEnv*(): Environment =
  ## Get the current environment
  if gameState.env == nil:
    initGameState()
  return gameState.env

proc setEnv*(env: Environment) =
  ## Set the current environment
  gameState.env = env

proc getSelection*(): Thing =
  ## Get the currently selected thing
  return gameState.selection

proc setSelection*(thing: Thing) =
  ## Set the currently selected thing
  gameState.selection = thing

proc clearSelection*() =
  ## Clear the current selection
  gameState.selection = nil

# Initialize on module load
initGameState()

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
