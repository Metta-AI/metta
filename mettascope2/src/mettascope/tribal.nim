import std/[strformat, random, strutils, tables, times], vmath, jsony, chroma
import terrain, placement, clippy, village
export terrain

# Global variables for storing village colors
var agentVillageColors*: seq[Color] = @[]
var altarColors*: Table[IVec2, Color] = initTable[IVec2, Color]()

const
  # From config
  # MapLayoutRoomsX* = 1
  # MapLayoutRoomsY* = 1
  # MapBorder* = 4
  # MapRoomWidth* = 100
  # MapRoomHeight* = 100
  # MapRoomBorder* = 0
  # MapRoomObjectsAgents* = 70
  # MapRoomObjectsAltars* = 50
  # MapRoomObjectsConverters* = 100
  # MapRoomObjectsMines* = 100
  # MapRoomObjectsWalls* = 400

  MapLayoutRoomsX* = 1
  MapLayoutRoomsY* = 1
  MapBorder* = 4
  MapRoomWidth* = 96  # 100 - 4 border = 96
  MapRoomHeight* = 46  # 50 - 4 border = 46
  MapRoomBorder* = 0
  MapRoomObjectsAgents* = 15  # Total agents to spawn (will be distributed across villages)
  MapRoomObjectsHouses* = 3  # Number of villages/houses to spawn
  MapAgentsPerHouse* = 5  # Agents to spawn per house/village
  MapRoomObjectsConverters* = 10  # Converters to process ore into batteries
  MapRoomObjectsMines* = 20  # Mines to extract ore (2x generators)
  MapRoomObjectsWalls* = 30  # Increased for larger map

  MapObjectAgentMaxInventory* = 5
  MapObjectAgentFreezeDuration* = 10  # Temporary freeze when caught by clippy

  MapObjectAltarInitialHearts* = 5  # Altars start with 5 hearts
  MapObjectAltarCooldown* = 10
  MapObjectAltarRespawnCost* = 1  # Cost 1 heart to respawn an agent
  # Altar uses batteries directly now, no energy cost

  MapObjectConverterCooldown* = 0  # No cooldown for instant conversion

  MapObjectMineCooldown* = 5
  MapObjectMineInitialResources* = 30
  MapObjectMineUseCost* = 0

  ObservationLayers* = 16  # Reduced after removing combat/energy systems
  ObservationWidth* = 11
  ObservationHeight* = 11

  # Computed
  MapAgents* = MapRoomObjectsAgents * MapLayoutRoomsX * MapLayoutRoomsY
  MapWidth* = MapLayoutRoomsX * (MapRoomWidth + MapRoomBorder) + MapBorder
  MapHeight* = MapLayoutRoomsY * (MapRoomHeight + MapRoomBorder) + MapBorder

proc ivec2*(x, y: int): IVec2 =
  ## Create a new 2D vector
  result.x = x.int32
  result.y = y.int32

type
  ObservationName* = enum
    AgentLayer = 0
    AgentOrientationLayer = 1
    AgentInventoryOreLayer = 2
    AgentInventoryBatteryLayer = 3
    AgentInventoryWaterLayer = 4
    AgentInventoryWheatLayer = 5
    AgentInventoryWoodLayer = 6
    WallLayer = 7
    MineLayer = 8
    MineResourceLayer = 9
    MineReadyLayer = 10
    ConverterLayer = 11  # Renamed from Converter
    ConverterReadyLayer = 12
    AltarLayer = 13
    AltarHeartsLayer = 14  # Hearts for respawning
    AltarReadyLayer = 15

  Orientation* = enum
    N # Up, Key W
    S # Down, Key S
    W # Right, Key D
    E # Left, Key A

  ThingKind* = enum
    Agent
    Wall
    Mine
    Converter  # Converts ore to batteries
    Altar
    Temple
    Clippy

  Thing* = ref object
    kind*: ThingKind
    pos*: IVec2
    id*: int
    layer*: int
    hearts*: int  # For altars only - used for respawning agents
    resources*: int  # For mines - remaining ore
    cooldown*: int
    frozen*: int  # Frozen duration (for agents caught by clippys)
    inventory*: int  # Generic inventory (ore) - deprecated, use specific inventories

    # Agent:
    agentId*: int
    orientation*: Orientation
    inventoryOre*: int      # Ore from mines
    inventoryBattery*: int  # Batteries from converters
    inventoryWater*: int    # Water from water tiles
    inventoryWheat*: int    # Wheat from wheat tiles
    inventoryWood*: int     # Wood from tree tiles
    reward*: float32
    homeAltar*: IVec2      # Position of agent's home altar for respawning
    
    # Clippy:
    homeTemple*: IVec2     # Position of clippy's home temple
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

  Environment* = ref object
    currentStep*: int
    things*: seq[Thing]
    agents*: seq[Thing]
    grid*: array[MapWidth, array[MapHeight, Thing]]
    terrain*: TerrainGrid
    observations*: array[
      MapAgents,
      array[ObservationLayers,
        array[ObservationWidth, array[ObservationHeight, uint8]]
      ]
    ]
    #rewards*: array[MapAgents, float32]
    terminated *: array[MapAgents, float32]
    truncated*: array[MapAgents, float32]
    stats: seq[Stats]

proc render*(env: Environment): string =
  ## Render the environment as a string
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
          of Temple:
            cell = "t"
          of Clippy:
            cell = "C"
          break
      result.add(cell)
    result.add("\n")

proc renderObservations*(env: Environment): string =
  ## Render the observations as a string
  const featureNames = [
    "agent",
    "agent:orientation",
    "agent:inv:ore",
    "agent:inv:battery",
    "agent:inv:water",
    "agent:inv:wheat",
    "agent:inv:wood",
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
  ## Clear the entire array and set everything to 0.
  let p = cast[pointer](s[0].addr)
  zeroMem(p, s.len * sizeof(T))

proc clear[N: int, T](s: ptr array[N, T]) =
  ## Clear the entire array and set everything to 0.
  let p = cast[pointer](s[][0].addr)
  zeroMem(p, s[].len * sizeof(T))

proc updateObservations(env: Environment, agentId: int) =
  ## Update observations
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

      of Wall:
        # Layer 7: WallLayer
        obs[7][x][y] = 1

      of Mine:
        # Layer 8: MineLayer
        obs[8][x][y] = 1
        # Layer 9: MineResourceLayer
        obs[9][x][y] = thing.resources.uint8
        # Layer 10: MineReadyLayer
        obs[10][x][y] = (thing.cooldown == 0).uint8

      of Converter:
        # Layer 11: ConverterLayer
        obs[11][x][y] = 1
        # Layer 12: ConverterReadyLayer
        obs[12][x][y] = (thing.cooldown == 0).uint8

      of Altar:
        # Layer 13: AltarLayer
        obs[13][x][y] = 1
        # Layer 14: AltarHeartsLayer
        obs[14][x][y] = thing.hearts.uint8
        # Layer 15: AltarReadyLayer
        obs[15][x][y] = (thing.cooldown == 0).uint8
      
      of Temple:
        # Temple acts similar to altar for observations
        obs[13][x][y] = 1
        obs[14][x][y] = thing.hearts.uint8
        obs[15][x][y] = (thing.cooldown == 0).uint8
      
      of Clippy:
        # Clippy acts similar to agent for observations
        obs[0][x][y] = 1
        obs[1][x][y] = 0  # Clippy orientation
        obs[2][x][y] = 0  # Clippys don't carry ore
        obs[3][x][y] = 0  # Clippys don't carry batteries
        obs[4][x][y] = 0  # No water
        obs[5][x][y] = 0  # No wheat
        obs[6][x][y] = 0  # No wood

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
  ## Get the thing at a position
  if pos.x < 0 or pos.x >= MapWidth or pos.y < 0 or pos.y >= MapHeight:
    return nil
  return env.grid[pos.x][pos.y]

proc isEmpty*(env: Environment, pos: IVec2): bool =
  ## Check if a position is empty (water is now passable)
  if pos.x < 0 or pos.x >= MapWidth or pos.y < 0 or pos.y >= MapHeight:
    return false
  # Water is now passable, only check for objects
  return env.grid[pos.x][pos.y] == nil

proc orientationToVec*(orientation: Orientation): IVec2 =
  ## Convert orientation to a vector
  case orientation
  of N: result = ivec2(0, -1)
  of S: result = ivec2(0, 1)
  of E: result = ivec2(1, 0)
  of W: result = ivec2(-1, 0)

proc relativeLocation*(orientation: Orientation, distance, offset: int): IVec2 =
  ## Calculate a relative location based on orientation.
  if orientation == N:
    ivec2(-offset, -distance)
  elif orientation == S:
    ivec2(offset, distance)
  elif orientation == E:
    ivec2(distance, -offset)
  elif orientation == W:
    ivec2(-distance, offset)
  else:
    ivec2(0, 0)

proc noopAction(env: Environment, id: int, agent: Thing) =
  ## Do nothing
  inc env.stats[id].actionNoop

proc moveAction(env: Environment, id: int, agent: Thing, argument: int) =
  ## Move the agent in a cardinal direction and auto-rotate to face that direction
  var newPos = agent.pos
  var newOrientation = agent.orientation
  
  case argument:
  of 0:  # Move North
    newPos.y -= 1
    newOrientation = N
  of 1:  # Move South
    newPos.y += 1
    newOrientation = S
  of 2:  # Move East
    newPos.x += 1
    newOrientation = E
  of 3:  # Move West
    newPos.x -= 1
    newOrientation = W
  else:
    inc env.stats[id].actionInvalid
    return
    
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
    agent.orientation = newOrientation  # Update orientation when moving

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

proc rotateAction(env: Environment, id: int, agent: Thing, argument: int) =
  ## Rotate the agent
  if argument < 0 or argument > 3:
    inc env.stats[id].actionInvalid
    return
  agent.orientation = Orientation(argument)
  env.updateObservations(AgentOrientationLayer, agent.pos, argument)
  inc env.stats[id].actionRotate

proc jumpAction(env: Environment, id: int, agent: Thing) =
  ## Jump the agent
  discard

proc transferAction(env: Environment, id: int, agent: Thing) =
  ## Transfer resources
  discard

proc useAction(env: Environment, id: int, agent: Thing, argument: int) =
  ## Use resources - argument specifies direction (0=N, 1=S, 2=E, 3=W)
  if argument > 3:
    inc env.stats[id].actionInvalid
    return
  
  # Calculate target position based on direction argument
  var usePos = agent.pos
  case argument:
  of 0:  # North
    usePos.y -= 1
  of 1:  # South  
    usePos.y += 1
  of 2:  # East
    usePos.x += 1
  of 3:  # West
    usePos.x -= 1
  else:
    inc env.stats[id].actionInvalid
    return
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
      inc env.stats[id].actionUseConverter
      inc env.stats[id].actionUse
  of Temple, Clippy:
    # Can't use temples or Clippys
    inc env.stats[id].actionInvalid

proc attackAction*(env: Environment, id: int, agent: Thing, argument: int) =
  ## Attack action removed - no longer used
  inc env.stats[id].actionInvalid

proc getAction(env: Environment, id: int, agent: Thing, argument: int) =
  ## Get resources from terrain (water, wheat, wood) - argument specifies direction (0=N, 1=S, 2=E, 3=W)
  if argument > 3:
    inc env.stats[id].actionInvalid
    return
  
  # Calculate target position based on direction argument
  var targetPos = agent.pos
  case argument:
  of 0:  # North
    targetPos.y -= 1
  of 1:  # South
    targetPos.y += 1
  of 2:  # East
    targetPos.x += 1
  of 3:  # West
    targetPos.x -= 1
  else:
    inc env.stats[id].actionInvalid
    return
  
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
      inc env.stats[id].actionGetWood
      inc env.stats[id].actionGet
    else:
      inc env.stats[id].actionInvalid  # Inventory full
  
  of Empty:
    inc env.stats[id].actionInvalid  # Nothing to get

proc shieldAction(env: Environment, id: int, agent: Thing, argument: int) =
  ## Shield action removed - no longer used
  inc env.stats[id].actionInvalid

proc giftAction(env: Environment, id: int, agent: Thing) =
  ## Gift
  discard

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

# proc updateGrid(env: Environment) =
#   ## Update the grid
#   for x in 0 ..< MapWidth:
#     for y in 0 ..< MapHeight:
#       env.grid[x][y] = nil
#   for thing in env.things:
#     env.grid[thing.pos.x][thing.pos.y] = thing

proc findEmptyPositionsAround(env: Environment, center: IVec2, radius: int): seq[IVec2] =
  ## Find empty positions around a center point within a given radius
  result = @[]
  for dx in -radius .. radius:
    for dy in -radius .. radius:
      if dx == 0 and dy == 0:
        continue  # Skip the center position
      let pos = ivec2(center.x + dx, center.y + dy)
      # Check bounds and emptiness
      if pos.x >= MapBorder and pos.x < MapWidth - MapBorder and
         pos.y >= MapBorder and pos.y < MapHeight - MapBorder and
         env.isEmpty(pos) and env.terrain[pos.x][pos.y] != Water:
        result.add(pos)

proc getHouseCorners(env: Environment, houseTopLeft: IVec2, houseSize: int = 5): seq[IVec2] =
  ## Get the 4 corners around a house (just outside the structure)
  result = @[]
  
  # Simple: just the 4 corners
  let corners = @[
    ivec2(houseTopLeft.x - 1, houseTopLeft.y - 1),                    # Top-left
    ivec2(houseTopLeft.x + houseSize, houseTopLeft.y - 1),            # Top-right
    ivec2(houseTopLeft.x - 1, houseTopLeft.y + houseSize),            # Bottom-left
    ivec2(houseTopLeft.x + houseSize, houseTopLeft.y + houseSize)     # Bottom-right
  ]
  
  # Check each corner is valid and empty
  for corner in corners:
    if corner.x >= MapBorder and corner.x < MapWidth - MapBorder and
       corner.y >= MapBorder and corner.y < MapHeight - MapBorder and
       env.isEmpty(corner) and env.terrain[corner.x][corner.y] != Water:
      result.add(corner)

proc randomEmptyPos(r: var Rand, env: Environment): IVec2 =
  ## Find an empty position in the environment (not on water)
  for i in 0 ..< 100:
    let pos = ivec2(r.rand(MapBorder ..< MapWidth - MapBorder), r.rand(MapBorder ..< MapHeight - MapBorder))
    if env.isEmpty(pos) and env.terrain[pos.x][pos.y] != Water:
      result = pos
      return
  # Try harder with more attempts
  for i in 0 ..< 1000:
    let pos = ivec2(r.rand(MapBorder ..< MapWidth - MapBorder), r.rand(MapBorder ..< MapHeight - MapBorder))
    if env.isEmpty(pos) and env.terrain[pos.x][pos.y] != Water:
      result = pos
      return
  quit("Failed to find an empty position, map too full!")

proc add(env: Environment, thing: Thing) =
  ## Add a thing to the environment
  env.things.add(thing)
  if thing.kind == Agent:
    env.agents.add(thing)
    env.stats.add(Stats())
  env.grid[thing.pos.x][thing.pos.y] = thing

proc init(env: Environment) =
  ## Initialize or reset the environment.

  # Use current time for random seed to get different maps each time
  let seed = int(epochTime() * 1000)
  var r = initRand(seed)
  echo "Generating map with seed: ", seed
  
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
  
  for i in 0 ..< numHouses:
    # Use the new unified placement system
    let houseStruct = createHouseStructure()
    var gridPtr = cast[PlacementGrid](env.grid.addr)
    var terrainPtr = env.terrain.addr
    let placementResult = findPlacement(gridPtr, terrainPtr, houseStruct, MapWidth, MapHeight, MapBorder, r, preferCorners = true)
    
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
      
      # Generate a unique color for this village
      let villageColor = color(
        (i.float32 * 137.5 / 360.0) mod 1.0,  # Hue using golden angle
        0.7 + (i.float32 * 0.13).mod(0.3),    # Saturation
        0.5 + (i.float32 * 0.17).mod(0.2),    # Lightness
        1.0
      )
      
      # Spawn agents around this house
      let agentsForThisHouse = min(MapAgentsPerHouse, MapRoomObjectsAgents - totalAgentsSpawned)
      
      # Add the altar with initial hearts
      env.add(Thing(
        kind: Altar,
        pos: elements.center,
        hearts: MapObjectAltarInitialHearts,  # Altar starts with default hearts
      ))
      altarColors[elements.center] = villageColor  # Associate altar position with village color
      
      # Add the walls
      for wallPos in elements.walls:
        env.add(Thing(
          kind: Wall,
          pos: wallPos,
        ))
      if agentsForThisHouse > 0:
        # Get corner positions first, then nearby positions
        let corners = env.getHouseCorners(placementResult.position, houseStruct.width)
        let nearbyPositions = env.findEmptyPositionsAround(elements.center, 3)
        
        for j in 0 ..< agentsForThisHouse:
          var agentPos: IVec2
          if j < corners.len:
            # Prefer corners
            agentPos = corners[j]
          elif j - corners.len < nearbyPositions.len:
            # Then nearby positions
            agentPos = nearbyPositions[j - corners.len]
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
      frozen: 0,
    ))
    
    totalAgentsSpawned += 1

  # Spawn temples with Clippys (same count as houses)
  for i in 0 ..< numHouses:
    let templeStruct = createTempleStructure()
    var gridPtr = cast[PlacementGrid](env.grid.addr)
    var terrainPtr = env.terrain.addr
    let placementResult = findPlacement(gridPtr, terrainPtr, templeStruct, MapWidth, MapHeight, MapBorder, r)
    
    if placementResult.success:  # Valid location found
      let elements = getStructureElements(templeStruct, placementResult.position)
      let templeCenter = elements.center
      
      # Clear terrain within the temple area to create a clearing
      for dy in 0 ..< templeStruct.height:
        for dx in 0 ..< templeStruct.width:
          let clearX = placementResult.position.x + dx
          let clearY = placementResult.position.y + dy
          if clearX >= 0 and clearX < MapWidth and clearY >= 0 and clearY < MapHeight:
            # Clear any terrain features (wheat, trees) but keep water
            if env.terrain[clearX][clearY] != Water:
              env.terrain[clearX][clearY] = Empty
      
      # Add the temple
      env.add(Thing(
        kind: Temple,
        pos: templeCenter,
        cooldown: 0,
      ))
      
      # Spawn initial Clippy near the temple (not on the temple itself)
      # Find an empty position adjacent to the temple
      let nearbyPositions = env.findEmptyPositionsAround(templeCenter, 1)
      if nearbyPositions.len > 0:
        env.add(Thing(
          kind: Clippy,
          pos: nearbyPositions[0],  # Pick first available position near temple
          orientation: Orientation(r.rand(0..3)),
          homeTemple: templeCenter,  # Remember home temple
          wanderRadius: 5,  # Start with medium radius
          wanderAngle: 0.0,
          targetPos: ivec2(-1, -1),  # No target initially
          wanderStepsRemaining: 0,  # Start ready to look for targets
        ))

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
    of Temple:
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
    elif thing.kind == Temple:
      if thing.cooldown > 0:
        thing.cooldown -= 1
      else:
        # Temple is ready to spawn a Clippy
        # Count nearby Clippys
        var nearbyClippyCount = 0
        for other in env.things:
          if other.kind == Clippy:
            let dist = abs(other.pos.x - thing.pos.x) + abs(other.pos.y - thing.pos.y)
            if dist <= 5:  # Within 5 tiles of temple
              nearbyClippyCount += 1
        
        # Spawn a new Clippy if under the max limit
        if nearbyClippyCount < TempleMaxClippys:
          # Find empty positions around temple
          let emptyPositions = env.findEmptyPositionsAround(thing.pos, 2)
          if emptyPositions.len > 0:
            var r = initRand(env.currentStep)
            let spawnPos = r.sample(emptyPositions)
            
            # Create new Clippy
            let newClippy = Thing(
              kind: Clippy,
              pos: spawnPos,
              orientation: Orientation(r.rand(0..3)),
              homeTemple: thing.pos,  # Remember home temple position
              wanderRadius: 5,  # Start with medium radius
              wanderAngle: 0.0,
              targetPos: ivec2(-1, -1),  # No target initially
              wanderStepsRemaining: 0,  # Start ready to look for targets
            )
            # Don't add immediately - collect for later
            newClippysToSpawn.add(newClippy)
            
            # Reset temple cooldown
            thing.cooldown = TempleCooldown
    elif thing.kind == Agent:
      if thing.frozen > 0:
        thing.frozen -= 1
        # Note: frozen status is visible in observations through updateObservations(id)

  # Add newly spawned clippys from temples
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
    # Convert things to seq of pointers for clippy module
    var thingPtrs: seq[pointer] = @[]
    for t in env.things:
      thingPtrs.add(cast[pointer](t))
    
    # Get movement direction from clippy AI
    let moveDir = getClippyMoveDirection(clippy.pos, thingPtrs, r)
    let newPos = clippy.pos + moveDir
    
    # Check if new position is valid and empty
    if env.isEmpty(newPos):
      # Move the clippy
      env.grid[clippy.pos.x][clippy.pos.y] = nil
      clippy.pos = newPos
      env.grid[clippy.pos.x][clippy.pos.y] = clippy
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
        
        # 50% chance agent dies and needs respawning
        if combatRoll < 0.5:
          # Agent dies - mark for respawn at altar
          adjacentThing.frozen = 999999  # Mark as dead (will be respawned)
          env.terminated[adjacentThing.agentId] = 1.0
          
          # Clear the agent from its current position
          env.grid[adjacentThing.pos.x][adjacentThing.pos.y] = nil
          
          # Clear inventory when agent dies
          env.updateObservations(AgentInventoryBatteryLayer, adjacentThing.pos, 0)
        
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
          env.updateObservations(AgentOrientationLayer, agent.pos, agent.orientation.int)
          # Shield layer removed\n        # env.updateObservations(AgentShieldLayer, agent.pos, agent.shield.int)
          env.updateObservations(agentId)

  # for agentId in 0 ..< MapAgents:
  #   env.updateObservations(agentId)

proc reset*(env: Environment) =
  ## Reset the environment
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

proc getEpisodeStats*(env: Environment): string =
  ## Get the episode stats
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
