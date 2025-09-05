## Common types, constants, and utilities shared across tribal modules
## This module centralizes core definitions to avoid circular imports and duplication

import std/[strformat], vmath, windy

# =============================================================================
# CORE CONSTANTS - Map Configuration
# =============================================================================

const
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
  
  # Object Configuration
  MapObjectAgentMaxInventory* = 5
  MapObjectAgentFreezeDuration* = 10  # Temporary freeze when caught by clippy
  MapObjectAltarInitialHearts* = 5  # Altars start with 5 hearts
  MapObjectAltarCooldown* = 10
  MapObjectAltarRespawnCost* = 1  # Cost 1 heart to respawn an agent
  MapObjectConverterCooldown* = 0  # No cooldown for instant conversion
  MapObjectMineCooldown* = 5
  MapObjectMineInitialResources* = 30
  MapObjectMineUseCost* = 0

  # Observation Configuration
  ObservationLayers* = 17  # Includes spear inventory layer
  ObservationWidth* = 11
  ObservationHeight* = 11

  # Computed Map Dimensions
  MapAgents* = MapRoomObjectsAgents * MapLayoutRoomsX * MapLayoutRoomsY
  MapWidth* = MapLayoutRoomsX * (MapRoomWidth + MapRoomBorder) + MapBorder
  MapHeight* = MapLayoutRoomsY * (MapRoomHeight + MapRoomBorder) + MapBorder
  
  # UI Constants
  WindowWidth* = 1280
  WindowHeight* = 800

# Movement key mappings for 8-way movement
type
  MovementKey* = tuple[primary, secondary: Button, direction: uint8]

const MovementKeys*: array[8, MovementKey] = [
  (primary: KeyW, secondary: KeyUp, direction: 0'u8),    # North
  (primary: KeyS, secondary: KeyDown, direction: 1'u8),  # South  
  (primary: KeyD, secondary: KeyRight, direction: 2'u8), # East
  (primary: KeyA, secondary: KeyLeft, direction: 3'u8),  # West
  (primary: KeyQ, secondary: KeyHome, direction: 4'u8),  # Northwest
  (primary: KeyE, secondary: KeyPageUp, direction: 5'u8), # Northeast
  (primary: KeyZ, secondary: KeyEnd, direction: 6'u8),   # Southwest
  (primary: KeyC, secondary: KeyPageDown, direction: 7'u8) # Southeast
]

# =============================================================================
# CORE ENUMS
# =============================================================================

type
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
    Temple
    Clippy
    Armory
    Forge
    ClayOven
    WeavingLoom

  ObservationName* = enum
    AgentLayer = 0
    AgentOrientationLayer = 1
    AgentInventoryOreLayer = 2
    AgentInventoryBatteryLayer = 3
    AgentInventoryWaterLayer = 4
    AgentInventoryWheatLayer = 5
    AgentInventoryWoodLayer = 6
    AgentInventorySpearLayer = 7
    WallLayer = 8
    MineLayer = 9
    MineResourceLayer = 10
    MineReadyLayer = 11
    ConverterLayer = 12  # Renamed from Converter
    ConverterReadyLayer = 13
    AltarLayer = 14
    AltarHeartsLayer = 15  # Hearts for respawning
    AltarReadyLayer = 16

  TerrainType* = enum
    Empty
    Water
    Wheat
    Tree

# =============================================================================
# CORE TYPES
# =============================================================================

type
  IRect* = object
    x*, y*, w*, h*: int
  
  TerrainGrid* = array[MapWidth, array[MapHeight, TerrainType]]
  
  OrientationDelta* = tuple[x, y: int]

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
    inventorySpear*: int    # Spears crafted from forge
    inventoryBread*: int    # Bread from clay ovens
    hunger*: int           # Steps since last meal (hunger system)
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
    terminated*: array[MapAgents, float32]
    truncated*: array[MapAgents, float32]
    stats*: seq[Stats]

# =============================================================================
# ORIENTATION UTILITIES
# =============================================================================

const OrientationDeltas*: array[8, OrientationDelta] = [
  (x: 0, y: -1),   # N (North)
  (x: 0, y: 1),    # S (South)
  (x: -1, y: 0),   # W (West)
  (x: 1, y: 0),    # E (East)
  (x: -1, y: -1),  # NW (Northwest)
  (x: 1, y: -1),   # NE (Northeast)
  (x: -1, y: 1),   # SW (Southwest)
  (x: 1, y: 1),    # SE (Southeast)
]

proc getOrientationDelta*(orient: Orientation): OrientationDelta =
  return OrientationDeltas[ord(orient)]

proc isDiagonal*(orient: Orientation): bool =
  return ord(orient) > 3  # Orientations 4-7 are diagonal

proc getOpposite*(orient: Orientation): Orientation =
  case orient:
  of N: S
  of S: N
  of W: E
  of E: W
  of NW: SE
  of NE: SW
  of SW: NE
  of SE: NW

proc orientationToVec*(orientation: Orientation): IVec2 =
  let delta = getOrientationDelta(orientation)
  return ivec2(delta.x.int32, delta.y.int32)

# =============================================================================
# VECTOR UTILITIES
# =============================================================================

proc ivec2*(x, y: int): IVec2 =
  ## Create a new 2D vector
  result.x = x.int32
  result.y = y.int32

proc toIVec2*(x, y: int): IVec2 =
  ## Alias for ivec2 for compatibility
  return ivec2(x, y)

# =============================================================================
# DISTANCE UTILITIES
# =============================================================================

proc manhattanDistance*(a, b: IVec2): int =
  ## Calculate Manhattan distance between two points
  return abs(a.x - b.x) + abs(a.y - b.y)

proc getManhattanDistance*(pos1, pos2: IVec2): int =
  ## Alias for manhattanDistance for compatibility
  return manhattanDistance(pos1, pos2)

proc distance*(a, b: IVec2): float =
  ## Calculate Euclidean distance between two points
  let dx = (a.x - b.x).float
  let dy = (a.y - b.y).float
  return sqrt(dx * dx + dy * dy)

proc distanceEuclidean*(a, b: IVec2): float =
  ## Alias for distance for compatibility
  return distance(a, b)

# =============================================================================
# POSITION UTILITIES
# =============================================================================

proc isAdjacent*(pos1, pos2: IVec2): bool =
  ## Check if two positions are adjacent (including diagonally)
  let dx = abs(pos1.x - pos2.x)
  let dy = abs(pos1.y - pos2.y)
  return dx <= 1 and dy <= 1 and (dx + dy) > 0

proc isCardinallyAdjacent*(pos1, pos2: IVec2): bool =
  ## Check if two positions are cardinally adjacent (not diagonal)
  let dx = abs(pos1.x - pos2.x)
  let dy = abs(pos1.y - pos2.y)
  return (dx == 1 and dy == 0) or (dx == 0 and dy == 1)

proc getDirectionTo*(fromPos, toPos: IVec2): IVec2 =
  ## Get direction vector from one position to another
  var dx = toPos.x - fromPos.x
  var dy = toPos.y - fromPos.y
  
  # Normalize to -1, 0, or 1
  if dx > 0: dx = 1
  elif dx < 0: dx = -1
  
  if dy > 0: dy = 1
  elif dy < 0: dy = -1
  
  return ivec2(dx, dy)

proc getDirectionToward*(fromPos, toPos: IVec2): IVec2 =
  ## Alias for getDirectionTo for compatibility
  return getDirectionTo(fromPos, toPos)

proc getOrientation*(dir: IVec2): Orientation =
  ## Convert direction vector to orientation
  if dir.x == 0 and dir.y == -1: return N
  elif dir.x == 0 and dir.y == 1: return S
  elif dir.x == -1 and dir.y == 0: return W
  elif dir.x == 1 and dir.y == 0: return E
  elif dir.x == -1 and dir.y == -1: return NW
  elif dir.x == 1 and dir.y == -1: return NE
  elif dir.x == -1 and dir.y == 1: return SW
  elif dir.x == 1 and dir.y == 1: return SE
  else: return N  # Default to North for invalid directions

proc relativeLocation*(orientation: Orientation, distance, offset: int): IVec2 =
  ## Calculate position relative to orientation
  let delta = getOrientationDelta(orientation)
  let perpDelta = getOrientationDelta(case orientation:
    of N: E
    of S: W
    of W: N
    of E: S
    of NW: NE
    of NE: SE
    of SW: NW
    of SE: SW
  )
  
  return ivec2(
    delta.x * distance + perpDelta.x * offset,
    delta.y * distance + perpDelta.y * offset
  )