import tribal_game
## Core environment types and utilities
## This module contains the fundamental types and basic utilities used throughout the system

import vmath, 
export terrain

# Map configuration constants
const
  MapLayoutRoomsX* = 1
  MapLayoutRoomsY* = 1
  MapBorder* = 4
  MapRoomWidth* = 96  # 100 - 4 border = 96
  MapRoomHeight* = 46  # 50 - 4 border = 46
  MapRoomBorder* = 0
  MapRoomObjectsAgents* = 20  # Total agents to spawn
  MapRoomObjectsHouses* = 4  # Number of villages/houses to spawn (one per corner)
  MapAgentsPerHouse* = 5  # Agents to spawn per house/village
  MapRoomObjectsConverters* = 10  # Converters to process ore into batteries
  MapRoomObjectsMines* = 20  # Mines to extract ore
  MapRoomObjectsWalls* = 30  # Random walls
  
  # Agent properties
  MapObjectAgentMaxInventory* = 5
  MapObjectAgentFreezeDuration* = 10  # Temporary freeze when caught by clippy
  
  # Altar properties
  MapObjectAltarInitialHearts* = 5  # Altars start with 5 hearts
  MapObjectAltarCooldown* = 10
  MapObjectAltarRespawnCost* = 1  # Cost 1 heart to respawn an agent
  
  # Converter properties
  MapObjectConverterCooldown* = 0  # No cooldown for instant conversion
  
  # Mine properties
  MapObjectMineCooldown* = 5
  MapObjectMineInitialResources* = 30
  MapObjectMineUseCost* = 0
  
  # Observation configuration
  ObservationLayers* = 17
  ObservationWidth* = 11
  ObservationHeight* = 11
  
  # Computed dimensions
  MapAgents* = MapRoomObjectsAgents * MapLayoutRoomsX * MapLayoutRoomsY
  MapWidth* = MapLayoutRoomsX * (MapRoomWidth + MapRoomBorder) + MapBorder
  MapHeight* = MapLayoutRoomsY * (MapRoomHeight + MapRoomBorder) + MapBorder

# Core types
type
  OrientationDelta* = tuple[x, y: int]
  
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
  
  Thing* = ref object
    kind*: ThingKind
    pos*: IVec2
    id*: int
    layer*: int
    hearts*: int  # For altars only - used for respawning agents
    resources*: int  # For mines - remaining ore
    cooldown*: int
    frozen*: int  # Frozen duration (for agents caught by clippys)
    inventory*: int  # Generic inventory (ore) - deprecated
    
    # Agent-specific fields
    agentId*: int
    orientation*: Orientation
    inventoryOre*: int      # Ore from mines
    inventoryBattery*: int  # Batteries from converters
    inventoryWater*: int    # Water from water tiles
    inventoryWheat*: int    # Wheat from wheat tiles
    inventoryWood*: int     # Wood from tree tiles
    inventorySpear*: int    # Spears crafted from forge
    reward*: float32
    homeAltar*: IVec2      # Position of agent's home altar for respawning
    
    # Clippy-specific fields
    homeTemple*: IVec2     # Position of clippy's home temple
    wanderRadius*: int     # Current radius for concentric circle wandering
    wanderAngle*: float    # Current angle in the circle pattern
    targetPos*: IVec2      # Current target (agent or altar)
    wanderStepsRemaining*: int  # Steps to wander before checking for targets

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
  
  Stats* = ref object
    # Agent action statistics
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

# Orientation delta lookup table
const OrientationDeltas*: array[8, OrientationDelta] = [
  (0, -1),   # N
  (0, 1),    # S
  (-1, 0),   # W
  (1, 0),    # E
  (-1, -1),  # NW
  (1, -1),   # NE
  (-1, 1),   # SW
  (1, 1)     # SE
]

# Basic utility functions
proc ivec2*(x, y: int): IVec2 =
  ## Create a new 2D integer vector
  result.x = x.int32
  result.y = y.int32

proc getOrientationDelta*(orient: Orientation): OrientationDelta =
  ## Get the movement delta for an orientation
  return OrientationDeltas[ord(orient)]

proc isDiagonal*(orient: Orientation): bool =
  ## Check if an orientation is diagonal
  return ord(orient) > 3

proc getOpposite*(orient: Orientation): Orientation =
  ## Get the opposite orientation
  case orient:
  of N: return S
  of S: return N
  of E: return W
  of W: return E
  of NW: return SE
  of NE: return SW
  of SW: return NE
  of SE: return NW

proc orientationToVec*(orientation: Orientation): IVec2 =
  ## Convert orientation to a vector
  let delta = getOrientationDelta(orientation)
  return ivec2(delta.x, delta.y)

proc relativeLocation*(orientation: Orientation, distance, offset: int): IVec2 =
  ## Calculate a relative location based on orientation
  case orientation:
  of N: ivec2(-offset, -distance)
  of S: ivec2(offset, distance)
  of E: ivec2(distance, -offset)
  of W: ivec2(-distance, offset)
  of NW: ivec2(-distance - offset, -distance + offset)
  of NE: ivec2(distance - offset, -distance - offset)
  of SW: ivec2(-distance + offset, distance + offset)
  of SE: ivec2(distance + offset, distance - offset)

proc getThing*(env: Environment, pos: IVec2): Thing =
  ## Get the thing at a position
  if pos.x < 0 or pos.x >= MapWidth or pos.y < 0 or pos.y >= MapHeight:
    return nil
  return env.grid[pos.x][pos.y]

proc isEmpty*(env: Environment, pos: IVec2): bool =
  ## Check if a position is empty
  if pos.x < 0 or pos.x >= MapWidth or pos.y < 0 or pos.y >= MapHeight:
    return false
  return env.grid[pos.x][pos.y] == nil

proc add*(env: Environment, thing: Thing) =
  ## Add a thing to the environment
  env.things.add(thing)
  if thing.kind == Agent:
    env.agents.add(thing)
    env.stats.add(Stats())
  env.grid[thing.pos.x][thing.pos.y] = thing