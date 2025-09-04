import vmath, std/random, terrain

type
  TempleStructure* = object
    width*: int
    height*: int
    centerPos*: IVec2  # Position of the temple center
  
  ClippyBehavior* = enum
    Patrol      # Wander around looking for targets
    Chase       # Actively pursuing a player
    Guard       # Protecting the temple
    Attack      # Engaging with player

const
  # Clippy agent properties
  ClippyMaxEnergy* = 20
  ClippyInitialEnergy* = 15
  ClippyAttackDamage* = 2
  ClippySpeed* = 1
  ClippyVisionRange* = 5
  ClippyHp* = 3
  
  # Temple properties
  TempleHp* = 50
  TempleCooldown* = 20  # Time between Clippy spawns
  TempleMaxClippys* = 3  # Max Clippys per temple

proc createTemple*(): TempleStructure =
  ## Create a temple structure (3x3 with center as spawn point)
  result.width = 3
  result.height = 3
  result.centerPos = ivec2(1, 1)

proc canPlaceTemple*(grid: ptr array[100, array[50, pointer]], terrain: ptr TerrainGrid,
                     pos: IVec2, temple: TempleStructure, 
                     mapWidth, mapHeight: int): bool =
  ## Check if a temple can be placed at the given position
  ## pos is the top-left corner of the temple
  
  # Check boundaries
  if pos.x < 0 or pos.y < 0:
    return false
  if pos.x + temple.width > mapWidth or pos.y + temple.height > mapHeight:
    return false
  
  # Check if all positions are empty (nil in grid) and not on water
  for y in 0 ..< temple.height:
    for x in 0 ..< temple.width:
      let gridX = pos.x + x
      let gridY = pos.y + y
      # Check if there's already something at this position
      if not isNil(grid[gridX][gridY]):
        return false
      # Check if this position is water
      if terrain[gridX][gridY] == Water:
        return false
  
  # Add extra spacing from other structures
  # Check for a buffer zone around the temple
  let buffer = 2
  for y in -buffer .. temple.height + buffer - 1:
    for x in -buffer .. temple.width + buffer - 1:
      let checkX = pos.x + x
      let checkY = pos.y + y
      if checkX >= 0 and checkX < mapWidth and checkY >= 0 and checkY < mapHeight:
        # Only check the buffer area, not the temple area itself
        if x < 0 or x >= temple.width or y < 0 or y >= temple.height:
          if not isNil(grid[checkX][checkY]):
            return false  # Something too close
  
  return true

proc findTempleLocation*(grid: ptr array[100, array[50, pointer]], terrain: ptr TerrainGrid,
                         temple: TempleStructure,
                         mapWidth, mapHeight, mapBorder: int, r: var Rand): IVec2 =
  ## Find a suitable location for a temple
  ## Returns ivec2(-1, -1) if no location found
  
  # Try random positions
  for attempt in 0 ..< 100:
    let x = r.rand(mapBorder + 5 ..< mapWidth - mapBorder - temple.width - 5)
    let y = r.rand(mapBorder + 5 ..< mapHeight - mapBorder - temple.height - 5)
    let pos = ivec2(x.int32, y.int32)
    
    if canPlaceTemple(grid, terrain, pos, temple, mapWidth, mapHeight):
      return pos
  
  # If random attempts fail, do a systematic search
  for y in mapBorder + 5 ..< mapHeight - mapBorder - temple.height - 5:
    for x in mapBorder + 5 ..< mapWidth - mapBorder - temple.width - 5:
      let pos = ivec2(x.int32, y.int32)
      if canPlaceTemple(grid, terrain, pos, temple, mapWidth, mapHeight):
        return pos
  
  return ivec2(-1, -1)  # No valid location found

proc getTempleCenter*(temple: TempleStructure, topLeft: IVec2): IVec2 =
  ## Get the world position of the temple's center (spawn point)
  return topLeft + temple.centerPos

proc shouldSpawnClippy*(templeCooldown: int, nearbyClippyCount: int): bool =
  ## Determine if a temple should spawn a new Clippy
  return templeCooldown == 0 and nearbyClippyCount < TempleMaxClippys

proc getClippyBehavior*(clippy: pointer, target: pointer, distanceToTarget: float): ClippyBehavior =
  ## Determine Clippy's current behavior based on game state
  if isNil(target):
    return Patrol
  elif distanceToTarget <= 1.5:
    return Attack
  elif distanceToTarget <= ClippyVisionRange.float:
    return Chase
  else:
    return Guard