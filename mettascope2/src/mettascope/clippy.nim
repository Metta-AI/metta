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

proc manhattanDistance*(a, b: IVec2): int =
  ## Calculate Manhattan distance between two points
  return abs(a.x - b.x) + abs(a.y - b.y)

proc findNearbyClippies*(clippyPos: IVec2, things: seq[pointer], visionRange: int): seq[IVec2] =
  ## Find positions of nearby Clippies within vision range
  result = @[]
  for thingPtr in things:
    if isNil(thingPtr):
      continue
    let thing = cast[ptr tuple[kind: int, pos: IVec2]](thingPtr)
    # Clippy is the 7th enum value (0-indexed), so value is 6
    if thing.kind == 6:  # Clippy kind
      let dist = manhattanDistance(clippyPos, thing.pos)
      if dist > 0 and dist <= visionRange:  # Don't include self (dist > 0)
        result.add(thing.pos)

proc findNearestAltar*(clippyPos: IVec2, things: seq[pointer], visionRange: int): IVec2 =
  ## Find the nearest altar within vision range
  ## Returns ivec2(-1, -1) if no altar found
  var nearestPos = ivec2(-1, -1)
  var minDist = int.high
  
  for thingPtr in things:
    if isNil(thingPtr):
      continue
    let thing = cast[ptr tuple[kind: int, pos: IVec2]](thingPtr)
    # Altar is the 5th enum value (0-indexed), so value is 4
    if thing.kind == 4:  # Altar kind
      let dist = manhattanDistance(clippyPos, thing.pos)
      if dist <= visionRange and dist < minDist:
        minDist = dist
        nearestPos = thing.pos
  
  return nearestPos

proc getAvoidanceDirection*(clippyPos: IVec2, clippyPositions: seq[IVec2]): IVec2 =
  ## Calculate direction to move away from other Clippies
  if clippyPositions.len == 0:
    return ivec2(0, 0)
  
  # Calculate average position of nearby Clippies
  var avgX, avgY: float32 = 0.0
  for pos in clippyPositions:
    avgX += pos.x.float32
    avgY += pos.y.float32
  avgX = avgX / clippyPositions.len.float32
  avgY = avgY / clippyPositions.len.float32
  
  # Move away from average position
  let dx = clippyPos.x.float32 - avgX
  let dy = clippyPos.y.float32 - avgY
  
  # Convert to unit direction
  if abs(dx) > abs(dy):
    if dx > 0: return ivec2(1, 0)
    else: return ivec2(-1, 0)
  else:
    if dy > 0: return ivec2(0, 1)
    else: return ivec2(0, -1)

proc getDirectionToward*(fromPos, toPos: IVec2): IVec2 =
  ## Calculate unit direction from one position toward another
  let dx = toPos.x - fromPos.x
  let dy = toPos.y - fromPos.y
  
  if dx == 0 and dy == 0:
    return ivec2(0, 0)
  
  # Move in the direction with larger difference
  if abs(dx) > abs(dy):
    if dx > 0: return ivec2(1, 0)
    else: return ivec2(-1, 0)
  else:
    if dy > 0: return ivec2(0, 1)
    else: return ivec2(0, -1)

proc getClippyMoveDirection*(clippyPos: IVec2, things: seq[pointer], r: var Rand): IVec2 =
  ## Determine which direction a Clippy should move
  ## Priority: 1) Move toward altar if seen, 2) Avoid other Clippies, 3) Random walk
  
  # Look for nearest altar
  let nearestAltar = findNearestAltar(clippyPos, things, ClippyVisionRange)
  if nearestAltar.x >= 0:  # Valid altar found
    return getDirectionToward(clippyPos, nearestAltar)
  
  # Look for nearby Clippies to avoid
  let nearbyClippies = findNearbyClippies(clippyPos, things, ClippyVisionRange)
  if nearbyClippies.len > 0:
    return getAvoidanceDirection(clippyPos, nearbyClippies)
  
  # Random walk
  let directions = @[ivec2(0, -1), ivec2(0, 1), ivec2(-1, 0), ivec2(1, 0)]
  return r.sample(directions)