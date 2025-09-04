import vmath, std/[random, math], terrain

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
  ClippyAttackDamage* = 2
  ClippySpeed* = 1
  ClippyVisionRange* = 12  # Increased to see much further for altar discovery
  ClippyWanderPriority* = 3  # How many wander steps before checking for targets
  ClippyAltarSearchRange* = 10  # Extended range specifically for altar searching
  
  # Temple properties
  TempleCooldown* = 20  # Time between Clippy spawns
  TempleMaxClippys* = 3  # Max Clippys per temple

proc createTemple*(): TempleStructure =
  ## Create a temple structure (3x3 with center as spawn point)
  result.width = 3
  result.height = 3
  result.centerPos = ivec2(1, 1)

# Temple placement logic moved to placement.nim for unified handling

# Temple location finding moved to placement.nim's findPlacement function

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

proc findNearestAltar*(clippyPos: IVec2, things: seq[pointer], searchRange: int): IVec2 =
  ## Find the nearest altar within search range
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
      if dist <= searchRange and dist < minDist:
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

proc findNearestAgent*(clippyPos: IVec2, things: seq[pointer], visionRange: int): IVec2 =
  ## Find the nearest agent within vision range
  ## Returns ivec2(-1, -1) if no agent found
  var nearestPos = ivec2(-1, -1)
  var minDist = int.high
  
  for thingPtr in things:
    if isNil(thingPtr):
      continue
    let thing = cast[ptr tuple[kind: int, pos: IVec2]](thingPtr)
    # Agent is the 1st enum value (0-indexed), so value is 0
    if thing.kind == 0:  # Agent kind
      let dist = manhattanDistance(clippyPos, thing.pos)
      if dist <= visionRange and dist < minDist:
        minDist = dist
        nearestPos = thing.pos
  
  return nearestPos

proc getConcentricWanderPoint*(clippy: pointer, r: var Rand): IVec2 =
  ## Get next point in expanding spiral pattern for better exploration coverage
  let clippyThing = cast[ptr tuple[
    kind: int, pos: IVec2, id: int, layer: int, hearts: int, 
    resources: int, cooldown: int, frozen: int, inventory: int,
    agentId: int, orientation: int,
    inventoryOre: int, inventoryBattery: int, inventoryWater: int,
    inventoryWheat: int, inventoryWood: int, reward: float32,
    homeAltar: IVec2, homeTemple: IVec2, wanderRadius: int,
    wanderAngle: float, targetPos: IVec2, wanderStepsRemaining: int
  ]](clippy)
  
  # Use clippy ID to create different exploration sectors for different clippys
  let sectorOffset = (clippyThing.id mod 4).float * 1.5708  # Divide into 4 sectors
  
  # Increment angle to move around the spiral with sector offset
  clippyThing.wanderAngle += 0.5236  # ~30 degrees for finer exploration pattern
  
  # Complete rotation, expand radius more aggressively
  if clippyThing.wanderAngle >= 6.28:  # 2*PI
    clippyThing.wanderAngle = 0.0
    clippyThing.wanderRadius = min(clippyThing.wanderRadius + 8, 80)  # Very aggressive expansion to radius 80
  
  # Calculate target position in the spiral with sector offset
  let adjustedAngle = clippyThing.wanderAngle + sectorOffset
  let x = clippyThing.homeTemple.x + int32(cos(adjustedAngle) * clippyThing.wanderRadius.float)
  let y = clippyThing.homeTemple.y + int32(sin(adjustedAngle) * clippyThing.wanderRadius.float)
  
  # Add some randomness for less predictable movement
  let jitterX = int32(r.rand(-1..1))
  let jitterY = int32(r.rand(-1..1))
  
  return ivec2(x + jitterX, y + jitterY)

proc getClippyMoveDirection*(clippyPos: IVec2, things: seq[pointer], r: var Rand): IVec2 =
  ## Determine which direction a Clippy should move
  ## Priority: 1) Chase agent if seen, 2) Move toward altar if seen, 
  ## 3) Wander in concentric circles around home temple
  
  # First, find the clippy in the things list to access its state
  var clippyPtr: pointer = nil
  for thingPtr in things:
    if isNil(thingPtr):
      continue
    let thing = cast[ptr tuple[kind: int, pos: IVec2]](thingPtr)
    if thing.kind == 6 and thing.pos == clippyPos:  # Found our clippy
      clippyPtr = thingPtr
      break
  
  if isNil(clippyPtr):
    # Fallback to random if we can't find ourselves
    let directions = @[ivec2(0, -1), ivec2(0, 1), ivec2(-1, 0), ivec2(1, 0)]
    return r.sample(directions)
  
  let clippyThing = cast[ptr tuple[
    kind: int, pos: IVec2, id: int, layer: int, hearts: int, 
    resources: int, cooldown: int, frozen: int, inventory: int,
    agentId: int, orientation: int,
    inventoryOre: int, inventoryBattery: int, inventoryWater: int,
    inventoryWheat: int, inventoryWood: int, reward: float32,
    homeAltar: IVec2, homeTemple: IVec2, wanderRadius: int,
    wanderAngle: float, targetPos: IVec2, wanderStepsRemaining: int
  ]](clippyPtr)
  
  # Check if we should continue wandering despite seeing targets
  if clippyThing.wanderStepsRemaining > 0:
    # Continue wandering, decrement counter
    clippyThing.wanderStepsRemaining -= 1
  else:
    # Ready to check for targets
    # Priority 1: Look for nearest altar to attack (prioritize altar destruction)
    let nearestAltar = findNearestAltar(clippyPos, things, ClippyAltarSearchRange)
    if nearestAltar.x >= 0:  # Valid altar found
      let dist = abs(nearestAltar.x - clippyPos.x) + abs(nearestAltar.y - clippyPos.y)
      # Move toward altar if within extended search range
      if dist <= ClippyAltarSearchRange:
        clippyThing.targetPos = nearestAltar
        return getDirectionToward(clippyPos, nearestAltar)
    
    # Priority 2: Look for nearest agent to chase (only if no altar found)
    let nearestAgent = findNearestAgent(clippyPos, things, ClippyVisionRange)
    if nearestAgent.x >= 0:  # Valid agent found
      let dist = abs(nearestAgent.x - clippyPos.x) + abs(nearestAgent.y - clippyPos.y)
      # Only chase if close enough (within 8 tiles)
      if dist <= 8:
        clippyThing.targetPos = nearestAgent
        return getDirectionToward(clippyPos, nearestAgent)
    
    # If we reach here, no close targets - set reduced wander steps for more frequent checks
    clippyThing.wanderStepsRemaining = r.rand(2..5)  # Reduced wander time for more frequent target checking
  
  # Priority 3: No targets - wander in concentric circles
  # If we have a valid home temple, wander around it
  if clippyThing.homeTemple.x >= 0 and clippyThing.homeTemple.y >= 0:
    let wanderTarget = getConcentricWanderPoint(clippyPtr, r)
    clippyThing.targetPos = wanderTarget
    return getDirectionToward(clippyPos, wanderTarget)
  
  # Fallback: Random walk if no home temple
  let directions = @[ivec2(0, -1), ivec2(0, 1), ivec2(-1, 0), ivec2(1, 0)]
  return r.sample(directions)