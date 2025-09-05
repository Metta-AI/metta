import vmath, std/[random, math]
import objects
export objects  # Re-export the types

# Import the constants from objects
const
  PI = 3.14159265358979323846

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

proc getOutwardExpansionDirection*(clippy: pointer, things: seq[pointer], r: var Rand): IVec2 =
  ## Get movement direction for plague-wave expansion - returns a unit direction vector
  let clippyThing = cast[ptr tuple[
    kind: int, pos: IVec2, id: int, layer: int, hearts: int, 
    resources: int, cooldown: int, frozen: int, inventory: int,
    agentId: int, orientation: int,
    inventoryOre: int, inventoryBattery: int, inventoryWater: int,
    inventoryWheat: int, inventoryWood: int, reward: float32,
    homeAltar: IVec2, homeSpawner: IVec2, wanderRadius: int,
    wanderAngle: float, targetPos: IVec2, wanderStepsRemaining: int
  ]](clippy)
  
  let currentPos = clippyThing.pos
  let homeSpawner = clippyThing.homeSpawner
  
  # Primary force: Move away from home spawner
  var outwardForce = vec2(0.0, 0.0)
  if homeSpawner.x >= 0 and homeSpawner.y >= 0:
    let dx = (currentPos.x - homeSpawner.x).float
    let dy = (currentPos.y - homeSpawner.y).float
    let distFromHome = sqrt(dx * dx + dy * dy)
    
    if distFromHome > 0.1:
      # Strong outward push, especially when close
      let pushStrength = if distFromHome < 10: 0.8
                        elif distFromHome < 30: 0.6
                        else: 0.4
      outwardForce.x = (dx / distFromHome) * pushStrength
      outwardForce.y = (dy / distFromHome) * pushStrength
    else:
      # At origin, pick random direction
      let angle = r.rand(0.0 .. 2*PI)
      outwardForce.x = cos(angle) * 0.8
      outwardForce.y = sin(angle) * 0.8
  
  # Secondary force: Avoid nearby clippys
  var avoidanceForce = vec2(0.0, 0.0)
  var nearbyCount = 0
  
  for thingPtr in things:
    if isNil(thingPtr):
      continue
    let thing = cast[ptr tuple[kind: int, pos: IVec2]](thingPtr)
    if thing.kind == 6 and thing.pos != currentPos:  # Other clippy
      let dist = manhattanDistance(currentPos, thing.pos)
      if dist <= 6:  # Repel from nearby clippys
        let dx = (currentPos.x - thing.pos.x).float
        let dy = (currentPos.y - thing.pos.y).float
        let strength = (7 - dist).float / 6.0 * 0.3  # Weaker than outward force
        if abs(dx) + abs(dy) > 0:
          avoidanceForce.x += dx * strength
          avoidanceForce.y += dy * strength
          nearbyCount += 1
  
  if nearbyCount > 0:
    avoidanceForce.x = avoidanceForce.x / nearbyCount.float
    avoidanceForce.y = avoidanceForce.y / nearbyCount.float
  
  # Tertiary force: Random exploration
  let exploreAngle = r.rand(0.0 .. 2*PI)
  let exploreForce = vec2(cos(exploreAngle) * 0.2, sin(exploreAngle) * 0.2)
  
  # Combine all forces
  let totalForce = vec2(
    outwardForce.x + avoidanceForce.x + exploreForce.x,
    outwardForce.y + avoidanceForce.y + exploreForce.y
  )
  
  # Convert to unit direction
  if abs(totalForce.x) > abs(totalForce.y):
    return ivec2(if totalForce.x > 0: 1 else: -1, 0)
  elif abs(totalForce.y) > 0.1:
    return ivec2(0, if totalForce.y > 0: 1 else: -1)
  else:
    # Fallback: move away from home
    if abs(currentPos.x - homeSpawner.x) > abs(currentPos.y - homeSpawner.y):
      return ivec2(if currentPos.x > homeSpawner.x: 1 else: -1, 0)
    else:
      return ivec2(0, if currentPos.y > homeSpawner.y: 1 else: -1)

proc getClippyMoveDirection*(clippyPos: IVec2, things: seq[pointer], r: var Rand): IVec2 =
  ## Simplified Clippy movement: Always move toward the closest altar
  
  # Find the closest altar
  var nearestAltar = ivec2(-1, -1)
  var minDist = int.high
  
  for thingPtr in things:
    if isNil(thingPtr):
      continue
    let thing = cast[ptr tuple[kind: int, pos: IVec2]](thingPtr)
    # Altar is the 5th enum value (0-indexed), so value is 4
    if thing.kind == 4:  # Altar kind
      let dist = manhattanDistance(clippyPos, thing.pos)
      if dist < minDist:
        minDist = dist
        nearestAltar = thing.pos
  
  # If we found an altar, move toward it
  if nearestAltar.x >= 0:
    return getDirectionToward(clippyPos, nearestAltar)
  
  # Fallback: Random walk if no altar found (shouldn't happen in normal gameplay)
  let directions = @[ivec2(0, -1), ivec2(0, 1), ivec2(-1, 0), ivec2(1, 0)]
  return r.sample(directions)