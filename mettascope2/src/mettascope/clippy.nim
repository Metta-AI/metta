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
  ClippyVisionRange* = 15  # Even further vision for plague-wave expansion
  ClippyWanderPriority* = 3  # How many wander steps before checking for targets
  ClippyAltarSearchRange* = 12  # Extended range for aggressive altar hunting
  ClippyAgentChaseRange* = 10  # Will chase agents within this range
  
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

proc getOutwardExpansionPoint*(clippy: pointer, things: seq[pointer], r: var Rand): IVec2 =
  ## Get next point for plague-wave expansion - always moving away from origin and other clippys
  let clippyThing = cast[ptr tuple[
    kind: int, pos: IVec2, id: int, layer: int, hearts: int, 
    resources: int, cooldown: int, frozen: int, inventory: int,
    agentId: int, orientation: int,
    inventoryOre: int, inventoryBattery: int, inventoryWater: int,
    inventoryWheat: int, inventoryWood: int, reward: float32,
    homeAltar: IVec2, homeTemple: IVec2, wanderRadius: int,
    wanderAngle: float, targetPos: IVec2, wanderStepsRemaining: int
  ]](clippy)
  
  # Calculate primary expansion direction - away from home temple
  let currentPos = clippyThing.pos
  let homeTemple = clippyThing.homeTemple
  
  # Vector away from home temple (the plague source)
  var awayFromHome = ivec2(0, 0)
  if homeTemple.x >= 0 and homeTemple.y >= 0:
    let dx = currentPos.x - homeTemple.x
    let dy = currentPos.y - homeTemple.y
    let distFromHome = sqrt((dx * dx + dy * dy).float)
    
    # Normalize and scale the "away" vector with stronger bias at close range
    if distFromHome > 0:
      # Stronger push when closer to home, weaker when far
      let pushStrength = max(5.0, 20.0 - distFromHome * 0.3)
      awayFromHome.x = int32((dx.float / distFromHome) * pushStrength)
      awayFromHome.y = int32((dy.float / distFromHome) * pushStrength)
  
  # Calculate avoidance vector from nearby clippys (creates spreading effect)
  var avoidanceVector = ivec2(0, 0)
  var nearbyClippyCount = 0
  
  for thingPtr in things:
    if isNil(thingPtr):
      continue
    let thing = cast[ptr tuple[kind: int, pos: IVec2]](thingPtr)
    if thing.kind == 6 and thing.pos != currentPos:  # Other clippy
      let dist = manhattanDistance(currentPos, thing.pos)
      if dist <= 8:  # Avoid other clippys within 8 tiles
        # Calculate repulsion force (stronger when closer)
        let repulsionStrength = (9 - dist).float / 8.0
        let dx = currentPos.x - thing.pos.x
        let dy = currentPos.y - thing.pos.y
        avoidanceVector.x += int32(dx.float * repulsionStrength)
        avoidanceVector.y += int32(dy.float * repulsionStrength)
        nearbyClippyCount += 1
  
  # Normalize avoidance vector if there are nearby clippys
  if nearbyClippyCount > 0:
    avoidanceVector.x = avoidanceVector.x div int32(nearbyClippyCount)
    avoidanceVector.y = avoidanceVector.y div int32(nearbyClippyCount)
  
  # Add exploration randomness for frontier discovery
  let explorationAngle = r.rand(0.0 .. 2*PI)
  let explorationX = int32(cos(explorationAngle) * 2.0)
  let explorationY = int32(sin(explorationAngle) * 2.0)
  
  # Combine all vectors with weights
  var targetOffset = ivec2(
    awayFromHome.x * 2 + avoidanceVector.x + explorationX,  # Strong outward bias
    awayFromHome.y * 2 + avoidanceVector.y + explorationY
  )
  
  # Ensure we're always moving outward (minimum distance from home)
  let minDistFromHome = clippyThing.wanderRadius
  if minDistFromHome < 100:  # Keep expanding up to 100 tiles
    clippyThing.wanderRadius += 2  # Gradually increase minimum exploration distance
  
  # If the combined vector is too small, use a random outward direction
  if abs(targetOffset.x) + abs(targetOffset.y) < 2:
    let angle = r.rand(0.0 .. 2*PI)
    targetOffset = ivec2(
      int32(cos(angle) * 5.0),
      int32(sin(angle) * 5.0)
    )
  
  # Convert to next step position (not far-away target)
  # We want single-step movement in the desired direction
  var nextStep = currentPos
  
  # Move one step in the strongest direction component
  if abs(targetOffset.x) > abs(targetOffset.y):
    nextStep.x += if targetOffset.x > 0: 1 else: -1
  elif abs(targetOffset.y) > 0:
    nextStep.y += if targetOffset.y > 0: 1 else: -1
  else:
    # If both are zero, pick a random outward direction
    let angle = r.rand(0.0 .. 2*PI)
    nextStep.x += int32(cos(angle))
    nextStep.y += int32(sin(angle))
  
  return nextStep

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
      # Chase agents aggressively within extended range
      if dist <= ClippyAgentChaseRange:
        clippyThing.targetPos = nearestAgent
        return getDirectionToward(clippyPos, nearestAgent)
    
    # If we reach here, no close targets - set reduced wander steps for more frequent checks
    clippyThing.wanderStepsRemaining = r.rand(2..5)  # Reduced wander time for more frequent target checking
  
  # Priority 3: No targets - expand outward like a plague wave
  # Move away from home and other clippys to explore new territory
  if clippyThing.homeTemple.x >= 0 and clippyThing.homeTemple.y >= 0:
    let wanderTarget = getOutwardExpansionPoint(clippyPtr, things, r)
    clippyThing.targetPos = wanderTarget
    return getDirectionToward(clippyPos, wanderTarget)
  
  # Fallback: Random walk if no home temple
  let directions = @[ivec2(0, -1), ivec2(0, 1), ivec2(-1, 0), ivec2(1, 0)]
  return r.sample(directions)