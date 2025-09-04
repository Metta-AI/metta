import std/[math, random, tables, sequtils]
import vmath
from tribal_game import Environment, Thing, MapAgents, ObservationLayers, Orientation,
  N, S, W, E, NW, NE, SW, SE, IVec2, MapWidth, MapHeight

type
  ControllerState* = ref object
    ## State for each agent's controller
    wanderRadius*: int  # DEPRECATED - keeping for compatibility
    wanderAngle*: float  # DEPRECATED - keeping for compatibility
    wanderStartAngle*: float  # DEPRECATED - keeping for compatibility
    wanderPointsVisited*: int # DEPRECATED - keeping for compatibility
    # New spiral state
    spiralArcLength*: int  # Current arc length (how many steps to take in current direction)
    spiralStepsInArc*: int  # Steps taken in current arc
    spiralDirection*: int  # Current direction (0=N, 1=E, 2=S, 3=W)
    spiralArcsCompleted*: int  # Number of arcs completed (used to increase arc length)
    # Stuck detection
    lastPosition*: IVec2  # Position from previous step
    stuckCounter*: int  # How many steps we've been in same position
    escapeMode*: bool  # Currently escaping from stuck situation
    escapeStepsRemaining*: int  # Steps left in escape mode
    escapeDirection*: IVec2  # Direction to escape in
    basePosition*: IVec2
    hasOre*: bool
    hasBattery*: bool
    currentTarget*: IVec2
    targetType*: TargetType
    
  TargetType* = enum
    NoTarget
    Mine
    Converter
    Altar
    Wander
    
  Controller* = ref object
    ## Main controller for all agents
    agentStates*: Table[int, ControllerState]
    rng: Rand
    stepCount: int

proc newController*(seed: int = 2024): Controller =
  ## Create a new controller for managing agent behaviors
  result = Controller(
    agentStates: initTable[int, ControllerState](),
    rng: initRand(seed),
    stepCount: 0
  )

proc initAgentState(controller: Controller, agentId: int, basePos: IVec2) =
  ## Initialize state for a new agent
  controller.agentStates[agentId] = ControllerState(
    wanderRadius: 3,  # DEPRECATED
    wanderAngle: 0.0,  # DEPRECATED
    wanderStartAngle: 0.0,  # DEPRECATED
    wanderPointsVisited: 0,  # DEPRECATED
    # New spiral initialization
    spiralArcLength: 1,  # Start with arc length of 1
    spiralStepsInArc: 0,  # No steps taken yet
    spiralDirection: 0,  # Start going North
    spiralArcsCompleted: 0,  # No arcs completed yet
    # Stuck detection initialization
    lastPosition: basePos,
    stuckCounter: 0,
    escapeMode: false,
    escapeStepsRemaining: 0,
    escapeDirection: ivec2(0, 0),
    basePosition: basePos,
    hasOre: false,
    hasBattery: false,
    currentTarget: basePos,
    targetType: NoTarget
  )

proc distance(a, b: IVec2): float =
  ## Calculate Manhattan distance between two points
  result = abs(a.x - b.x).float + abs(a.y - b.y).float

proc distanceEuclidean(a, b: IVec2): float =
  ## Calculate Euclidean distance between two points
  let dx = (a.x - b.x).float
  let dy = (a.y - b.y).float
  result = sqrt(dx * dx + dy * dy)

proc resetWanderState(state: ControllerState) =
  ## Reset wander state when breaking out to pursue a resource
  # Keep the spiral progress to resume expanding search from where we left off
  # Don't reset the arc length or arcs completed - this ensures continuous expansion
  # Just reset the steps in current arc to start fresh from current position
  state.spiralStepsInArc = 0

proc getNextWanderPoint*(controller: Controller, state: ControllerState): IVec2 =
  ## Get next point in expanding spiral pattern using 8 directions
  ## Pattern: Move in increasing arc lengths using all 8 directions for efficient coverage
  ## This creates an outward spiral from the base position
  
  # Track current position in the spiral (accumulated from all steps)
  var totalOffset = ivec2(0, 0)
  var currentArcLength = 1
  var stepsAccumulated = 0
  var direction = 0
  
  # Rebuild the position by simulating all steps up to current point
  for arcNum in 0 ..< state.spiralArcsCompleted:
    # Calculate arc length for this arc number
    let arcLen = (arcNum div 4) + 1  # Slower growth since we have 8 directions now
    let dir = arcNum mod 8  # Direction cycles through 0-7 (8 directions)
    
    # Add the full arc's offset using 8 directions
    case dir:
    of 0: totalOffset.y -= int32(arcLen)  # North
    of 1: 
      totalOffset.x += int32(arcLen)  # NE diagonal
      totalOffset.y -= int32(arcLen)
    of 2: totalOffset.x += int32(arcLen)  # East  
    of 3: 
      totalOffset.x += int32(arcLen)  # SE diagonal
      totalOffset.y += int32(arcLen)
    of 4: totalOffset.y += int32(arcLen)  # South
    of 5: 
      totalOffset.x -= int32(arcLen)  # SW diagonal
      totalOffset.y += int32(arcLen)
    of 6: totalOffset.x -= int32(arcLen)  # West
    of 7: 
      totalOffset.x -= int32(arcLen)  # NW diagonal
      totalOffset.y -= int32(arcLen)
    else: discard
  
  # Add partial progress in current arc
  currentArcLength = (state.spiralArcsCompleted div 4) + 1  # Adjusted for 8 directions
  direction = state.spiralArcsCompleted mod 8
  
  # Add the steps we've taken in the current arc with diagonal support
  case direction:
  of 0: totalOffset.y -= int32(state.spiralStepsInArc)  # North
  of 1: 
    totalOffset.x += int32(state.spiralStepsInArc)  # NE
    totalOffset.y -= int32(state.spiralStepsInArc)
  of 2: totalOffset.x += int32(state.spiralStepsInArc)  # East
  of 3: 
    totalOffset.x += int32(state.spiralStepsInArc)  # SE
    totalOffset.y += int32(state.spiralStepsInArc)
  of 4: totalOffset.y += int32(state.spiralStepsInArc)  # South  
  of 5: 
    totalOffset.x -= int32(state.spiralStepsInArc)  # SW
    totalOffset.y += int32(state.spiralStepsInArc)
  of 6: totalOffset.x -= int32(state.spiralStepsInArc)  # West
  of 7: 
    totalOffset.x -= int32(state.spiralStepsInArc)  # NW
    totalOffset.y -= int32(state.spiralStepsInArc)
  else: discard
  
  # Now calculate next step
  state.spiralStepsInArc += 1
  
  # Check if we completed the current arc
  if state.spiralStepsInArc > currentArcLength:
    # Move to next arc
    state.spiralArcsCompleted += 1
    state.spiralStepsInArc = 1  # Start new arc
    
    # Recalculate for new arc with 8 directions
    currentArcLength = (state.spiralArcsCompleted div 4) + 1  # Adjusted for 8 directions
    direction = state.spiralArcsCompleted mod 8
    
    # Continue expanding with much larger spirals before reset
    # This ensures agents keep searching even in large maps
    if state.spiralArcsCompleted > 120:  # Much larger spirals with 8 directions
      # Only reset if we've been wandering for a very long time
      # Keep significant progress to avoid searching the same area repeatedly
      state.spiralArcsCompleted = controller.rng.rand(40..50)  # Start much further out
      state.spiralStepsInArc = 1
      # Don't return to base, continue from current position to explore new areas
      # This helps agents explore the entire map rather than staying near home
  
  # Calculate next position offset using 8 directions for efficient diagonal movement
  case direction:
  of 0: totalOffset.y -= 1  # Take one step North
  of 1: # Take one step NE (diagonal)
    totalOffset.x += 1
    totalOffset.y -= 1
  of 2: totalOffset.x += 1  # Take one step East
  of 3: # Take one step SE (diagonal)
    totalOffset.x += 1
    totalOffset.y += 1
  of 4: totalOffset.y += 1  # Take one step South
  of 5: # Take one step SW (diagonal)
    totalOffset.x -= 1
    totalOffset.y += 1
  of 6: totalOffset.x -= 1  # Take one step West
  of 7: # Take one step NW (diagonal)
    totalOffset.x -= 1
    totalOffset.y -= 1
  else: discard
  
  result = state.basePosition + totalOffset

proc findNearestThing(env: Environment, pos: IVec2, kind: ThingKind, maxDist: float = 10.0): Thing =
  ## Find the nearest thing of a specific kind within max distance
  result = nil
  var minDist = maxDist
  
  for thing in env.things:
    if thing.kind == kind:
      let dist = distance(pos, thing.pos)
      if dist < minDist:
        minDist = dist
        result = thing

proc findVisibleThings(env: Environment, agent: Thing, viewRadius: int = 5): seq[Thing] =
  ## Find all things visible to the agent within view radius
  result = @[]
  for thing in env.things:
    if thing != agent:
      let dist = distance(agent.pos, thing.pos)
      if dist <= viewRadius.float:
        result.add(thing)

proc getDirectionTo(fromPos, toPos: IVec2): IVec2 =
  ## Get the unit direction vector from one position to another (supports 8 directions)
  let dx = toPos.x - fromPos.x
  let dy = toPos.y - fromPos.y
  
  # Support diagonal movement
  var dirX = 0'i32
  var dirY = 0'i32
  
  if dx > 0:
    dirX = 1
  elif dx < 0:
    dirX = -1
  
  if dy > 0:
    dirY = 1
  elif dy < 0:
    dirY = -1
  
  result = ivec2(dirX, dirY)

proc getOrientation(dir: IVec2): Orientation =
  ## Convert direction vector to orientation (8 directions)
  # Handle diagonal directions first
  if dir.x > 0 and dir.y < 0: return NE
  if dir.x > 0 and dir.y > 0: return SE
  if dir.x < 0 and dir.y < 0: return NW
  if dir.x < 0 and dir.y > 0: return SW
  # Handle cardinal directions
  if dir.x > 0: return E
  if dir.x < 0: return W
  if dir.y > 0: return S
  if dir.y < 0: return N
  return N  # Default

proc isAdjacent(pos1, pos2: IVec2): bool =
  ## Check if two positions are adjacent (including diagonals)
  let dx = abs(pos1.x - pos2.x)
  let dy = abs(pos1.y - pos2.y)
  result = dx <= 1 and dy <= 1 and (dx + dy) > 0

proc isCardinallyAdjacent(pos1, pos2: IVec2): bool =
  ## Check if two positions are adjacent in a cardinal direction (N/S/E/W only)
  let dx = abs(pos1.x - pos2.x)
  let dy = abs(pos1.y - pos2.y)
  result = (dx == 1 and dy == 0) or (dx == 0 and dy == 1)

proc getMoveToCardinalPosition(agentPos, targetPos: IVec2, env: Environment): IVec2 =
  ## If agent is diagonally adjacent to target, return direction to move to cardinal position
  ## Returns ivec2(0, 0) if already cardinally adjacent or not adjacent at all
  let dx = targetPos.x - agentPos.x
  let dy = targetPos.y - agentPos.y
  
  # Check if diagonally adjacent
  if abs(dx) == 1 and abs(dy) == 1:
    # Try to move to cardinal position - prefer horizontal movement first
    let horizontalPos = agentPos + ivec2(dx, 0)
    let verticalPos = agentPos + ivec2(0, dy)
    
    # Check which position is empty and return direction to it
    if env.isEmpty(horizontalPos):
      return ivec2(dx, 0)
    elif env.isEmpty(verticalPos):
      return ivec2(0, dy)
    # If both are blocked, return no movement
    
  return ivec2(0, 0)

proc decideAction*(controller: Controller, env: Environment, agentId: int): array[2, uint8] =
  ## Decide the next action for an agent
  let agent = env.agents[agentId]
  
  # Skip frozen agents
  if agent.frozen > 0:
    return [0'u8, 0'u8]
  
  # Initialize agent state if needed
  if agentId notin controller.agentStates:
    # Use home altar as base, or current position if no home
    let basePos = if agent.homeAltar.x >= 0: agent.homeAltar else: agent.pos
    controller.initAgentState(agentId, basePos)
  
  var state = controller.agentStates[agentId]
  
  # Stuck detection: Check if we haven't moved
  const StuckThreshold = 5  # Consider stuck after 5 steps in same position
  const EscapeSteps = 5  # Escape for 5 steps when stuck
  
  if agent.pos == state.lastPosition:
    state.stuckCounter += 1
    if state.stuckCounter >= StuckThreshold and not state.escapeMode:
      # We're stuck! Enter escape mode
      state.escapeMode = true
      state.escapeStepsRemaining = EscapeSteps
      
      # Choose escape direction (opposite of where we were trying to go)
      if state.currentTarget != agent.pos:
        let targetDir = getDirectionTo(agent.pos, state.currentTarget)
        # Go opposite direction
        state.escapeDirection = ivec2(-targetDir.x, -targetDir.y)
      else:
        # Random escape direction
        let dirs = @[ivec2(0, -1), ivec2(0, 1), ivec2(-1, 0), ivec2(1, 0)]
        state.escapeDirection = controller.rng.sample(dirs)
  else:
    # We moved, reset stuck counter
    state.stuckCounter = 0
  
  # Update last position for next step
  state.lastPosition = agent.pos
  
  # Handle escape mode
  if state.escapeMode:
    if state.escapeStepsRemaining > 0:
      state.escapeStepsRemaining -= 1
      
      # Try to move in escape direction
      let nextPos = agent.pos + state.escapeDirection
      if env.isEmpty(nextPos):
        # Convert direction to orientation-based move argument
        let orient = getOrientation(state.escapeDirection)
        return [1'u8, ord(orient).uint8]  # Move in escape direction
      else:
        # Can't move in escape direction, try perpendicular
        let perpDir = ivec2(-state.escapeDirection.y, state.escapeDirection.x)
        let perpPos = agent.pos + perpDir
        if env.isEmpty(perpPos):
          let orient = getOrientation(perpDir)
          return [1'u8, ord(orient).uint8]
    else:
      # Escape complete, exit escape mode and reset target
      state.escapeMode = false
      state.targetType = NoTarget  # Force re-evaluation of goals
  
  # Update inventory state from agent
  state.hasOre = agent.inventoryOre > 0
  state.hasBattery = agent.inventoryBattery > 0
  
  # Find visible things
  let visibleThings = env.findVisibleThings(agent, viewRadius = 5)
  
  # Decision logic based on current inventory
  if state.hasBattery:
    # Priority 1: If we have a battery, go back to altar to deposit it
    if state.targetType != Altar:
      # Clear any previous target when switching to altar
      state.targetType = NoTarget
      
      # Find our home altar
      var altar: Thing = nil
      for thing in env.things:
        if thing.kind == Altar and thing.pos == agent.homeAltar:
          altar = thing
          break
      
      if altar != nil:
        state.currentTarget = altar.pos
        state.targetType = Altar
    
    # Check if we're near the altar
    if state.targetType == Altar:
      if isCardinallyAdjacent(agent.pos, state.currentTarget):
        # We can use the altar!
        let dir = state.currentTarget - agent.pos
        
        # Determine orientation argument for use action
        let useOrientation = getOrientation(dir)
        let useArg = ord(useOrientation).uint8
        
        # Use the altar
        return [3'u8, useArg]  # Use action with direction
      elif isAdjacent(agent.pos, state.currentTarget):
        # We're diagonally adjacent - move to cardinal position
        let moveDir = getMoveToCardinalPosition(agent.pos, state.currentTarget, env)
        if moveDir.x != 0 or moveDir.y != 0:
          let moveOrientation = getOrientation(moveDir)
          return [1'u8, ord(moveOrientation).uint8]  # Move to cardinal position
    
  elif state.hasOre:
    # Priority 2: If we have ore, find a converter
    # Clear mine target when we get ore to ensure we leave the mine area
    if state.targetType == Mine or state.targetType == Wander:
      state.targetType = NoTarget
      state.currentTarget = agent.pos
    
    if state.targetType != Converter:
      var nearestConverter: Thing = nil
      var minDist = 999999.0
      
      for thing in visibleThings:
        if thing.kind == Converter and thing.cooldown == 0:
          let dist = distance(agent.pos, thing.pos)
          if dist < minDist:
            minDist = dist
            nearestConverter = thing
      
      # If no visible converter, search much wider and keep looking
      if nearestConverter == nil:
        nearestConverter = env.findNearestThing(agent.pos, Converter, maxDist = 50.0)  # Increased from 20
      
      if nearestConverter != nil:
        state.currentTarget = nearestConverter.pos
        state.targetType = Converter
        resetWanderState(state)  # Reset wander when we find a converter
      else:
        # No converter found, wander away from current position to explore
        state.currentTarget = controller.getNextWanderPoint(state)
        state.targetType = Wander
    
    # Check if we're near the converter
    if state.targetType == Converter:
      if isCardinallyAdjacent(agent.pos, state.currentTarget):
        # We can use the converter!
        let dir = state.currentTarget - agent.pos
        
        # Determine orientation argument for use action
        let useOrientation = getOrientation(dir)
        let useArg = ord(useOrientation).uint8
        
        # Use the converter
        return [3'u8, useArg]  # Use action with direction
      elif isAdjacent(agent.pos, state.currentTarget):
        # We're diagonally adjacent - move to cardinal position
        let moveDir = getMoveToCardinalPosition(agent.pos, state.currentTarget, env)
        if moveDir.x != 0 or moveDir.y != 0:
          let moveOrientation = getOrientation(moveDir)
          return [1'u8, ord(moveOrientation).uint8]  # Move to cardinal position
    
  else:
    # Priority 3: No inventory, look for mines
    # Re-evaluate target if we're at a mine that's on cooldown or exhausted
    var needNewTarget = false
    if state.targetType == Mine:
      # Check if current mine target is still valid
      var currentMine: Thing = nil
      for thing in env.things:
        if thing.kind == Mine and thing.pos == state.currentTarget:
          currentMine = thing
          break
      
      # Abandon this mine if it's on cooldown or out of resources
      if currentMine != nil and (currentMine.cooldown > 0 or currentMine.resources == 0):
        needNewTarget = true
        state.targetType = NoTarget
    
    # Look for a new mine if we don't have a valid target - search much wider area
    if state.targetType != Mine or distance(agent.pos, state.currentTarget) > 30 or needNewTarget:  # Increased from 15
      var nearestMine: Thing = nil
      var minDist = 999999.0
      
      for thing in visibleThings:
        if thing.kind == Mine and thing.cooldown == 0 and thing.resources > 0:
          let dist = distance(agent.pos, thing.pos)
          if dist < minDist:
            minDist = dist
            nearestMine = thing
      
      if nearestMine != nil:
        state.currentTarget = nearestMine.pos
        state.targetType = Mine
        resetWanderState(state)  # Reset wander when we find a mine
      else:
        # No active mine visible, wander in expanding circles to find one
        state.currentTarget = controller.getNextWanderPoint(state)
        state.targetType = Wander
    
    # Check if we're near the mine
    if state.targetType == Mine:
      if isCardinallyAdjacent(agent.pos, state.currentTarget):
        # We can use the mine!
        let dir = state.currentTarget - agent.pos
        
        # Determine orientation argument for use action
        let useOrientation = getOrientation(dir)
        let useArg = ord(useOrientation).uint8
        
        # Use the mine
        return [3'u8, useArg]  # Use action with direction
      elif isAdjacent(agent.pos, state.currentTarget):
        # We're diagonally adjacent - move to cardinal position
        let moveDir = getMoveToCardinalPosition(agent.pos, state.currentTarget, env)
        if moveDir.x != 0 or moveDir.y != 0:
          let moveOrientation = getOrientation(moveDir)
          return [1'u8, ord(moveOrientation).uint8]  # Move to cardinal position
  
  # Movement logic: Move towards current target using 8-directional movement
  if state.currentTarget != agent.pos:
    let dir = getDirectionTo(agent.pos, state.currentTarget)
    
    if dir.x != 0 or dir.y != 0:
      # Convert direction vector to orientation
      let moveOrientation = getOrientation(dir)
      
      # Check if we can move in that direction
      let nextPos = agent.pos + dir
      if env.isEmpty(nextPos):
        return [1'u8, ord(moveOrientation).uint8]  # Move in direction with auto-rotation
      else:
        # Obstacle in the way, try alternative directions
        # First try cardinal directions if diagonal was blocked
        if ord(moveOrientation) >= 4:  # Was diagonal
          # Try component directions
          if dir.x != 0:
            let cardinalDir = ivec2(dir.x, 0)
            let cardinalPos = agent.pos + cardinalDir
            if env.isEmpty(cardinalPos):
              let cardinalOrient = getOrientation(cardinalDir)
              return [1'u8, ord(cardinalOrient).uint8]
          
          if dir.y != 0:
            let cardinalDir = ivec2(0, dir.y)
            let cardinalPos = agent.pos + cardinalDir
            if env.isEmpty(cardinalPos):
              let cardinalOrient = getOrientation(cardinalDir)
              return [1'u8, ord(cardinalOrient).uint8]
        
        # Last resort: try a random direction (including diagonals)
        let randomOrientation = Orientation(controller.rng.rand(0..7))
        let randomDelta = case randomOrientation
          of N: ivec2(0, -1)
          of S: ivec2(0, 1)
          of W: ivec2(-1, 0)
          of E: ivec2(1, 0)
          of NW: ivec2(-1, -1)
          of NE: ivec2(1, -1)
          of SW: ivec2(-1, 1)
          of SE: ivec2(1, 1)
        
        let testPos = agent.pos + randomDelta
        if env.isEmpty(testPos):
          return [1'u8, ord(randomOrientation).uint8]  # Move in random direction
  
  # Default: do nothing
  return [0'u8, 0'u8]  # Noop

proc updateController*(controller: Controller) =
  ## Update controller state (called each step)
  controller.stepCount += 1
  
  # Periodically reset wander patterns to prevent getting stuck
  if controller.stepCount mod 100 == 0:
    for state in controller.agentStates.mvalues:
      if state.targetType == Wander:
        state.wanderAngle = controller.rng.rand(0.0 .. 2*PI)