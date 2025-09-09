import std/[random, tables]
import vmath
import environment, common


type
  ControllerState* = ref object
    spiralArcLength*: int
    spiralStepsInArc*: int
    spiralDirection*: int
    spiralArcsCompleted*: int
    lastPosition*: IVec2
    stuckCounter*: int
    escapeMode*: bool
    escapeStepsRemaining*: int
    escapeDirection*: IVec2
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
    agentStates*: Table[int, ControllerState]
    rng: Rand
    stepCount: int

proc newController*(seed: int = 2024): Controller =
  result = Controller(
    agentStates: initTable[int, ControllerState](),
    rng: initRand(seed),
    stepCount: 0
  )

proc initAgentState(controller: Controller, agentId: int, basePos: IVec2) =
  controller.agentStates[agentId] = ControllerState(
    spiralArcLength: 1,
    spiralStepsInArc: 0,
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



proc applyDirectionOffset(offset: var IVec2, direction: int, distance: int32) =
  case direction:
  of 0: offset.y -= distance
  of 1: offset.x += distance
  of 2: offset.y += distance
  of 3: offset.x -= distance
  else: discard

proc resetWanderState(state: ControllerState) =
  state.spiralStepsInArc = 0

proc getNextWanderPoint*(controller: Controller, state: ControllerState): IVec2 =
  
  # Track current position in the spiral (accumulated from all steps)
  var totalOffset = ivec2(0, 0)
  var currentArcLength = 1
  var direction = 0
  
  # Rebuild the position by simulating all steps up to current point
  for arcNum in 0 ..< state.spiralArcsCompleted:
    # Calculate arc length for this arc number
    let arcLen = (arcNum div 2) + 1  # 1,1,2,2,3,3,4,4...
    let dir = arcNum mod 4  # Direction cycles through 0,1,2,3
    
    # Add the full arc's offset
    applyDirectionOffset(totalOffset, dir, int32(arcLen))
  
  # Add partial progress in current arc
  currentArcLength = (state.spiralArcsCompleted div 2) + 1
  direction = state.spiralArcsCompleted mod 4
  
  # Add the steps we've taken in the current arc
  applyDirectionOffset(totalOffset, direction, int32(state.spiralStepsInArc))
  
  # Now calculate next step
  state.spiralStepsInArc += 1
  
  # Check if we completed the current arc
  if state.spiralStepsInArc > currentArcLength:
    # Move to next arc
    state.spiralArcsCompleted += 1
    state.spiralStepsInArc = 1  # Start new arc
    
    # Recalculate for new arc
    currentArcLength = (state.spiralArcsCompleted div 2) + 1
    direction = state.spiralArcsCompleted mod 4
    
    # Cap the spiral size (reset after ~30 arcs which gives us 15 radius)
    if state.spiralArcsCompleted > 30:
      # Reset the spiral but with some randomization
      state.spiralArcsCompleted = controller.rng.rand(0..3)  # Start at random direction
      state.spiralStepsInArc = 1
      return state.basePosition  # Return to base to start new spiral
  
  # Calculate next position offset
  applyDirectionOffset(totalOffset, direction, 1)
  
  result = state.basePosition + totalOffset

proc findNearestThing(env: Environment, pos: IVec2, kind: ThingKind, maxDist: int = 10): Thing =
  result = nil
  var minDist = maxDist
  
  for thing in env.things:
    if thing.kind == kind:
      let dist = manhattanDistance(pos, thing.pos)
      if dist < minDist:
        minDist = dist
        result = thing

proc findVisibleThings(env: Environment, agent: Thing, viewRadius: int = 5): seq[Thing] =
  result = @[]
  for thing in env.things:
    if thing != agent:
      let dist = manhattanDistance(agent.pos, thing.pos)
      if dist <= viewRadius:
        result.add(thing)

proc getMoveToCardinalPosition(agentPos, targetPos: IVec2, env: Environment): IVec2 =
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


proc getOrientation(dir: IVec2): Orientation =
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
  let dx = abs(pos1.x - pos2.x)
  let dy = abs(pos1.y - pos2.y)
  result = dx <= 1 and dy <= 1 and (dx + dy) > 0

proc isCardinallyAdjacent(pos1, pos2: IVec2): bool =
  let dx = abs(pos1.x - pos2.x)
  let dy = abs(pos1.y - pos2.y)
  result = (dx == 1 and dy == 0) or (dx == 0 and dy == 1)

proc decideAction*(controller: Controller, env: Environment, agentId: int): array[2, uint8] =
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
        
        # Determine orientation argument for put action (action 5)
        let useOrientation = getOrientation(dir)
        let useArg = ord(useOrientation).uint8
        
        # Put battery into altar
        return [5'u8, useArg]  # Put action with direction
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
          let dist = manhattanDistance(agent.pos, thing.pos).float
          if dist < minDist:
            minDist = dist
            nearestConverter = thing
      
      # If no visible converter, search wider and wander to find one
      if nearestConverter == nil:
        nearestConverter = env.findNearestThing(agent.pos, Converter, maxDist = 20)
      
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
        # We can get battery from converter (put ore in, get battery out)
        let dir = state.currentTarget - agent.pos
        
        # Determine orientation argument for get action (action 3)
        let useOrientation = getOrientation(dir)
        let useArg = ord(useOrientation).uint8
        
        # Get battery from converter (converts ore to battery)
        return [3'u8, useArg]  # Get action with direction
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
    
    # Look for a new mine if we don't have a valid target
    if state.targetType != Mine or manhattanDistance(agent.pos, state.currentTarget) > 15 or needNewTarget:
      var nearestMine: Thing = nil
      var minDist = 999999.0
      
      for thing in visibleThings:
        if thing.kind == Mine and thing.cooldown == 0 and thing.resources > 0:
          let dist = manhattanDistance(agent.pos, thing.pos).float
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
        # We can get ore from mine
        let dir = state.currentTarget - agent.pos
        
        # Determine orientation argument for get action (action 3)
        let useOrientation = getOrientation(dir)
        let useArg = ord(useOrientation).uint8
        
        # Get ore from mine
        return [3'u8, useArg]  # Get action with direction
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
  controller.stepCount += 1