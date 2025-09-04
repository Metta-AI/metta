import std/[math, random, tables, sequtils]
import vmath
import tribal

type
  ControllerState* = ref object
    ## State for each agent's controller
    wanderRadius*: int
    wanderAngle*: float
    wanderStartAngle*: float  # Track where we started this circle
    wanderPointsVisited*: int # Count points visited in current circle
    basePosition*: IVec2
    hasOre*: bool
    hasBattery*: bool
    currentTarget*: IVec2
    targetType*: TargetType
    startingEnergy*: int  # Track agent's starting energy
    
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

proc initAgentState(controller: Controller, agentId: int, basePos: IVec2, startingEnergy: int) =
  ## Initialize state for a new agent
  controller.agentStates[agentId] = ControllerState(
    wanderRadius: 5,  # Start with radius 5
    wanderAngle: 0.0,
    wanderStartAngle: 0.0,
    wanderPointsVisited: 0,
    basePosition: basePos,
    hasOre: false,
    hasBattery: false,
    currentTarget: basePos,
    targetType: NoTarget,
    startingEnergy: startingEnergy
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
  state.wanderPointsVisited = 0
  state.wanderStartAngle = state.wanderAngle
  # Keep the current radius so we continue from where we left off

proc getNextWanderPoint(controller: Controller, state: ControllerState): IVec2 =
  ## Get next point in expanding circle pattern with proper tracking
  const pointsPerCircle = 8  # 8 points for a complete circle (45 degree increments)
  
  # Move to next point
  state.wanderAngle += PI / 4  # 45 degree increments for 8 directions
  state.wanderPointsVisited += 1
  
  # Check if we've completed a full circle
  if state.wanderPointsVisited >= pointsPerCircle:
    # We've completed a circle, expand radius
    state.wanderRadius += 1
    state.wanderPointsVisited = 0
    state.wanderStartAngle = state.wanderAngle  # Reset start angle
    # No max limit on radius - it can grow as needed
  
  # Keep angle in 0-2PI range
  if state.wanderAngle >= 2 * PI:
    state.wanderAngle -= 2 * PI
  
  let x = state.basePosition.x + int(cos(state.wanderAngle) * state.wanderRadius.float)
  let y = state.basePosition.y + int(sin(state.wanderAngle) * state.wanderRadius.float)
  result = ivec2(x, y)

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
  ## Get the unit direction vector from one position to another
  let dx = toPos.x - fromPos.x
  let dy = toPos.y - fromPos.y
  
  if abs(dx) > abs(dy):
    result = ivec2(if dx > 0: 1 else: -1, 0)
  elif dy != 0:
    result = ivec2(0, if dy > 0: 1 else: -1)
  else:
    result = ivec2(0, 0)

proc getOrientation(dir: IVec2): Orientation =
  ## Convert direction vector to orientation
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
    controller.initAgentState(agentId, basePos, 0)  # energy no longer used
  
  var state = controller.agentStates[agentId]
  
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
        
        # Determine direction argument for use action
        var useArg: uint8
        if dir.x > 0:
          useArg = 2  # East
        elif dir.x < 0:
          useArg = 3  # West
        elif dir.y > 0:
          useArg = 1  # South
        else:  # dir.y < 0
          useArg = 0  # North
        
        # Use the altar
        return [3'u8, useArg]  # Use action with direction
      elif isAdjacent(agent.pos, state.currentTarget):
        # We're diagonally adjacent - move to cardinal position
        let moveDir = getMoveToCardinalPosition(agent.pos, state.currentTarget, env)
        if moveDir.x != 0 or moveDir.y != 0:
          var moveArg: uint8
          if moveDir.x > 0:
            moveArg = 2  # Move East
          elif moveDir.x < 0:
            moveArg = 3  # Move West
          elif moveDir.y > 0:
            moveArg = 1  # Move South
          else:  # moveDir.y < 0
            moveArg = 0  # Move North
          return [1'u8, moveArg]  # Move to cardinal position
    
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
      
      # If no visible converter, search wider and wander to find one
      if nearestConverter == nil:
        nearestConverter = env.findNearestThing(agent.pos, Converter, maxDist = 20.0)
      
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
        
        # Determine direction argument for use action
        var useArg: uint8
        if dir.x > 0:
          useArg = 2  # East
        elif dir.x < 0:
          useArg = 3  # West
        elif dir.y > 0:
          useArg = 1  # South
        else:  # dir.y < 0
          useArg = 0  # North
        
        # Use the converter
        return [3'u8, useArg]  # Use action with direction
      elif isAdjacent(agent.pos, state.currentTarget):
        # We're diagonally adjacent - move to cardinal position
        let moveDir = getMoveToCardinalPosition(agent.pos, state.currentTarget, env)
        if moveDir.x != 0 or moveDir.y != 0:
          var moveArg: uint8
          if moveDir.x > 0:
            moveArg = 2  # Move East
          elif moveDir.x < 0:
            moveArg = 3  # Move West
          elif moveDir.y > 0:
            moveArg = 1  # Move South
          else:  # moveDir.y < 0
            moveArg = 0  # Move North
          return [1'u8, moveArg]  # Move to cardinal position
    
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
    if state.targetType != Mine or distance(agent.pos, state.currentTarget) > 15 or needNewTarget:
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
        
        # Determine direction argument for use action
        var useArg: uint8
        if dir.x > 0:
          useArg = 2  # East
        elif dir.x < 0:
          useArg = 3  # West
        elif dir.y > 0:
          useArg = 1  # South
        else:  # dir.y < 0
          useArg = 0  # North
        
        # Use the mine
        return [3'u8, useArg]  # Use action with direction
      elif isAdjacent(agent.pos, state.currentTarget):
        # We're diagonally adjacent - move to cardinal position
        let moveDir = getMoveToCardinalPosition(agent.pos, state.currentTarget, env)
        if moveDir.x != 0 or moveDir.y != 0:
          var moveArg: uint8
          if moveDir.x > 0:
            moveArg = 2  # Move East
          elif moveDir.x < 0:
            moveArg = 3  # Move West
          elif moveDir.y > 0:
            moveArg = 1  # Move South
          else:  # moveDir.y < 0
            moveArg = 0  # Move North
          return [1'u8, moveArg]  # Move to cardinal position
  
  # Movement logic: Move towards current target using new directional movement
  if state.currentTarget != agent.pos:
    let dir = getDirectionTo(agent.pos, state.currentTarget)
    
    if dir.x != 0 or dir.y != 0:
      # Determine which direction to move
      var moveArg: uint8
      if dir.x > 0:
        moveArg = 2  # Move East
      elif dir.x < 0:
        moveArg = 3  # Move West
      elif dir.y > 0:
        moveArg = 1  # Move South
      else:  # dir.y < 0
        moveArg = 0  # Move North
      
      # Check if we can move in that direction
      let nextPos = agent.pos + dir
      if env.isEmpty(nextPos):
        return [1'u8, moveArg]  # Move in direction with auto-rotation
      else:
        # Obstacle in the way, try a random direction
        let randomDir = controller.rng.rand(0..3)
        # Check if that random direction is free
        var testPos = agent.pos
        case randomDir:
        of 0: testPos.y -= 1  # North
        of 1: testPos.y += 1  # South
        of 2: testPos.x += 1  # East
        of 3: testPos.x -= 1  # West
        else: discard
        
        if env.isEmpty(testPos):
          return [1'u8, randomDir.uint8]  # Move in random direction
  
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