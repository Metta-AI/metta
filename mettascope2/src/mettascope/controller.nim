import std/[math, random, tables, sequtils]
import vmath
import tribal

type
  ControllerState* = ref object
    ## State for each agent's controller
    wanderRadius*: int
    wanderAngle*: float
    basePosition*: IVec2
    hasOre*: bool
    hasBattery*: bool
    currentTarget*: IVec2
    targetType*: TargetType
    startingEnergy*: int  # Track agent's starting energy
    
  TargetType* = enum
    NoTarget
    Mine
    Generator
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
    wanderRadius: 3,  # Start with small radius
    wanderAngle: 0.0,
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

proc getNextWanderPoint(controller: Controller, state: ControllerState): IVec2 =
  ## Get next point in expanding spiral pattern
  state.wanderAngle += PI / 4  # 45 degree increments for 8 directions
  if state.wanderAngle >= 2 * PI:
    state.wanderAngle -= 2 * PI
    state.wanderRadius = min(state.wanderRadius + 2, 20)  # Expand radius, max 20
  
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
    controller.initAgentState(agentId, basePos, agent.energy)
  
  var state = controller.agentStates[agentId]
  
  # Update inventory state from agent
  state.hasOre = agent.inventory > 0
  # Consider having a battery only if energy is above starting amount (i.e., gained from converter)
  state.hasBattery = agent.energy > state.startingEnergy
  
  # Find visible things
  let visibleThings = env.findVisibleThings(agent, viewRadius = 5)
  
  # Decision logic based on current inventory
  if state.hasBattery:
    # Priority 1: If we have a battery, go back to altar to deposit it
    if state.targetType != Altar:
      # Find our home altar
      var altar: Thing = nil
      for thing in env.things:
        if thing.kind == Altar and thing.pos == agent.homeAltar:
          altar = thing
          break
      
      if altar != nil:
        state.currentTarget = altar.pos
        state.targetType = Altar
    
    # Check if we're adjacent to the altar
    if state.targetType == Altar and isAdjacent(agent.pos, state.currentTarget):
      # Use the altar to deposit battery
      let dir = state.currentTarget - agent.pos
      let orientation = getOrientation(dir)
      
      # First rotate to face the altar if needed
      if agent.orientation != orientation:
        return [2'u8, orientation.uint8]
      
      # Then use the altar
      return [3'u8, 0'u8]  # Use action
    
  elif state.hasOre:
    # Priority 2: If we have ore, find a generator/converter
    if state.targetType != Generator:
      var nearestGenerator: Thing = nil
      var minDist = 999999.0
      
      for thing in visibleThings:
        if thing.kind == Generator and thing.cooldown == 0:
          let dist = distance(agent.pos, thing.pos)
          if dist < minDist:
            minDist = dist
            nearestGenerator = thing
      
      # If no visible generator, search wider
      if nearestGenerator == nil:
        nearestGenerator = env.findNearestThing(agent.pos, Generator, maxDist = 15.0)
      
      if nearestGenerator != nil:
        state.currentTarget = nearestGenerator.pos
        state.targetType = Generator
    
    # Check if we're adjacent to a generator
    if state.targetType == Generator and isAdjacent(agent.pos, state.currentTarget):
      # Use the generator to convert ore to battery
      let dir = state.currentTarget - agent.pos
      let orientation = getOrientation(dir)
      
      # First rotate to face the generator if needed
      if agent.orientation != orientation:
        return [2'u8, orientation.uint8]
      
      # Then use the generator
      return [3'u8, 0'u8]  # Use action
    
  else:
    # Priority 3: No inventory, look for mines
    if state.targetType != Mine or distance(agent.pos, state.currentTarget) > 10:
      var nearestMine: Thing = nil
      var minDist = 999999.0
      
      for thing in visibleThings:
        if thing.kind == Mine and thing.cooldown == 0:
          let dist = distance(agent.pos, thing.pos)
          if dist < minDist:
            minDist = dist
            nearestMine = thing
      
      if nearestMine != nil:
        state.currentTarget = nearestMine.pos
        state.targetType = Mine
      else:
        # No mine visible, wander in expanding circles
        state.currentTarget = controller.getNextWanderPoint(state)
        state.targetType = Wander
    
    # Check if we're adjacent to a mine
    if state.targetType == Mine and isAdjacent(agent.pos, state.currentTarget):
      # Use the mine to get ore
      let dir = state.currentTarget - agent.pos
      let orientation = getOrientation(dir)
      
      # First rotate to face the mine if needed
      if agent.orientation != orientation:
        return [2'u8, orientation.uint8]
      
      # Then use the mine
      return [3'u8, 0'u8]  # Use action
  
  # Movement logic: Move towards current target
  if state.currentTarget != agent.pos:
    let dir = getDirectionTo(agent.pos, state.currentTarget)
    
    if dir.x != 0 or dir.y != 0:
      let targetOrientation = getOrientation(dir)
      
      # Rotate to face the target direction if needed
      if agent.orientation != targetOrientation:
        return [2'u8, targetOrientation.uint8]  # Rotate action
      
      # Check if we can move forward
      let nextPos = agent.pos + orientationToVec(agent.orientation)
      if env.isEmpty(nextPos):
        return [1'u8, 1'u8]  # Move forward
      else:
        # Obstacle in the way, try to go around
        # Rotate to a random direction
        let newOrientation = Orientation(controller.rng.rand(0..3))
        return [2'u8, newOrientation.uint8]
  
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