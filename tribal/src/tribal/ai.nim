import std/[random, tables, math]
import vmath
import environment, common


type
  AgentRole* = enum
    AltarSpecialist    # Default role - handles ore→battery→hearts loop
    ArmorySpecialist   # Gathers wood, crafts armor at Armory
    ForgeSpecialist    # Gathers wood, crafts spear at Forge
    ClayOvenSpecialist # Gathers wheat, crafts food at ClayOven
    WeavingLoomSpecialist # Gathers wheat, crafts lantern at WeavingLoom
  
  ControllerState* = ref object
    spiralArcLength*: int
    spiralStepsInArc*: int
    spiralDirection*: int
    spiralArcsCompleted*: int
    lastPosition*: IVec2
    recentPositions*: seq[IVec2]  # Track last few positions to detect oscillation
    stuckCounter*: int
    escapeMode*: bool
    escapeStepsRemaining*: int
    escapeDirection*: IVec2
    basePosition*: IVec2
    hasOre*: bool
    hasBattery*: bool
    currentTarget*: IVec2
    targetType*: TargetType
    role*: AgentRole  # Assigned specialization role
    hasCompletedRole*: bool  # Whether agent has completed their specialized task
    
  TargetType* = enum
    NoTarget
    Mine
    Converter
    Altar
    Wander
    # New target types for specialized buildings
    Armory
    Forge
    ClayOven
    WeavingLoom
    # Terrain resource gathering
    Tree   # For wood gathering
    Wheat  # For wheat gathering
    # Lantern placement
    LanternPlantSpot  # For lantern placement
    # Clippy hunting
    ClippyHunt  # Hunting clippys with spear
    
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

proc assignAgentRole(controller: Controller, agentId: int, env: Environment): AgentRole =
  ## Assign specialized roles to agents based on their position in their village team
  let agent = env.agents[agentId]
  
  # Find which village this agent belongs to by checking home altar
  if agent.homeAltar.x < 0:
    return AltarSpecialist  # Default role for agents without home
  
  # Find agent's position within their team by counting agents with same home altar before this one
  var agentIndexInTeam = 0
  
  for i in 0..<agentId:
    if i < env.agents.len and env.agents[i].homeAltar == agent.homeAltar:
      agentIndexInTeam += 1
  
  # Assign role based on position in team (0-4 for each village)
  case agentIndexInTeam mod 5:
  of 0: AltarSpecialist     # Agent 0: Always on default altar loop
  of 1: ArmorySpecialist    # Agent 1: Wood → Armor
  of 2: ForgeSpecialist     # Agent 2: Wood → Spear  
  of 3: ClayOvenSpecialist  # Agent 3: Wheat → Food
  of 4: WeavingLoomSpecialist # Agent 4: Wheat → Lantern
  else: AltarSpecialist     # Fallback

proc initAgentState(controller: Controller, agentId: int, basePos: IVec2, env: Environment) =
  let assignedRole = controller.assignAgentRole(agentId, env)
  
  controller.agentStates[agentId] = ControllerState(
    spiralStepsInArc: 0,
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
    targetType: NoTarget,
    role: assignedRole,
    hasCompletedRole: false
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

proc findNearestTerrainResource(env: Environment, pos: IVec2, terrainType: TerrainType, maxDist: int = 15): IVec2 =
  ## Find nearest terrain tile of specified type (Water, Wheat, Tree, Empty)
  result = ivec2(-1, -1)
  var minDist = maxDist + 1
  
  for x in 0..<MapWidth:
    for y in 0..<MapHeight:
      if env.terrain[x][y] == terrainType:
        let dist = manhattanDistance(pos, ivec2(x.int32, y.int32))
        if dist < minDist:
          minDist = dist
          result = ivec2(x.int32, y.int32)

proc findHomeBuilding(env: Environment, agent: Thing, buildingKind: ThingKind): Thing =
  ## Find a building of the specified type near the agent's home altar
  result = nil
  var minDist = 999999
  
  for thing in env.things:
    if thing.kind == buildingKind:
      # Check if this building is near the agent's home altar (same village)
      let distFromHome = manhattanDistance(thing.pos, agent.homeAltar)
      if distFromHome <= 10:  # Buildings should be close to home altar
        let distFromAgent = manhattanDistance(thing.pos, agent.pos)
        if distFromAgent < minDist:
          minDist = distFromAgent
          result = thing

proc hasNearbyLanterns(env: Environment, pos: IVec2, radius: int): bool =
  ## Check if there are any lanterns within radius tiles (like clippy logic)
  for dx in -radius .. radius:
    for dy in -radius .. radius:
      if dx == 0 and dy == 0:
        continue  # Skip center position
      
      let checkPos = pos + ivec2(dx, dy)
      if checkPos.x >= 0 and checkPos.x < MapWidth and checkPos.y >= 0 and checkPos.y < MapHeight:
        for thing in env.things:
          if thing.kind == PlantedLantern and thing.pos == checkPos:
            return true
  
  return false

proc checkAndTransitionAfterCollection(state: ControllerState, agent: Thing, env: Environment) =
  ## Check if agent has resources and transition to next phase if needed
  ## This is called at the START of decision making, not during action execution
  case state.role:
  of ArmorySpecialist:
    if agent.inventoryWood > 0 and state.targetType == Tree:
      # Have wood, transition to Armory crafting
      let armory = env.findHomeBuilding(agent, Armory)
      if armory != nil:
        state.currentTarget = armory.pos
        state.targetType = Armory
  of ForgeSpecialist:
    if agent.inventoryWood > 0 and state.targetType == Tree:
      # Have wood, transition to Forge crafting
      let forge = env.findHomeBuilding(agent, Forge)
      if forge != nil:
        state.currentTarget = forge.pos
        state.targetType = Forge
  of ClayOvenSpecialist:
    if agent.inventoryWheat > 0 and state.targetType == Wheat:
      # Have wheat, transition to ClayOven crafting
      let oven = env.findHomeBuilding(agent, ClayOven)
      if oven != nil:
        state.currentTarget = oven.pos
        state.targetType = ClayOven
  of WeavingLoomSpecialist:
    if agent.inventoryWheat > 0 and state.targetType == Wheat:
      # Have wheat, transition to WeavingLoom crafting
      let loom = env.findHomeBuilding(agent, WeavingLoom)
      if loom != nil:
        state.currentTarget = loom.pos
        state.targetType = WeavingLoom
  else:
    discard  # AltarSpecialist doesn't have resource collection phases

proc findLanternPlacementSpot(env: Environment, agent: Thing, controller: Controller): IVec2 =
  ## Find a good spot to place a lantern: at least 7 squares from altar, 2+ squares from other lanterns
  result = ivec2(-1, -1)
  
  let basePos = agent.homeAltar
  let minDistanceFromAltar = 7  # Stay away from base buildings
  let maxSearchRadius = 20     # Maximum search distance
  
  # Try multiple attempts to find a good spot
  for attempt in 0..80:
    # Generate a random position within search radius
    let angle = controller.rng.rand(0.0..2.0*PI) 
    let distance = controller.rng.rand(minDistanceFromAltar..maxSearchRadius)
    
    let dx = int32(cos(angle) * distance.float)
    let dy = int32(sin(angle) * distance.float)
    let pos = basePos + ivec2(dx, dy)
    
    # Check if position is valid
    if pos.x >= 0 and pos.x < MapWidth and pos.y >= 0 and pos.y < MapHeight:
      if env.isEmpty(pos) and env.terrain[pos.x][pos.y] != Water:
        # Must be at least 7 tiles from altar (to avoid base buildings)
        let distanceFromAltar = manhattanDistance(pos, basePos)
        if distanceFromAltar >= minDistanceFromAltar:
          # Use clippy-style spacing: no lanterns within 2 tiles  
          if not env.hasNearbyLanterns(pos, 2):
            return pos
  
  # No good spot found
  return ivec2(-1, -1)

proc findNearestClippy(env: Environment, pos: IVec2, maxDist: int = 10): Thing =
  ## Find nearest clippy within specified range
  result = nil
  var minDist = maxDist + 1
  
  for thing in env.things:
    if thing.kind == Clippy:
      let dist = manhattanDistance(pos, thing.pos)
      if dist <= maxDist and dist < minDist:
        minDist = dist
        result = thing

proc canAttackClippy(agentPos: IVec2, clippyPos: IVec2): bool =
  ## Check if clippy is within spear attack range (Manhattan distance <= 2)
  let dist = manhattanDistance(agentPos, clippyPos)
  result = dist <= 2 and dist >= 1  # Must be 1-2 tiles away

proc decideAction*(controller: Controller, env: Environment, agentId: int): array[2, uint8] =
  let agent = env.agents[agentId]
  
  # Skip frozen agents
  if agent.frozen > 0:
    return [0'u8, 0'u8]
  
  # Initialize agent state if needed
  if agentId notin controller.agentStates:
    # Use home altar as base, or current position if no home
    let basePos = if agent.homeAltar.x >= 0: agent.homeAltar else: agent.pos
    controller.initAgentState(agentId, basePos, env)
  
  var state = controller.agentStates[agentId]
  
  # Enhanced dithering to prevent getting stuck
  # Higher chance for specialists who need to reach specific targets
  let ditherChance = if state.role != AltarSpecialist: 0.25 else: 0.1  # 25% for specialists, 10% for altar agents
  if controller.rng.rand(0.0..1.0) < ditherChance:
    # Try multiple random directions to find a valid move
    let allDirections = @[ivec2(0, -1), ivec2(1, 0), ivec2(0, 1), ivec2(-1, 0),  # Cardinals first
                         ivec2(1, -1), ivec2(1, 1), ivec2(-1, 1), ivec2(-1, -1)] # Then diagonals
    var shuffledDirs = allDirections
    for i in countdown(shuffledDirs.len - 1, 1):
      let j = controller.rng.rand(0..i)
      let temp = shuffledDirs[i]
      shuffledDirs[i] = shuffledDirs[j]
      shuffledDirs[j] = temp
    
    for dir in shuffledDirs:
      let testPos = agent.pos + dir
      if env.isEmpty(testPos):
        let orientation = getOrientation(dir)
        return [1'u8, ord(orientation).uint8]  # Random move
    # If all random moves fail, fall through to normal logic
  
  
  # Enhanced stuck detection: Check for both stationary and oscillation patterns
  const StuckThreshold = 3  # Consider stuck after 3 repetitions
  const EscapeSteps = 8  # Escape for 8 steps when stuck
  const PositionHistorySize = 4  # Track last 4 positions
  
  # Update position history
  state.recentPositions.add(agent.pos)
  if state.recentPositions.len > PositionHistorySize:
    state.recentPositions.delete(0)  # Remove oldest position
  
  # Check for stuck patterns: either same position or oscillation
  if state.recentPositions.len >= StuckThreshold:
    # Check if we're repeating the same position
    if agent.pos == state.lastPosition:
      state.stuckCounter += 1
    # Check for oscillation pattern (alternating between two positions)
    elif state.recentPositions.len >= 4:
      let pos0 = state.recentPositions[^1]  # Current
      let pos1 = state.recentPositions[^2]  # Previous
      let pos2 = state.recentPositions[^3]  # Two steps ago
      let pos3 = state.recentPositions[^4]  # Three steps ago
      
      if (pos0 == pos2 and pos1 == pos3) or (pos0 == pos1):  # Oscillation or stuck
        state.stuckCounter += 1
      else:
        state.stuckCounter = 0  # Making progress
    else:
      state.stuckCounter = 0  # Not enough history yet
  
  if state.stuckCounter >= StuckThreshold and not state.escapeMode:
      # We're stuck! Enter escape mode
      state.escapeMode = true
      state.escapeStepsRemaining = EscapeSteps
      state.recentPositions.setLen(0)  # Clear history when entering escape mode
      
      # Choose smart escape direction
      if state.currentTarget != agent.pos:
        let targetDir = getDirectionTo(agent.pos, state.currentTarget)
        # Try perpendicular directions first (better for navigating around obstacles)
        let perpendicular1 = ivec2(-targetDir.y, targetDir.x)  # 90 degrees clockwise
        let perpendicular2 = ivec2(targetDir.y, -targetDir.x)  # 90 degrees counter-clockwise
        let opposite = ivec2(-targetDir.x, -targetDir.y)       # 180 degrees
        
        # Test perpendicular directions first, then opposite
        let candidateEscapes = @[perpendicular1, perpendicular2, opposite]
        var foundEscape = false
        for escapeDir in candidateEscapes:
          let testPos = agent.pos + escapeDir
          if env.isEmpty(testPos):
            state.escapeDirection = escapeDir
            foundEscape = true
            break
        
        if not foundEscape:
          # All directions blocked, try random cardinal direction
          let dirs = @[ivec2(0, -1), ivec2(0, 1), ivec2(-1, 0), ivec2(1, 0)]
          state.escapeDirection = controller.rng.sample(dirs)
      else:
        # Random escape direction
        let dirs = @[ivec2(0, -1), ivec2(0, 1), ivec2(-1, 0), ivec2(1, 0)]
        state.escapeDirection = controller.rng.sample(dirs)
  
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
  
  # Check if role is completed for specialized agents
  case state.role:
  of ArmorySpecialist:
    state.hasCompletedRole = agent.inventoryArmor > 0
  of ForgeSpecialist:
    state.hasCompletedRole = false  # Never complete - keep hunting clippys with spear
  of ClayOvenSpecialist:
    state.hasCompletedRole = true  # Food is consumed immediately, assume completed after crafting
  of WeavingLoomSpecialist:
    state.hasCompletedRole = false  # Never complete - keep producing lanterns continuously
  else:
    state.hasCompletedRole = false  # AltarSpecialist never "completes"
  
  
  # Check for transitions after resource collection
  checkAndTransitionAfterCollection(state, agent, env)
  
  # SPECIALIZATION LOGIC: If agent has a specialized role and hasn't completed it, prioritize that
  if state.role != AltarSpecialist and not state.hasCompletedRole:
    case state.role:
    of ArmorySpecialist:
      # Need wood, then craft at Armory
      if agent.inventoryWood == 0:
        # Find wood (trees)
        if state.targetType != Tree:
          let nearestTree = env.findNearestTerrainResource(agent.pos, Tree, maxDist = 20)
          if nearestTree.x >= 0:
            state.currentTarget = nearestTree
            state.targetType = Tree
          else:
            # No trees found, wander to find some
            state.currentTarget = controller.getNextWanderPoint(state)
            state.targetType = Wander
        
        # Actions will be handled by unified action system below
      
      else:
        # Have wood, find Armory in our village
        if state.targetType != Armory:
          let armory = env.findHomeBuilding(agent, Armory)
          if armory != nil:
            state.currentTarget = armory.pos
            state.targetType = Armory
        
        # Actions will be handled by unified action system below
    
    of ForgeSpecialist:
      # Need wood, then craft at Forge, then hunt with spear (never complete)
      if agent.inventorySpear > 0:
        # Have spear, hunt clippies
        let clippy = findNearestClippy(env, agent.pos, maxDist = 15)
        if clippy != nil and canAttackClippy(agent.pos, clippy.pos):
          # Can attack! Do it now
          let dir = clippy.pos - agent.pos
          return [2'u8, ord(getOrientation(dir)).uint8]  # Attack
        elif clippy != nil:
          # Move toward clippy
          state.currentTarget = clippy.pos
          state.targetType = ClippyHunt
        else:
          # No clippies nearby, wander to find them
          state.currentTarget = controller.getNextWanderPoint(state)
          state.targetType = ClippyHunt
      elif agent.inventoryWood > 0:
        # Have wood, go to forge to craft spear
        if state.targetType != Forge:
          let forge = env.findHomeBuilding(agent, Forge)
          if forge != nil:
            state.currentTarget = forge.pos
            state.targetType = Forge
        
        # Actions will be handled by unified action system below
      else:
        # Need wood, find trees
        if state.targetType != Tree:
          let nearestTree = env.findNearestTerrainResource(agent.pos, Tree, maxDist = 20)
          if nearestTree.x >= 0:
            state.currentTarget = nearestTree
            state.targetType = Tree
          else:
            state.currentTarget = controller.getNextWanderPoint(state)
            state.targetType = Wander
        
        # Actions will be handled by unified action system below
    
    of ClayOvenSpecialist:
      # Need wheat, then craft at ClayOven
      if agent.inventoryWheat == 0:
        if state.targetType != Wheat:
          let nearestWheat = env.findNearestTerrainResource(agent.pos, Wheat, maxDist = 20)
          if nearestWheat.x >= 0:
            state.currentTarget = nearestWheat
            state.targetType = Wheat
          else:
            state.currentTarget = controller.getNextWanderPoint(state)
            state.targetType = Wander
        
        # Actions will be handled by unified action system below
      
      else:
        if state.targetType != ClayOven:
          let clayOven = env.findHomeBuilding(agent, ClayOven)
          if clayOven != nil:
            state.currentTarget = clayOven.pos
            state.targetType = ClayOven
        
        # Actions will be handled by unified action system below
    
    of WeavingLoomSpecialist:
      # Need wheat, then craft at WeavingLoom, then plant lantern (never complete)
      if agent.inventoryLantern > 0:
        # Have lantern, find place to plant (7+ tiles from altar)
        let plantSpot = env.findLanternPlacementSpot(agent, controller)
        if plantSpot.x >= 0:
          state.currentTarget = plantSpot
          state.targetType = LanternPlantSpot
        else:
          # No good spot, continue spiraling to find one
          state.currentTarget = controller.getNextWanderPoint(state)
          state.targetType = Wander
      elif agent.inventoryWheat > 0:
        # Have wheat, craft lantern
        if state.targetType != WeavingLoom:
          let weavingLoom = env.findHomeBuilding(agent, WeavingLoom)
          if weavingLoom != nil:
            state.currentTarget = weavingLoom.pos
            state.targetType = WeavingLoom
        
        # Actions will be handled by unified action system below
      else:
        # Need wheat
        if state.targetType != Wheat:
          let nearestWheat = env.findNearestTerrainResource(agent.pos, Wheat, maxDist = 20)
          if nearestWheat.x >= 0:
            state.currentTarget = nearestWheat
            state.targetType = Wheat
          else:
            state.currentTarget = controller.getNextWanderPoint(state)
            state.targetType = Wander
        
        # Actions will be handled by unified action system below
    
    else:
      discard  # AltarSpecialist handled below
    
    # If we have a specialized target, move toward it
    if state.targetType in [Tree, Wheat, Armory, Forge, ClayOven, WeavingLoom, LanternPlantSpot, ClippyHunt]:
      # Special handling for resource tiles (Tree, Wheat) - need to be adjacent, not on top
      if state.targetType in [Tree, Wheat] and state.currentTarget == agent.pos:
        # We're on top of the resource, move to an adjacent cardinal position
        let adjacentPositions = @[
          state.currentTarget + ivec2(0, -1),  # North
          state.currentTarget + ivec2(1, 0),   # East  
          state.currentTarget + ivec2(0, 1),   # South
          state.currentTarget + ivec2(-1, 0)   # West
        ]
        for adjPos in adjacentPositions:
          if env.isEmpty(adjPos):
            let moveDir = adjPos - agent.pos
            let moveOrient = getOrientation(moveDir)
            return [1'u8, ord(moveOrient).uint8]  # Move to adjacent position
      elif state.currentTarget != agent.pos:
        let dir = getDirectionTo(agent.pos, state.currentTarget)
        if dir.x != 0 or dir.y != 0:
          let moveOrientation = getOrientation(dir)
          let nextPos = agent.pos + dir
          if env.isEmpty(nextPos):
            return [1'u8, ord(moveOrientation).uint8]  # Move toward specialized target
          elif isAdjacent(agent.pos, state.currentTarget):
            # Try to get to cardinal position
            let moveDir = getMoveToCardinalPosition(agent.pos, state.currentTarget, env)
            if moveDir.x != 0 or moveDir.y != 0:
              let moveOrient = getOrientation(moveDir)
              return [1'u8, ord(moveOrient).uint8]
    
    # UNIFIED ACTION SYSTEM: Handle actions at target locations
    if isCardinallyAdjacent(agent.pos, state.currentTarget):
      case state.targetType:
      of Tree:
        if env.terrain[state.currentTarget.x][state.currentTarget.y] == Tree:
          let dir = state.currentTarget - agent.pos
          # After collecting wood, agent will transition on next decision cycle
          return [3'u8, ord(getOrientation(dir)).uint8]  # Get wood from tree
        else:
          # Tree gone, search for new target based on role
          state.targetType = NoTarget
      of Wheat:
        if env.terrain[state.currentTarget.x][state.currentTarget.y] == Wheat:
          let dir = state.currentTarget - agent.pos
          # After collecting wheat, agent will transition on next decision cycle
          return [3'u8, ord(getOrientation(dir)).uint8]  # Get wheat
        else:
          # Wheat gone, search for new target based on role
          state.targetType = NoTarget
      of Armory, Forge, ClayOven, WeavingLoom:
        # Craft at building (put action)
        let dir = state.currentTarget - agent.pos
        return [5'u8, ord(getOrientation(dir)).uint8]  # Put/craft action
      of LanternPlantSpot:
        # Plant lantern
        let dir = state.currentTarget - agent.pos
        return [6'u8, ord(getOrientation(dir)).uint8]  # Plant action
      else:
        discard
    
    # If we're a specialist but couldn't move or act, just wait (don't fall through to altar behavior)
    return [0'u8, 0'u8]  # Noop for specialists
  
  # DEFAULT BEHAVIOR: Standard altar loop (for AltarSpecialist or completed specialists)
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
        
        # Put battery into altar
        return [5'u8, ord(getOrientation(dir)).uint8]  # Put action with direction
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
      
      for thing in env.things:
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
        
        # Get battery from converter (converts ore to battery)
        return [3'u8, ord(getOrientation(dir)).uint8]  # Get action with direction
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
      
      for thing in env.things:
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
        let randomDelta = getOrientationDelta(randomOrientation)
        let testPos = agent.pos + ivec2(randomDelta.x.int32, randomDelta.y.int32)
        if env.isEmpty(testPos):
          return [1'u8, ord(randomOrientation).uint8]  # Move in random direction
  
  # Default: do nothing
  return [0'u8, 0'u8]  # Noop



proc updateController*(controller: Controller) =
  controller.stepCount += 1