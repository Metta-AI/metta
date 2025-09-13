## Simplified AI system - clean and efficient
## Replaces the 1200+ line complex system with ~150 lines
import std/[random, tables]
import vmath
import environment, common

type
  # Simple agent roles - one per team member
  AgentRole* = enum
    AltarSpecialist      # Handles altar/battery workflow
    ArmorySpecialist     # Wood → Armor
    ForgeSpecialist      # Wood → Spear → Hunt Clippies  
    ClayOvenSpecialist   # Wheat → Bread
    WeavingLoomSpecialist # Wheat → Lantern → Plant

  # Minimal state tracking with spiral search
  AgentState = object
    role: AgentRole
    initialized: bool
    # Spiral search state
    spiralStepsInArc: int
    spiralArcsCompleted: int
    basePosition: IVec2
    lastSearchPosition: IVec2
    # Bail-out / anti-oscillation state
    lastPosition: IVec2
    recentPositions: seq[IVec2]
    stuckCounter: int
    escapeMode: bool
    escapeStepsRemaining: int
    escapeDirection: IVec2

  # Simple controller
  Controller* = ref object
    rng*: Rand
    agents: Table[int, AgentState]

# Global controller instance for compatibility  
var globalSimpleController: Controller

proc newController*(seed: int): Controller =
  result = Controller(
    rng: initRand(seed),
    agents: initTable[int, AgentState]()
  )
  globalSimpleController = result

# Helper proc to save state and return action
proc saveStateAndReturn(controller: Controller, agentId: int, state: AgentState, action: array[2, uint8]): array[2, uint8] =
  controller.agents[agentId] = state
  return action

proc vecToOrientation(vec: IVec2): int =
  ## Map a step vector to orientation index (0..7)
  let x = vec.x
  let y = vec.y
  if x == 0'i32 and y == -1'i32: return 0  # N
  elif x == 0'i32 and y == 1'i32: return 1  # S
  elif x == -1'i32 and y == 0'i32: return 2 # W
  elif x == 1'i32 and y == 0'i32: return 3  # E
  elif x == -1'i32 and y == -1'i32: return 4 # NW
  elif x == 1'i32 and y == -1'i32: return 5  # NE
  elif x == -1'i32 and y == 1'i32: return 6  # SW
  elif x == 1'i32 and y == 1'i32: return 7   # SE
  else: return 0

proc applyDirectionOffset(offset: var IVec2, direction: int, distance: int32) =
  case direction:
  of 0: offset.y -= distance  # North
  of 1: offset.x += distance  # East
  of 2: offset.y += distance  # South
  of 3: offset.x -= distance  # West
  else: discard

proc getNextSpiralPoint(state: var AgentState, rng: var Rand): IVec2 =
  ## Generate next position in expanding spiral search pattern
  # Track current position in spiral
  var totalOffset = ivec2(0, 0)
  var currentArcLength = 1
  var direction = 0
  
  # Rebuild position by simulating all steps up to current point
  for arcNum in 0 ..< state.spiralArcsCompleted:
    let arcLen = (arcNum div 2) + 1  # 1,1,2,2,3,3,4,4...
    let dir = arcNum mod 4  # Direction cycles 0,1,2,3 (N,E,S,W)
    applyDirectionOffset(totalOffset, dir, int32(arcLen))
  
  # Add partial progress in current arc
  currentArcLength = (state.spiralArcsCompleted div 2) + 1
  direction = state.spiralArcsCompleted mod 4
  
  # Add steps taken in current arc
  applyDirectionOffset(totalOffset, direction, int32(state.spiralStepsInArc))
  
  # Calculate next step
  state.spiralStepsInArc += 1
  
  # Check if we completed the current arc
  if state.spiralStepsInArc > currentArcLength:
    # Move to next arc
    state.spiralArcsCompleted += 1
    state.spiralStepsInArc = 1
    
    # Reset spiral after ~100 arcs (radius ~50) to avoid going too far
    if state.spiralArcsCompleted > 100:
      state.spiralArcsCompleted = 0  # Reset to start of spiral
      state.spiralStepsInArc = 1
      # Don't return to base immediately, continue spiral from current area
      state.basePosition = state.lastSearchPosition
  
  # Calculate next position
  applyDirectionOffset(totalOffset, direction, 1)
  result = state.basePosition + totalOffset

proc findNearestThing(env: Environment, pos: IVec2, kind: ThingKind): Thing =
  result = nil
  var minDist = 999999
  for thing in env.things:
    if thing.kind == kind:
      let dist = abs(thing.pos.x - pos.x) + abs(thing.pos.y - pos.y)
      if dist < minDist and dist < 30:  # Reasonable search radius
        minDist = dist
        result = thing

proc findNearestThingSpiral(env: Environment, state: var AgentState, kind: ThingKind, rng: var Rand): Thing =
  ## Find nearest thing using spiral search pattern - more systematic than random search
  # First check immediate area around current position
  result = findNearestThing(env, state.lastSearchPosition, kind)
  if result != nil:
    return result
    
  # Also check around agent's current position before advancing spiral
  result = findNearestThing(env, state.basePosition, kind)
  if result != nil:
    return result
  
  # If not found, advance spiral search only every few calls to reduce dithering
  let nextSearchPos = getNextSpiralPoint(state, rng)
  state.lastSearchPosition = nextSearchPos
  
  # Search from new spiral position
  result = findNearestThing(env, nextSearchPos, kind)
  return result

proc findNearestTerrain(env: Environment, pos: IVec2, terrain: TerrainType): IVec2 =
  result = ivec2(-1, -1)
  var minDist = 999999
  for x in max(0, pos.x - 20)..<min(MapWidth, pos.x + 21):
    for y in max(0, pos.y - 20)..<min(MapHeight, pos.y + 21):
      if env.terrain[x][y] == terrain:
        let terrainPos = ivec2(x.int32, y.int32)
        let dist = abs(terrainPos.x - pos.x) + abs(terrainPos.y - pos.y)
        if dist < minDist:
          minDist = dist
          result = terrainPos

proc findNearestTerrainSpiral(env: Environment, state: var AgentState, terrain: TerrainType, rng: var Rand): IVec2 =
  ## Find terrain using spiral search pattern
  # First check from current spiral search position
  result = findNearestTerrain(env, state.lastSearchPosition, terrain)
  if result.x >= 0:
    return result
    
  # Also check around agent's current position before advancing spiral
  result = findNearestTerrain(env, state.basePosition, terrain)
  if result.x >= 0:
    return result
  
  # If not found, advance spiral search
  let nextSearchPos = getNextSpiralPoint(state, rng)
  state.lastSearchPosition = nextSearchPos
  
  # Search from new spiral position
  result = findNearestTerrain(env, nextSearchPos, terrain)
  return result

proc getCardinalDirIndex(fromPos, toPos: IVec2): int =
  ## Convert direction to orientation (0=N, 1=S, 2=W, 3=E, 4=NW, 5=NE, 6=SW, 7=SE)
  let dx = toPos.x - fromPos.x
  let dy = toPos.y - fromPos.y
  
  # Handle cardinal directions first (simpler pathfinding)
  if abs(dx) > abs(dy):
    if dx > 0: return 3  # East
    else: return 2       # West
  else:
    if dy > 0: return 1  # South  
    else: return 0       # North

proc neighborDirIndex(fromPos, toPos: IVec2): int =
  ## Orientation index (0..7) toward adjacent target (includes diagonals)
  let dx = toPos.x - fromPos.x
  let dy = toPos.y - fromPos.y
  let sx = (if dx > 0: 1'i32 elif dx < 0: -1'i32 else: 0'i32)
  let sy = (if dy > 0: 1'i32 elif dy < 0: -1'i32 else: 0'i32)
  return vecToOrientation(ivec2(sx.int, sy.int))

proc isPassable(env: Environment, pos: IVec2): bool =
  ## Consider lantern tiles passable for movement planning
  if env.isEmpty(pos): return true
  for t in env.things:
    if t.pos == pos and t.kind == PlantedLantern:
      return true
  return false

proc getMoveTowards(env: Environment, fromPos, toPos: IVec2, rng: var Rand): int =
  ## Get a movement direction towards target, with obstacle avoidance
  let primaryDir = getCardinalDirIndex(fromPos, toPos)
  
  # Try primary direction first
  let directions = [
    ivec2(0, -1),  # 0: North
    ivec2(0, 1),   # 1: South  
    ivec2(-1, 0),  # 2: West
    ivec2(1, 0),   # 3: East
    ivec2(-1, -1), # 4: NW
    ivec2(1, -1),  # 5: NE
    ivec2(-1, 1),  # 6: SW  
    ivec2(1, 1)    # 7: SE
  ]
  
  let primaryMove = fromPos + directions[primaryDir]
  if isPassable(env, primaryMove):
    return primaryDir
  
  # Primary blocked, try adjacent directions
  let alternatives = case primaryDir:
    of 0: @[5, 4, 1, 3, 2]  # North blocked, try NE, NW, South, East, West
    of 1: @[7, 6, 0, 3, 2]  # South blocked, try SE, SW, North, East, West  
    of 2: @[4, 6, 3, 0, 1]  # West blocked, try NW, SW, East, North, South
    of 3: @[5, 7, 2, 0, 1]  # East blocked, try NE, SE, West, North, South
    else: @[0, 1, 2, 3]     # Diagonal blocked, try cardinals
  
  for altDir in alternatives:
    let altMove = fromPos + directions[altDir]
    if isPassable(env, altMove):
      return altDir
  
  # All blocked, try random movement
  return rng.rand(0..3)

proc manhattanDistance(a, b: IVec2): int =
  abs(a.x - b.x) + abs(a.y - b.y)

proc decideAction*(controller: Controller, env: Environment, agentId: int): array[2, uint8] =
  let agent = env.agents[agentId]
  
  # Skip frozen agents
  if agent.frozen > 0:
    return [0'u8, 0'u8]
  
  # Initialize agent role if needed
  if agentId notin controller.agents:
    let role = case agentId mod 5:
      of 0: AltarSpecialist
      of 1: ArmorySpecialist
      of 2: ForgeSpecialist  
      of 3: ClayOvenSpecialist
      of 4: WeavingLoomSpecialist
      else: AltarSpecialist
    
    controller.agents[agentId] = AgentState(
      role: role,
      initialized: true,
      spiralStepsInArc: 0,
      spiralArcsCompleted: 0,
      basePosition: agent.pos,
      lastSearchPosition: agent.pos,
      lastPosition: agent.pos,
      recentPositions: @[],
      stuckCounter: 0,
      escapeMode: false,
      escapeStepsRemaining: 0,
      escapeDirection: ivec2(0, -1)
    )
  
  var state = controller.agents[agentId]

  # --- Simple bail-out and dithering to avoid getting stuck/oscillation ---
  # Update recent positions history (size 4)
  state.recentPositions.add(agent.pos)
  if state.recentPositions.len > 4:
    state.recentPositions.delete(0)

  # Detect stuck: same position or simple 2-cycle oscillation
  if state.recentPositions.len >= 2 and agent.pos == state.lastPosition:
    inc state.stuckCounter
  elif state.recentPositions.len >= 4:
    let p0 = state.recentPositions[^1]
    let p1 = state.recentPositions[^2]
    let p2 = state.recentPositions[^3]
    let p3 = state.recentPositions[^4]
    if (p0 == p2 and p1 == p3) or (p0 == p1):
      inc state.stuckCounter
    else:
      state.stuckCounter = 0
  else:
    state.stuckCounter = 0

  # Enter escape mode if stuck
  if not state.escapeMode and state.stuckCounter >= 3:
    state.escapeMode = true
    state.escapeStepsRemaining = 6
    state.recentPositions.setLen(0)
    # Choose an escape direction: prefer any empty cardinal, shuffled
    var dirs = @[ivec2(0, -1), ivec2(1, 0), ivec2(0, 1), ivec2(-1, 0)]
    for i in countdown(dirs.len - 1, 1):
      let j = controller.rng.rand(0..i)
      let tmp = dirs[i]
      dirs[i] = dirs[j]
      dirs[j] = tmp
    var chosen = ivec2(0, -1)
    for d in dirs:
      if env.isEmpty(agent.pos + d):
        chosen = d
        break
    state.escapeDirection = chosen

  # If in escape mode, try to move in escape direction for a few steps
  if state.escapeMode and state.escapeStepsRemaining > 0:
    let tryDirs = @[state.escapeDirection,
                    ivec2(state.escapeDirection.y, -state.escapeDirection.x),  # perpendicular 1
                    ivec2(-state.escapeDirection.y, state.escapeDirection.x),  # perpendicular 2
                    ivec2(-state.escapeDirection.x, -state.escapeDirection.y)] # opposite
    for d in tryDirs:
      let np = agent.pos + d
      if env.isEmpty(np):
        dec state.escapeStepsRemaining
        if state.escapeStepsRemaining <= 0:
          state.escapeMode = false
          state.stuckCounter = 0
        state.lastPosition = agent.pos
        return saveStateAndReturn(controller, agentId, state, [1'u8, vecToOrientation(d).uint8])
    # If all blocked, drop out of escape for this tick
    state.escapeMode = false
    state.stuckCounter = 0

  # Small dithering chance to break deadlocks (higher for non-altar roles)
  let ditherChance = if state.role == AltarSpecialist: 0.10 else: 0.20
  if controller.rng.rand(0.0..1.0) < ditherChance:
    var candidates = @[ivec2(0, -1), ivec2(1, 0), ivec2(0, 1), ivec2(-1, 0),
                       ivec2(1, -1), ivec2(1, 1), ivec2(-1, 1), ivec2(-1, -1)]
    for i in countdown(candidates.len - 1, 1):
      let j = controller.rng.rand(0..i)
      let tmp = candidates[i]
      candidates[i] = candidates[j]
      candidates[j] = tmp
    for d in candidates:
      if env.isEmpty(agent.pos + d):
        state.lastPosition = agent.pos
        return saveStateAndReturn(controller, agentId, state, [1'u8, vecToOrientation(d).uint8])

  # From here on, ensure lastPosition is updated this tick regardless of branch
  state.lastPosition = agent.pos
  
  # Role-based decision making
  case state.role:
  
  of WeavingLoomSpecialist:
    # Priority 1: Plant lantern if we have one
    if agent.inventoryLantern > 0:
      # Find a good spot nearby to plant (simple: just plant north of current position)
      return saveStateAndReturn(controller, agentId, state, [6'u8, 0'u8])  # Plant lantern North
    
    # Priority 2: Craft lantern if we have wheat  
    elif agent.inventoryWheat > 0:
      let loom = env.findNearestThingSpiral(state, WeavingLoom, controller.rng)
      if loom != nil:
        let dx = abs(loom.pos.x - agent.pos.x)
        let dy = abs(loom.pos.y - agent.pos.y)
        if max(dx, dy) == 1'i32:
          # Adjacent (8-neighborhood) to loom - craft lantern
          return saveStateAndReturn(controller, agentId, state, [5'u8, neighborDirIndex(agent.pos, loom.pos).uint8])
        else:
          # Move toward loom
          return saveStateAndReturn(controller, agentId, state, [1'u8, getMoveTowards(env, agent.pos, loom.pos, controller.rng).uint8])
    
    # Priority 3: Collect wheat using spiral search
    else:
      let wheatPos = env.findNearestTerrainSpiral(state, Wheat, controller.rng)
      if wheatPos.x >= 0:
        let dx = abs(wheatPos.x - agent.pos.x)
        let dy = abs(wheatPos.y - agent.pos.y)
        if max(dx, dy) == 1'i32:
          # Adjacent (8-neighborhood) to wheat - harvest it
          return saveStateAndReturn(controller, agentId, state, [3'u8, neighborDirIndex(agent.pos, wheatPos).uint8])
        else:
          # Move toward wheat
          return saveStateAndReturn(controller, agentId, state, [1'u8, getMoveTowards(env, agent.pos, wheatPos, controller.rng).uint8])
      else:
        # No wheat found, continue spiral search
        let nextSearchPos = getNextSpiralPoint(state, controller.rng)
        return saveStateAndReturn(controller, agentId, state, [1'u8, getMoveTowards(env, agent.pos, nextSearchPos, controller.rng).uint8])

  of ArmorySpecialist:
    # Priority 1: Craft armor if we have wood
    if agent.inventoryWood > 0:
      let armory = env.findNearestThingSpiral(state, Armory, controller.rng)
      if armory != nil:
        let dx = abs(armory.pos.x - agent.pos.x)
        let dy = abs(armory.pos.y - agent.pos.y)
        if max(dx, dy) == 1'i32:
          return saveStateAndReturn(controller, agentId, state, [5'u8, neighborDirIndex(agent.pos, armory.pos).uint8])
        else:
          return saveStateAndReturn(controller, agentId, state, [1'u8, getMoveTowards(env, agent.pos, armory.pos, controller.rng).uint8])
      else:
        # No armory found, continue spiral search
        let nextSearchPos = getNextSpiralPoint(state, controller.rng)
        return saveStateAndReturn(controller, agentId, state, [1'u8, getMoveTowards(env, agent.pos, nextSearchPos, controller.rng).uint8])
    
    # Priority 2: Collect wood using spiral search
    else:
      let treePos = env.findNearestTerrainSpiral(state, Tree, controller.rng)
      if treePos.x >= 0:
        let dx = abs(treePos.x - agent.pos.x)
        let dy = abs(treePos.y - agent.pos.y)
        if max(dx, dy) == 1'i32:
          return saveStateAndReturn(controller, agentId, state, [3'u8, neighborDirIndex(agent.pos, treePos).uint8])
        else:
          return saveStateAndReturn(controller, agentId, state, [1'u8, getMoveTowards(env, agent.pos, treePos, controller.rng).uint8])
      else:
        # No trees found, continue spiral search
        let nextSearchPos = getNextSpiralPoint(state, controller.rng)
        return saveStateAndReturn(controller, agentId, state, [1'u8, getMoveTowards(env, agent.pos, nextSearchPos, controller.rng).uint8])

  of ForgeSpecialist:
    # Priority 1: Hunt clippies if we have spear using spiral search
    if agent.inventorySpear > 0:
      let clippy = env.findNearestThingSpiral(state, Clippy, controller.rng)
      if clippy != nil:
        let dx = abs(clippy.pos.x - agent.pos.x)
        let dy = abs(clippy.pos.y - agent.pos.y)
        if max(dx, dy) in [1'i32, 2'i32]:  # Spear attack range (chebyshev 1-2)
          return saveStateAndReturn(controller, agentId, state, [2'u8, neighborDirIndex(agent.pos, clippy.pos).uint8])
        else:
          return saveStateAndReturn(controller, agentId, state, [1'u8, getMoveTowards(env, agent.pos, clippy.pos, controller.rng).uint8])
      else:
        # No clippies found, continue spiral search for hunting
        let nextSearchPos = getNextSpiralPoint(state, controller.rng)
        return saveStateAndReturn(controller, agentId, state, [1'u8, getMoveTowards(env, agent.pos, nextSearchPos, controller.rng).uint8])
    
    # Priority 2: Craft spear if we have wood
    elif agent.inventoryWood > 0:
      let forge = env.findNearestThingSpiral(state, Forge, controller.rng)
      if forge != nil:
        let dx = abs(forge.pos.x - agent.pos.x)
        let dy = abs(forge.pos.y - agent.pos.y)
        if max(dx, dy) == 1'i32:
          return saveStateAndReturn(controller, agentId, state, [5'u8, neighborDirIndex(agent.pos, forge.pos).uint8])
        else:
          return saveStateAndReturn(controller, agentId, state, [1'u8, getMoveTowards(env, agent.pos, forge.pos, controller.rng).uint8])
    
    # Priority 3: Collect wood using spiral search
    else:
      let treePos = env.findNearestTerrainSpiral(state, Tree, controller.rng)
      if treePos.x >= 0:
        let dx = abs(treePos.x - agent.pos.x)
        let dy = abs(treePos.y - agent.pos.y)
        if max(dx, dy) == 1'i32:
          return saveStateAndReturn(controller, agentId, state, [3'u8, neighborDirIndex(agent.pos, treePos).uint8])
        else:
          return saveStateAndReturn(controller, agentId, state, [1'u8, getMoveTowards(env, agent.pos, treePos, controller.rng).uint8])
      else:
        # No trees found, continue spiral search
        let nextSearchPos = getNextSpiralPoint(state, controller.rng)
        return saveStateAndReturn(controller, agentId, state, [1'u8, getMoveTowards(env, agent.pos, nextSearchPos, controller.rng).uint8])

  of ClayOvenSpecialist:
    # Priority 1: Craft bread if we have wheat
    if agent.inventoryWheat > 0:
      let oven = env.findNearestThingSpiral(state, ClayOven, controller.rng)
      if oven != nil:
        let dx = abs(oven.pos.x - agent.pos.x)
        let dy = abs(oven.pos.y - agent.pos.y)
        if max(dx, dy) == 1'i32:
          return saveStateAndReturn(controller, agentId, state, [5'u8, neighborDirIndex(agent.pos, oven.pos).uint8])
        else:
          return saveStateAndReturn(controller, agentId, state, [1'u8, getMoveTowards(env, agent.pos, oven.pos, controller.rng).uint8])
      else:
        # No oven found, continue spiral search
        let nextSearchPos = getNextSpiralPoint(state, controller.rng)
        return saveStateAndReturn(controller, agentId, state, [1'u8, getMoveTowards(env, agent.pos, nextSearchPos, controller.rng).uint8])
    
    # Priority 2: Collect wheat using spiral search
    else:
      let wheatPos = env.findNearestTerrainSpiral(state, Wheat, controller.rng)
      if wheatPos.x >= 0:
        let dx = abs(wheatPos.x - agent.pos.x)
        let dy = abs(wheatPos.y - agent.pos.y)
        if max(dx, dy) == 1'i32:
          return saveStateAndReturn(controller, agentId, state, [3'u8, neighborDirIndex(agent.pos, wheatPos).uint8])
        else:
          return saveStateAndReturn(controller, agentId, state, [1'u8, getMoveTowards(env, agent.pos, wheatPos, controller.rng).uint8])
      else:
        # No wheat found, continue spiral search
        let nextSearchPos = getNextSpiralPoint(state, controller.rng)
        return saveStateAndReturn(controller, agentId, state, [1'u8, getMoveTowards(env, agent.pos, nextSearchPos, controller.rng).uint8])

  of AltarSpecialist:
    # Handle ore → battery → altar workflow
    if agent.inventoryBattery > 0:
      # Find altar and deposit battery
      for thing in env.things:
        if thing.kind == Altar and thing.pos == agent.homeAltar:
          let dx = abs(thing.pos.x - agent.pos.x)
          let dy = abs(thing.pos.y - agent.pos.y)
          if max(dx, dy) == 1'i32:
            return saveStateAndReturn(controller, agentId, state, [5'u8, neighborDirIndex(agent.pos, thing.pos).uint8])
          else:
            return saveStateAndReturn(controller, agentId, state, [1'u8, getMoveTowards(env, agent.pos, thing.pos, controller.rng).uint8])
    
    elif agent.inventoryOre > 0:
      # Find converter and make battery using spiral search
      let converterThing = env.findNearestThingSpiral(state, Converter, controller.rng)
      if converterThing != nil:
        let dx = abs(converterThing.pos.x - agent.pos.x)
        let dy = abs(converterThing.pos.y - agent.pos.y)
        if max(dx, dy) == 1'i32:
          return saveStateAndReturn(controller, agentId, state, [5'u8, neighborDirIndex(agent.pos, converterThing.pos).uint8])
        else:
          return saveStateAndReturn(controller, agentId, state, [1'u8, getMoveTowards(env, agent.pos, converterThing.pos, controller.rng).uint8])
      else:
        # No converter found, continue spiral search
        let nextSearchPos = getNextSpiralPoint(state, controller.rng)
        return saveStateAndReturn(controller, agentId, state, [1'u8, getMoveTowards(env, agent.pos, nextSearchPos, controller.rng).uint8])
    
    else:
      # Find mine and collect ore using spiral search
      let mine = env.findNearestThingSpiral(state, Mine, controller.rng)
      if mine != nil:
        let dx = abs(mine.pos.x - agent.pos.x)
        let dy = abs(mine.pos.y - agent.pos.y)
        if max(dx, dy) == 1'i32:
          return saveStateAndReturn(controller, agentId, state, [3'u8, neighborDirIndex(agent.pos, mine.pos).uint8])
        else:
          return saveStateAndReturn(controller, agentId, state, [1'u8, getMoveTowards(env, agent.pos, mine.pos, controller.rng).uint8])
      else:
        # No mine found, continue spiral search
        let nextSearchPos = getNextSpiralPoint(state, controller.rng)
        return saveStateAndReturn(controller, agentId, state, [1'u8, getMoveTowards(env, agent.pos, nextSearchPos, controller.rng).uint8])
  
  # Save last position for next tick and return a default random move
  state.lastPosition = agent.pos
  return saveStateAndReturn(controller, agentId, state, [1'u8, controller.rng.rand(0..7).uint8])

# Compatibility function for updateController
proc updateController*(controller: Controller) =
  # No complex state to update - keep it simple
  discard
