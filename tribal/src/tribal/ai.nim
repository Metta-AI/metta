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

  # Minimal state tracking
  AgentState = object
    role: AgentRole
    initialized: bool

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

proc findNearestThing(env: Environment, pos: IVec2, kind: ThingKind): Thing =
  result = nil
  var minDist = 999999
  for thing in env.things:
    if thing.kind == kind:
      let dist = abs(thing.pos.x - pos.x) + abs(thing.pos.y - pos.y)
      if dist < minDist and dist < 30:  # Reasonable search radius
        minDist = dist
        result = thing

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

proc getDirectionTo(fromPos, toPos: IVec2): int =
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

proc getMoveTowards(env: Environment, fromPos, toPos: IVec2, rng: var Rand): int =
  ## Get a movement direction towards target, with obstacle avoidance
  let primaryDir = getDirectionTo(fromPos, toPos)
  
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
  if env.isEmpty(primaryMove):
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
    if env.isEmpty(altMove):
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
    
    controller.agents[agentId] = AgentState(role: role, initialized: true)
  
  let state = controller.agents[agentId]
  
  # Role-based decision making
  case state.role:
  
  of WeavingLoomSpecialist:
    # Priority 1: Plant lantern if we have one
    if agent.inventoryLantern > 0:
      # Find a good spot nearby to plant (simple: just plant north of current position)
      return [6'u8, 0'u8]  # Plant lantern North
    
    # Priority 2: Craft lantern if we have wheat  
    elif agent.inventoryWheat > 0:
      let loom = env.findNearestThing(agent.pos, WeavingLoom)
      if loom != nil:
        let dist = manhattanDistance(agent.pos, loom.pos)
        if dist == 1:
          # Adjacent to loom - craft lantern
          return [5'u8, getDirectionTo(agent.pos, loom.pos).uint8]
        else:
          # Move toward loom
          return [1'u8, getMoveTowards(env, agent.pos, loom.pos, controller.rng).uint8]
    
    # Priority 3: Collect wheat
    else:
      let wheatPos = env.findNearestTerrain(agent.pos, Wheat)
      if wheatPos.x >= 0:
        let dist = manhattanDistance(agent.pos, wheatPos)
        if dist == 1:
          # Adjacent to wheat - harvest it
          return [3'u8, getDirectionTo(agent.pos, wheatPos).uint8]
        else:
          # Move toward wheat
          return [1'u8, getMoveTowards(env, agent.pos, wheatPos, controller.rng).uint8]

  of ArmorySpecialist:
    # Priority 1: Craft armor if we have wood
    if agent.inventoryWood > 0:
      let armory = env.findNearestThing(agent.pos, Armory)
      if armory != nil:
        let dist = manhattanDistance(agent.pos, armory.pos)
        if dist == 1:
          return [5'u8, getDirectionTo(agent.pos, armory.pos).uint8]
        else:
          return [1'u8, getMoveTowards(env, agent.pos, armory.pos, controller.rng).uint8]
    
    # Priority 2: Collect wood
    else:
      let treePos = env.findNearestTerrain(agent.pos, Tree)
      if treePos.x >= 0:
        let dist = manhattanDistance(agent.pos, treePos)
        if dist == 1:
          return [3'u8, getDirectionTo(agent.pos, treePos).uint8]
        else:
          return [1'u8, getMoveTowards(env, agent.pos, treePos, controller.rng).uint8]

  of ForgeSpecialist:
    # Priority 1: Hunt clippies if we have spear
    if agent.inventorySpear > 0:
      let clippy = env.findNearestThing(agent.pos, Clippy)
      if clippy != nil:
        let dist = manhattanDistance(agent.pos, clippy.pos)
        if dist <= 2 and dist >= 1:  # Spear attack range
          return [2'u8, getDirectionTo(agent.pos, clippy.pos).uint8]
        else:
          return [1'u8, getMoveTowards(env, agent.pos, clippy.pos, controller.rng).uint8]
    
    # Priority 2: Craft spear if we have wood
    elif agent.inventoryWood > 0:
      let forge = env.findNearestThing(agent.pos, Forge)
      if forge != nil:
        let dist = manhattanDistance(agent.pos, forge.pos)
        if dist == 1:
          return [5'u8, getDirectionTo(agent.pos, forge.pos).uint8]
        else:
          return [1'u8, getMoveTowards(env, agent.pos, forge.pos, controller.rng).uint8]
    
    # Priority 3: Collect wood
    else:
      let treePos = env.findNearestTerrain(agent.pos, Tree)
      if treePos.x >= 0:
        let dist = manhattanDistance(agent.pos, treePos)
        if dist == 1:
          return [3'u8, getDirectionTo(agent.pos, treePos).uint8]
        else:
          return [1'u8, getMoveTowards(env, agent.pos, treePos, controller.rng).uint8]

  of ClayOvenSpecialist:
    # Priority 1: Craft bread if we have wheat
    if agent.inventoryWheat > 0:
      let oven = env.findNearestThing(agent.pos, ClayOven)
      if oven != nil:
        let dist = manhattanDistance(agent.pos, oven.pos)
        if dist == 1:
          return [5'u8, getDirectionTo(agent.pos, oven.pos).uint8]
        else:
          return [1'u8, getMoveTowards(env, agent.pos, oven.pos, controller.rng).uint8]
    
    # Priority 2: Collect wheat
    else:
      let wheatPos = env.findNearestTerrain(agent.pos, Wheat)
      if wheatPos.x >= 0:
        let dist = manhattanDistance(agent.pos, wheatPos)
        if dist == 1:
          return [3'u8, getDirectionTo(agent.pos, wheatPos).uint8]
        else:
          return [1'u8, getMoveTowards(env, agent.pos, wheatPos, controller.rng).uint8]

  of AltarSpecialist:
    # Handle ore → battery → altar workflow
    if agent.inventoryBattery > 0:
      # Find altar and deposit battery
      for thing in env.things:
        if thing.kind == Altar and thing.pos == agent.homeAltar:
          let dist = manhattanDistance(agent.pos, thing.pos)
          if dist == 1:
            return [5'u8, getDirectionTo(agent.pos, thing.pos).uint8]
          else:
            return [1'u8, getMoveTowards(env, agent.pos, thing.pos, controller.rng).uint8]
    
    elif agent.inventoryOre > 0:
      # Find converter and make battery
      let converterThing = env.findNearestThing(agent.pos, Converter)
      if converterThing != nil:
        let dist = manhattanDistance(agent.pos, converterThing.pos)
        if dist == 1:
          return [5'u8, getDirectionTo(agent.pos, converterThing.pos).uint8]
        else:
          return [1'u8, getMoveTowards(env, agent.pos, converterThing.pos, controller.rng).uint8]
    
    else:
      # Find mine and collect ore
      let mine = env.findNearestThing(agent.pos, Mine)
      if mine != nil:
        let dist = manhattanDistance(agent.pos, mine.pos)
        if dist == 1:
          return [3'u8, getDirectionTo(agent.pos, mine.pos).uint8]
        else:
          return [1'u8, getMoveTowards(env, agent.pos, mine.pos, controller.rng).uint8]
  
  # Default: random movement to avoid getting stuck
  return [1'u8, controller.rng.rand(0..7).uint8]

# Compatibility function for updateController
proc updateController*(controller: Controller) =
  # No complex state to update - keep it simple
  discard