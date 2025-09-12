## Simplified AI replacement - dramatically reduced complexity
import std/[random, tables]
import vmath
import environment, common

type
  SimpleRole = enum
    AltarWorker, LanternMaker, ArmorMaker, SpearMaker, BreadMaker
  
  SimpleState = object
    role: SimpleRole
    initialized: bool

var
  rng = initRand()
  agentStates: Table[int, SimpleState]

proc findNearest(env: Environment, agentPos: IVec2, targetKind: ThingKind): Thing =
  result = nil
  var minDist = 999999
  for thing in env.things:
    if thing.kind == targetKind:
      let dist = abs(thing.pos.x - agentPos.x) + abs(thing.pos.y - agentPos.y)
      if dist < minDist:
        minDist = dist
        result = thing

proc findNearestTerrain(env: Environment, agentPos: IVec2, terrain: TerrainType): IVec2 =
  result = ivec2(-1, -1)
  var minDist = 999999
  for x in 0..<MapWidth:
    for y in 0..<MapHeight:
      if env.terrain[x][y] == terrain:
        let pos = ivec2(x.int32, y.int32)
        let dist = abs(pos.x - agentPos.x) + abs(pos.y - agentPos.y)
        if dist < minDist:
          minDist = dist
          result = pos

proc getDirection(fromPos, toPos: IVec2): int =
  ## Get orientation (0=N, 1=S, 2=W, 3=E) from fromPos to toPos
  let dx = toPos.x - fromPos.x
  let dy = toPos.y - fromPos.y
  if abs(dx) > abs(dy):
    if dx > 0: 3 else: 2  # East or West
  else:
    if dy > 0: 1 else: 0  # South or North

proc simpleDecideAction*(env: Environment, agentId: int): array[2, uint8] =
  let agent = env.agents[agentId]
  
  # Initialize agent if needed
  if agentId notin agentStates:
    let role = case agentId mod 5:
      of 0: AltarWorker
      of 1: ArmorMaker
      of 2: SpearMaker  
      of 3: BreadMaker
      of 4: LanternMaker
      else: AltarWorker
    agentStates[agentId] = SimpleState(role: role, initialized: true)
  
  let state = agentStates[agentId]
  
  case state.role:
  of LanternMaker:
    if agent.inventoryLantern > 0:
      # Plant lantern
      return [6'u8, 0'u8]
    elif agent.inventoryWheat > 0:
      # Find WeavingLoom and craft lantern
      let loom = env.findNearest(agent.pos, WeavingLoom)
      if loom != nil:
        let dist = abs(loom.pos.x - agent.pos.x) + abs(loom.pos.y - agent.pos.y)
        if dist == 1:
          # Adjacent to loom, craft
          return [5'u8, getDirection(agent.pos, loom.pos).uint8]
        else:
          # Move toward loom
          return [1'u8, getDirection(agent.pos, loom.pos).uint8]
    else:
      # Find wheat and collect it
      let wheatPos = env.findNearestTerrain(agent.pos, Wheat)
      if wheatPos.x >= 0:
        let dist = abs(wheatPos.x - agent.pos.x) + abs(wheatPos.y - agent.pos.y)
        if dist == 1:
          # Adjacent to wheat, harvest
          return [3'u8, getDirection(agent.pos, wheatPos).uint8]
        else:
          # Move toward wheat
          return [1'u8, getDirection(agent.pos, wheatPos).uint8]
  
  of ArmorMaker:
    if agent.inventoryWood > 0:
      # Find Armory and craft armor
      let armory = env.findNearest(agent.pos, Armory)
      if armory != nil:
        let dist = abs(armory.pos.x - agent.pos.x) + abs(armory.pos.y - agent.pos.y)
        if dist == 1:
          return [5'u8, getDirection(agent.pos, armory.pos).uint8]
        else:
          return [1'u8, getDirection(agent.pos, armory.pos).uint8]
    else:
      # Find wood
      let treePos = env.findNearestTerrain(agent.pos, Tree)
      if treePos.x >= 0:
        let dist = abs(treePos.x - agent.pos.x) + abs(treePos.y - agent.pos.y)
        if dist == 1:
          return [3'u8, getDirection(agent.pos, treePos).uint8]
        else:
          return [1'u8, getDirection(agent.pos, treePos).uint8]
  
  of SpearMaker:
    if agent.inventorySpear > 0:
      # Hunt clippies
      let clippy = env.findNearest(agent.pos, Clippy)
      if clippy != nil:
        let dist = abs(clippy.pos.x - agent.pos.x) + abs(clippy.pos.y - agent.pos.y)
        if dist <= 2 and dist >= 1:  # Spear range
          return [2'u8, getDirection(agent.pos, clippy.pos).uint8]
        else:
          return [1'u8, getDirection(agent.pos, clippy.pos).uint8]
    elif agent.inventoryWood > 0:
      # Find Forge and craft spear
      let forge = env.findNearest(agent.pos, Forge)
      if forge != nil:
        let dist = abs(forge.pos.x - agent.pos.x) + abs(forge.pos.y - agent.pos.y)
        if dist == 1:
          return [5'u8, getDirection(agent.pos, forge.pos).uint8]
        else:
          return [1'u8, getDirection(agent.pos, forge.pos).uint8]
    else:
      # Find wood
      let treePos = env.findNearestTerrain(agent.pos, Tree)
      if treePos.x >= 0:
        let dist = abs(treePos.x - agent.pos.x) + abs(treePos.y - agent.pos.y)
        if dist == 1:
          return [3'u8, getDirection(agent.pos, treePos).uint8]
        else:
          return [1'u8, getDirection(agent.pos, treePos).uint8]
  
  else:
    discard  # Other roles can be added
  
  # Default: random movement to prevent getting stuck
  return [1'u8, rng.rand(0..3).uint8]