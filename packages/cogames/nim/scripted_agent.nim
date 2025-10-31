## cogames/nim/scripted_agent.nim
## -------------------------------------------------------------------------
## Standalone scripted agent for CoGames, inspired by tribal-village AI.
## Features:
##   * Hard-coded heart crafting workflow with prioritized behaviors.
##   * Zero-copy buffer hooks for Python ⇄ Nim interop.
##   * Shared map memory that grows as we explore (spiral frontier logic).
##   * Role negotiation via vibes plus multi-armed bandit resource routing.
##   * A* path-finding with optional mettascope integration.
## -------------------------------------------------------------------------

import std/[tables, sets, options, math, sequtils, times]
import rng_compat

when compiles(import vmath):
  import vmath
else:
  type
    IVec2* = object
      x*, y*: int32

    proc ivec2*(x, y: int32): IVec2 =
      result.x = x
      result.y = y

when compiles(import mettascope/pathfinding):
  import mettascope/pathfinding as mpf
  type PathResult = mpf.PathResult
else:
  type
    PathNode = object
      pos: IVec2
      gCost, hCost: float
      parent: int

    PathResult = object
      success: bool
      nodes: seq[IVec2]

    proc aStarFallback(startPos, goalPos: IVec2, passable: proc (p: IVec2): bool): PathResult =
      ## Lightweight A* placeholder; replace with mettascope once linked.
      var openSet = @[startPos]
      var cameFrom = initTable[IVec2, IVec2]()
      var gScore = initTable[IVec2, float]()
      var fScore = initTable[IVec2, float]()
      gScore[startPos] = 0.0
      fScore[startPos] = abs(float(goalPos.x - startPos.x)) + abs(float(goalPos.y - startPos.y))

      while openSet.len > 0:
        var currentIdx = 0
        var best = fScore.getOrDefault(openSet[0], Inf)
        for i, p in openSet:
          let score = fScore.getOrDefault(p, Inf)
          if score < best:
            best = score
            currentIdx = i

        let current = openSet[currentIdx]
        openSet.del(currentIdx)

        if current == goalPos:
          result.success = true
          result.nodes = @[current]
          var cursor = current
          while cursor in cameFrom:
            cursor = cameFrom[cursor]
            result.nodes.add(cursor)
          result.nodes.reverse()
          return

        const dirs = [ivec2(-1, 0), ivec2(1, 0), ivec2(0, -1), ivec2(0, 1)]
        for dir in dirs:
          let neighbor = ivec2(current.x + dir.x, current.y + dir.y)
          if not passable(neighbor):
            continue
          let tentative = gScore.getOrDefault(current, Inf) + 1.0
          if tentative < gScore.getOrDefault(neighbor, Inf):
            cameFrom[neighbor] = current
            gScore[neighbor] = tentative
            fScore[neighbor] = tentative + abs(float(goalPos.x - neighbor.x)) + abs(float(goalPos.y - neighbor.y))
            if neighbor notin openSet:
              openSet.add(neighbor)

      result.success = false
      result.nodes = @[]

## -------------------------------------------------------------------------
## Domain model
## -------------------------------------------------------------------------

type
  ResourceKind* = enum
    rkCarbon, rkOxygen, rkGermanium, rkSilicon, rkEnergy, rkHeart, rkUnknown

  StationKind* = enum
    skUnknown, skCharger, skMine, skRefinery, skAssembler, skChest

  TerrainKind* = enum
    tkUnknown, tkPassable, tkWall, tkClip, tkWater

  VibeSignal* = enum
    vibeNone,
    vibeCarbon,
    vibeOxygen,
    vibeGermanium,
    vibeSilicon,
    vibeEnergy

  RecipeRequirement = object
    resource: ResourceKind
    amount: int

  HeartRecipe* = object
    inputs*: seq[RecipeRequirement]
    cooldown*: int

  TileInfo = object
    terrain: TerrainKind
    station: StationKind
    lastSeenTick: int
    cooldownEnds: int

  FrontierNode = object
    pos: IVec2
    priority: float
    heuristic: float

  FrontierQueue = seq[FrontierNode]

  BanditArm = object
    label: string
    pulls: int
    totalReward: float
    explorationBonus: float

  SpiralState = object
    stepsInArc: int
    arcsCompleted: int
    origin: IVec2
    lastPoint: IVec2

  AgentRole* = object
    focus*: ResourceKind
    secondary*: Option[ResourceKind]
    vibe*: VibeSignal

  AgentMemory* = object
    initialized: bool
    role: AgentRole
    rng: Rand
    tick: int
    energyBudget: int
    map*: Table[IVec2, TileInfo]
    knownStations: Table[StationKind, seq[IVec2]]
    frontier: FrontierQueue
    spiral: SpiralState
    banditArms: seq[BanditArm]
    outstandingRecipe*: Table[ResourceKind, int]
    lastChoice: string
    lastActionCooldown: int
    heartbeat: int

  AgentAction* = object
    moveDir*: int        ## 0..7 as in tribal village (N, S, W, E, diagonals)
    interactDir*: int    ## -1 if no interaction
    emitVibe*: VibeSignal

  AgentView* = object
    id*: int
    pos*: IVec2
    energy*: int
    inventory*: Table[ResourceKind, int]
    cooldownRemaining*: int
    observedTiles*: seq[(IVec2, TerrainKind, StationKind, int)]
    teammates*: seq[int]
    sharedVibes*: Table[int, VibeSignal]

  EnvironmentView* = object
    tick*: int
    assemblerCooldown*: int
    heartCost*: int
    targetHearts*: int
    teamHeartInventory*: int

  Controller* = ref object
    rng*: Rand
    agents*: Table[int, AgentMemory]
    recipe*: HeartRecipe
    zeroCopyObsBuffer: pointer ## placeholders for python interop
    zeroCopyActBuffer: pointer

## -------------------------------------------------------------------------
## Utility helpers
## -------------------------------------------------------------------------

proc resourceToVibe(r: ResourceKind): VibeSignal =
  case r
  of rkCarbon: vibeCarbon
  of rkOxygen: vibeOxygen
  of rkGermanium: vibeGermanium
  of rkSilicon: vibeSilicon
  of rkEnergy: vibeEnergy
  else: vibeNone

proc vibeToResource(v: VibeSignal): ResourceKind =
  case v
  of vibeCarbon: rkCarbon
  of vibeOxygen: rkOxygen
  of vibeGermanium: rkGermanium
  of vibeSilicon: rkSilicon
  of vibeEnergy: rkEnergy
  else: rkUnknown

proc cumulativeAmount(t: Table[ResourceKind, int]): int =
  for _, amount in t:
    result += amount

proc ensureAgent(controller: Controller, agentId: int): var AgentMemory =
  if agentId notin controller.agents:
    let seed = controller.rng.rand(high(int32))
    var mem = AgentMemory(
      initialized: false,
      rng: initRand(seed),
      tick: 0,
      energyBudget: 0,
      map: initTable[IVec2, TileInfo](),
      knownStations: initTable[StationKind, seq[IVec2]](),
      frontier: @[],
      spiral: SpiralState(origin: ivec2(0, 0), lastPoint: ivec2(0, 0)),
      banditArms: @[],
      outstandingRecipe: initTable[ResourceKind, int](),
      lastActionCooldown: 0,
      heartbeat: 0
    )
    for kind in StationKind:
      mem.knownStations[kind] = @[]
    controller.agents[agentId] = mem
  return controller.agents[agentId]

proc pushFrontier(mem: var AgentMemory, pos: IVec2, heuristic: float) =
  let node = FrontierNode(pos: pos, priority: heuristic, heuristic: heuristic)
  mem.frontier.add(node)

proc popFrontier(mem: var AgentMemory): Option[IVec2] =
  if mem.frontier.len == 0:
    return none[IVec2]()
  var idx = 0
  var best = mem.frontier[0].priority
  for i in 1 .. mem.frontier.high:
    if mem.frontier[i].priority < best:
      best = mem.frontier[i].priority
      idx = i
  let node = mem.frontier[idx]
  mem.frontier.delete(idx)
  return some(node.pos)

proc updateBanditArm(mem: var AgentMemory, label: string, reward: float) =
  var idx = -1
  for i, arm in mem.banditArms:
    if arm.label == label:
      idx = i
      break
  if idx < 0:
    mem.banditArms.add(BanditArm(label: label, pulls: 0, totalReward: 0.0, explorationBonus: 2.0))
    idx = mem.banditArms.high
  mem.banditArms[idx].pulls.inc()
  mem.banditArms[idx].totalReward += reward

proc selectBanditArm(mem: var AgentMemory, defaultLabel: string): string =
  if mem.banditArms.len == 0:
    return defaultLabel
  let totalPulls = mem.banditArms.mapIt(it.pulls).foldl(a + b, 0)
  var bestLabel = defaultLabel
  var bestValue = -Inf
  for arm in mem.banditArms:
    var meanReward = if arm.pulls == 0: 0.0 else arm.totalReward / float(arm.pulls)
    var bonus = if totalPulls == 0: 0.0 else sqrt(arm.explorationBonus * ln(float(totalPulls + 1)) / float(arm.pulls + 1))
    let ucb = meanReward + bonus
    if ucb > bestValue:
      bestValue = ucb
      bestLabel = arm.label
  return bestLabel

proc resetSpiral(mem: var AgentMemory, origin: IVec2) =
  mem.spiral.origin = origin
  mem.spiral.lastPoint = origin
  mem.spiral.stepsInArc = 0
  mem.spiral.arcsCompleted = 0

proc nextSpiralPoint(mem: var AgentMemory): IVec2 =
  ## Generate expanding spiral in Manhattan lattice.
  if mem.spiral.stepsInArc == 0 and mem.spiral.arcsCompleted == 0:
    mem.spiral.stepsInArc = 1
    mem.spiral.lastPoint = mem.spiral.origin
    return mem.spiral.origin

  let directions = [ivec2(1, 0), ivec2(0, 1), ivec2(-1, 0), ivec2(0, -1)]
  let dirIndex = mem.spiral.arcsCompleted mod directions.len
  let currentDir = directions[dirIndex]
  let arcLength = (mem.spiral.arcsCompleted div 2) + 1
  if mem.spiral.stepsInArc > arcLength:
    mem.spiral.arcsCompleted.inc()
    mem.spiral.stepsInArc = 1
    return nextSpiralPoint(mem)

  mem.spiral.lastPoint = ivec2(
    mem.spiral.lastPoint.x + currentDir.x,
    mem.spiral.lastPoint.y + currentDir.y
  )
  mem.spiral.stepsInArc.inc()
  return mem.spiral.lastPoint

proc heuristic(a, b: IVec2): float =
  abs(float(a.x - b.x)) + abs(float(a.y - b.y))

## -------------------------------------------------------------------------
## Role negotiation via vibes
## -------------------------------------------------------------------------

proc chooseInitialRole(mem: var AgentMemory, agent: AgentView, env: EnvironmentView) =
  if mem.initialized:
    return

  let focusOptions = @[rkCarbon, rkOxygen, rkGermanium, rkSilicon]
  var claimed = initHashSet[ResourceKind]()
  for teammate in agent.teammates:
    if teammate in agent.sharedVibes:
      let res = vibeToResource(agent.sharedVibes[teammate])
      if res != rkUnknown:
        claimed.incl(res)

  var pool = focusOptions.filterIt(it notin claimed)
  if pool.len == 0:
    pool = focusOptions
  let pick = pool[mem.rng.rand(pool.high)]

  mem.role = AgentRole(focus: pick, secondary: some(rkEnergy), vibe: resourceToVibe(pick))
  mem.initialized = true
  mem.heartbeat = env.tick
  mem.energyBudget = max(agent.energy, 0)
  resetSpiral(mem, agent.pos)

proc updateRoleFromConflicts(mem: var AgentMemory, agent: AgentView) =
  var conflict = false
  for teammate, vibe in agent.sharedVibes:
    if teammate == agent.id:
      continue
    if vibeToResource(vibe) == mem.role.focus:
      conflict = true
      break
  if not conflict:
    return

  let focusOptions = @[rkCarbon, rkOxygen, rkGermanium, rkSilicon]
  var available = focusOptions.filterIt(it != mem.role.focus)
  if available.len == 0:
    return
  mem.role.focus = available[mem.rng.rand(available.high)]
  mem.role.vibe = resourceToVibe(mem.role.focus)

## -------------------------------------------------------------------------
## Map assimilation and knowledge tracking
## -------------------------------------------------------------------------

proc mergeObservation(mem: var AgentMemory, agent: AgentView, env: EnvironmentView) =
  mem.tick = env.tick

  for (pos, terrain, station, cooldown) in agent.observedTiles:
    mem.map[pos] = TileInfo(
      terrain: terrain,
      station: station,
      lastSeenTick: env.tick,
      cooldownEnds: env.tick + cooldown
    )
    if station != skUnknown:
      if station notin mem.knownStations:
        mem.knownStations[station] = @[pos]
      elif pos notin mem.knownStations[station]:
        mem.knownStations[station].add(pos)

  ## Update frontier around visible tiles
  for (pos, _, _, _) in agent.observedTiles:
    const dirs = [ivec2(1, 0), ivec2(-1, 0), ivec2(0, 1), ivec2(0, -1)]
    for dir in dirs:
      let neighbor = ivec2(pos.x + dir.x, pos.y + dir.y)
      if neighbor notin mem.map:
        let estimate = heuristic(agent.pos, neighbor)
        mem.pushFrontier(neighbor, estimate)

## -------------------------------------------------------------------------
## Prioritized behavior stack
## -------------------------------------------------------------------------

type
  BehaviorKind = enum
    bkResolveConflict,
    bkEmitVibe,
    bkAvoidClips,
    bkRechargeEnergy,
    bkDepositResources,
    bkCraftHeart,
    bkGatherRoleResource,
    bkGatherSecondaryResource,
    bkExploreFrontier,
    bkIdle

  BehaviorOutcome = object
    kind: BehaviorKind
    action: AgentAction
    valid: bool

proc defaultAction(): AgentAction =
  AgentAction(moveDir: -1, interactDir: -1, emitVibe: vibeNone)

proc actionWithMove(dir: int): AgentAction =
  AgentAction(moveDir: dir, interactDir: -1, emitVibe: vibeNone)

proc actionWithInteract(dir: int): AgentAction =
  AgentAction(moveDir: -1, interactDir: dir, emitVibe: vibeNone)

proc actionWithVibe(v: VibeSignal): AgentAction =
  AgentAction(moveDir: -1, interactDir: -1, emitVibe: v)

proc behaviorResolveConflict(mem: var AgentMemory, agent: AgentView): Option[BehaviorOutcome] =
  if not mem.initialized:
    return none[BehaviorOutcome]()
  var conflict = false
  for teammate, vibe in agent.sharedVibes:
    if teammate == agent.id:
      continue
    if vibeToResource(vibe) == mem.role.focus and teammate < agent.id:
      conflict = true
      break
  if conflict:
    updateRoleFromConflicts(mem, agent)
    return some(BehaviorOutcome(kind: bkResolveConflict, action: actionWithVibe(mem.role.vibe), valid: true))
  return none[BehaviorOutcome]()

proc behaviorEmitPersistentVibe(mem: AgentMemory): Option[BehaviorOutcome] =
  if mem.role.vibe == vibeNone:
    return none[BehaviorOutcome]()
  return some(BehaviorOutcome(kind: bkEmitVibe, action: actionWithVibe(mem.role.vibe), valid: true))

proc behaviorAvoidClips(mem: AgentMemory, agent: AgentView): Option[BehaviorOutcome] =
  if agent.cooldownRemaining > 0:
    return none[BehaviorOutcome]()
  let currentTile = mem.map.getOrDefault(agent.pos, TileInfo(terrain: tkPassable))
  if currentTile.terrain != tkClip:
    return none[BehaviorOutcome]()
  ## Move opposite of last drift
  let dirs = [actionWithMove(0), actionWithMove(1), actionWithMove(2), actionWithMove(3)]
  return some(BehaviorOutcome(kind: bkAvoidClips, action: dirs[mem.rng.rand(dirs.high)], valid: true))

proc checkNeedEnergy(mem: AgentMemory, agent: AgentView): bool =
  let threshold = max(10, mem.energyBudget div 4)
  agent.energy <= threshold

proc findNearestStation(mem: AgentMemory, agent: AgentView, kind: StationKind): Option[IVec2] =
  if kind notin mem.knownStations:
    return none[IVec2]()
  var bestPos: Option[IVec2]
  var bestDist = Inf
  for pos in mem.knownStations[kind]:
    let dist = heuristic(agent.pos, pos)
    if dist < bestDist:
      bestDist = dist
      bestPos = some(pos)
  return bestPos

proc directionToward(agentPos, target: IVec2, rng: var Rand): int =
  let dx = target.x - agentPos.x
  let dy = target.y - agentPos.y
  if dx == 0 and dy < 0: return 0
  if dx == 0 and dy > 0: return 1
  if dx < 0 and dy == 0: return 2
  if dx > 0 and dy == 0: return 3
  if dx < 0 and dy < 0: return 4
  if dx > 0 and dy < 0: return 5
  if dx < 0 and dy > 0: return 6
  if dx > 0 and dy > 0: return 7
  return rng.rand(0..7)

proc behaviorRecharge(mem: var AgentMemory, agent: AgentView): Option[BehaviorOutcome] =
  if not checkNeedEnergy(mem, agent):
    return none[BehaviorOutcome]()

  let station = mem.findNearestStation(agent, skCharger)
  if station.isSome:
    let target = station.get()
    if heuristic(agent.pos, target) <= 1:
      return some(BehaviorOutcome(kind: bkRechargeEnergy, action: actionWithInteract(directionToward(agent.pos, target, mem.rng)), valid: true))
    else:
      return some(BehaviorOutcome(kind: bkRechargeEnergy, action: actionWithMove(directionToward(agent.pos, target, mem.rng)), valid: true))

  let nextSpiral = mem.nextSpiralPoint()
  return some(BehaviorOutcome(kind: bkRechargeEnergy, action: actionWithMove(directionToward(agent.pos, nextSpiral, mem.rng)), valid: true))

proc behaviorDeposit(mem: var AgentMemory, agent: AgentView): Option[BehaviorOutcome] =
  if agent.inventory.getOrDefault(rkHeart, 0) > 0:
    let chest = mem.findNearestStation(agent, skChest)
    if chest.isSome:
      let target = chest.get()
      if heuristic(agent.pos, target) <= 1:
        return some(BehaviorOutcome(kind: bkDepositResources, action: actionWithInteract(directionToward(agent.pos, target, mem.rng)), valid: true))
      else:
        return some(BehaviorOutcome(kind: bkDepositResources, action: actionWithMove(directionToward(agent.pos, target, mem.rng)), valid: true))
  return none[BehaviorOutcome]()

proc behaviorCraftHeart(mem: var AgentMemory, agent: AgentView, env: EnvironmentView): Option[BehaviorOutcome] =
  if env.assemblerCooldown > env.tick:
    return none[BehaviorOutcome]()

  var requirements = initTable[ResourceKind, int]()
  for req in mem.recipe.inputs:
    requirements[req.resource] = requirements.getOrDefault(req.resource, 0) + req.amount
  var complete = true
  for req, needed in requirements.pairs:
    if agent.inventory.getOrDefault(req, 0) < needed:
      complete = false
      break
  if not complete:
    return none[BehaviorOutcome]()

  let assembler = mem.findNearestStation(agent, skAssembler)
  if assembler.isNone:
    return none[BehaviorOutcome]()
  let target = assembler.get()
  if heuristic(agent.pos, target) <= 1:
    return some(BehaviorOutcome(kind: bkCraftHeart, action: actionWithInteract(directionToward(agent.pos, target, mem.rng)), valid: true))
  else:
    return some(BehaviorOutcome(kind: bkCraftHeart, action: actionWithMove(directionToward(agent.pos, target, mem.rng)), valid: true))

proc selectResourceTarget(mem: var AgentMemory, resource: ResourceKind, agent: AgentView): Option[IVec2] =
  let label = $resource
  let targetLabel = mem.selectBanditArm(label)
  if targetLabel != label:
    for station in mem.knownStations[skChest]:
      if heuristic(agent.pos, station) < 10:
        return some(station)
  if skMine in mem.knownStations and mem.knownStations[skMine].len > 0:
    let idx = mem.rng.rand(0 ..< mem.knownStations[skMine].len)
    return some(mem.knownStations[skMine][idx])
  return none[IVec2]()

proc behaviorGather(mem: var AgentMemory, agent: AgentView, resource: ResourceKind): Option[BehaviorOutcome] =
  let target = mem.selectResourceTarget(resource, agent)
  if target.isSome:
    let pos = target.get()
    if heuristic(agent.pos, pos) <= 1:
      return some(BehaviorOutcome(kind: bkGatherRoleResource, action: actionWithInteract(directionToward(agent.pos, pos, mem.rng)), valid: true))
    else:
      return some(BehaviorOutcome(kind: bkGatherRoleResource, action: actionWithMove(directionToward(agent.pos, pos, mem.rng)), valid: true))
  return none[BehaviorOutcome]()

proc behaviorExplore(mem: var AgentMemory, agent: AgentView): Option[BehaviorOutcome] =
  let frontier = mem.popFrontier()
  if frontier.isSome:
    let goal = frontier.get()
    return some(BehaviorOutcome(kind: bkExploreFrontier, action: actionWithMove(directionToward(agent.pos, goal, mem.rng)), valid: true))
  let spiralPos = mem.nextSpiralPoint()
  return some(BehaviorOutcome(kind: bkExploreFrontier, action: actionWithMove(directionToward(agent.pos, spiralPos, mem.rng)), valid: true))

proc runPrioritizedBehaviors(mem: var AgentMemory, agent: AgentView, env: EnvironmentView): AgentAction =
  var attempt: Option[BehaviorOutcome]

  attempt = behaviorResolveConflict(mem, agent)
  if attempt.isSome and attempt.get().valid:
    return attempt.get().action

  attempt = behaviorEmitPersistentVibe(mem)
  if attempt.isSome and attempt.get().valid:
    return attempt.get().action

  attempt = behaviorAvoidClips(mem, agent)
  if attempt.isSome and attempt.get().valid:
    return attempt.get().action

  attempt = behaviorRecharge(mem, agent)
  if attempt.isSome and attempt.get().valid:
    return attempt.get().action

  attempt = behaviorDeposit(mem, agent)
  if attempt.isSome and attempt.get().valid:
    return attempt.get().action

  attempt = behaviorCraftHeart(mem, agent, env)
  if attempt.isSome and attempt.get().valid:
    return attempt.get().action

  attempt = behaviorGather(mem, agent, mem.role.focus)
  if attempt.isSome and attempt.get().valid:
    return attempt.get().action

  if mem.role.secondary.isSome:
    attempt = behaviorGather(mem, agent, mem.role.secondary.get())
    if attempt.isSome and attempt.get().valid:
      return attempt.get().action

  attempt = behaviorExplore(mem, agent)
  if attempt.isSome and attempt.get().valid:
    return attempt.get().action

  return defaultAction()

## -------------------------------------------------------------------------
## Controller lifecycle
## -------------------------------------------------------------------------

proc initController*(seed = 0, heartCost = 10): Controller =
  let nowSeed = if seed == 0: int(epochTime() * 1000.0) else: seed
  let rng = initRand(nowSeed)
  let inputs = @[
    RecipeRequirement(resource: rkCarbon, amount: heartCost * 2),
    RecipeRequirement(resource: rkOxygen, amount: heartCost * 2),
    RecipeRequirement(resource: rkGermanium, amount: max(heartCost div 2, 1)),
    RecipeRequirement(resource: rkSilicon, amount: heartCost * 5),
    RecipeRequirement(resource: rkEnergy, amount: heartCost * 2)
  ]
  Controller(
    rng: rng,
    agents: initTable[int, AgentMemory](),
    recipe: HeartRecipe(inputs: inputs, cooldown: 4),
    zeroCopyObsBuffer: nil,
    zeroCopyActBuffer: nil
  )

proc registerSharedBuffers*(controller: Controller, observations, actions: pointer) =
  controller.zeroCopyObsBuffer = observations
  controller.zeroCopyActBuffer = actions

proc controllerStep*(controller: Controller, agent: AgentView, env: EnvironmentView): AgentAction =
  var mem = controller.ensureAgent(agent.id)
  if not mem.initialized:
    mem.chooseInitialRole(agent, env)

  mem.mergeObservation(agent, env)
  let action = runPrioritizedBehaviors(mem, agent, env)
  mem.lastChoice = $action.moveDir & ":" & $action.interactDir
  mem.lastActionCooldown = agent.cooldownRemaining
  mem.heartbeat = env.tick
  return action

proc controllerReset*(controller: Controller) =
  controller.agents.clear()

## -------------------------------------------------------------------------
## C export hooks for Python FFI
## -------------------------------------------------------------------------

type
  AgentOpaque* = pointer

var globalController: Controller

proc cogames_agent_create*(seed: int32, heartCost: int32): AgentOpaque {.exportc, dynlib.} =
  globalController = initController(seed, heartCost)
  return cast[AgentOpaque](globalController.addr)

proc cogames_agent_reset*() {.exportc, dynlib.} =
  if globalController != nil:
    globalController.controllerReset()

proc cogames_agent_register_buffers*(observations, actions: pointer) {.exportc, dynlib.} =
  if globalController != nil:
    globalController.registerSharedBuffers(observations, actions)

proc cogames_agent_step*(agentId: int32, viewPtr: pointer, envPtr: pointer): AgentAction {.exportc, dynlib.} =
  ## NOTE: viewPtr/envPtr are expected to be marshalled by Python into AgentView/EnvironmentView.
  ## This stub casts them directly; replace with actual zero-copy structs when interface stabilizes.
  if globalController == nil:
    return defaultAction()
  let agent = cast[ptr AgentView](viewPtr)
  let env = cast[ptr EnvironmentView](envPtr)
  return globalController.controllerStep(agent[], env[])
