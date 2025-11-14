import
  std/[algorithm, deques, options, random, sequtils, sets, strformat, strutils, tables],
  genny, jsony,
  common

type
  Phase = enum
    gatherPhase,
    assemblePhase,
    deliverPhase,
    rechargePhase

  ObservedObject = object
    name: string
    converting: bool
    cooldownRemaining: int
    clipped: bool
    remainingUses: int
    protocolInputs: Table[string, int]
    protocolOutputs: Table[string, int]

  ExtractorInfo = object
    position: Location
    resource: string
    lastSeen: int
    converting: bool
    cooldownRemaining: int
    clipped: bool
    remainingUses: int

  LadybugState = object
    row: int
    col: int
    mapHeight: int
    mapWidth: int
    obsHalfWidth: int
    obsHalfHeight: int
    occupancy: seq[seq[int]]
    agentOccupancy: HashSet[Location]
    phase: Phase
    stepCount: int
    positionHistory: seq[Location]
    energy: int
    carbon: int
    oxygen: int
    germanium: int
    silicon: int
    hearts: int
    decoder: int
    modulator: int
    resonator: int
    scrambler: int
    heartRecipe: Table[string, int]
    stations: Table[string, Option[Location]]
    extractors: Table[string, seq[ExtractorInfo]]
    currentGlyph: string
    targetResource: string
    pendingUseResource: string
    pendingUseAmount: int
    waitingAtExtractor: Option[Location]
    waitSteps: int
    usingObjectThisStep: bool
    lastAction: int
    explorationDirection: string
    explorationDirectionSetStep: int
    explorationEscapeUntilStep: int

  LadybugAgent* = ref object
    agentId*: int
    cfg*: Config
    random*: Rand
    state*: LadybugState

  LadybugPolicy* = ref object
    agents*: seq[LadybugAgent]

const
  defaultMapSize = 200
  rechargeThresholdLow = 35
  rechargeThresholdHigh = 85
  positionHistorySize = 40
  explorationAreaCheckWindow = 35
  explorationAreaSizeThreshold = 9
  explorationEscapeDuration = 8
  explorationDirectionPersistence = 18
  explorationAssemblerDistanceThreshold = 12
  cellFree = 1
  cellObstacle = 2
  parentSentinel = (-9999, -9999)

proc initState(agent: LadybugAgent) =
  agent.state = LadybugState()
  agent.state.mapHeight = defaultMapSize
  agent.state.mapWidth = defaultMapSize
  agent.state.obsHalfWidth = agent.cfg.config.obsWidth div 2
  agent.state.obsHalfHeight = agent.cfg.config.obsHeight div 2
  agent.state.row = defaultMapSize div 2
  agent.state.col = defaultMapSize div 2
  agent.state.phase = gatherPhase
  agent.state.currentGlyph = "default"
  agent.state.targetResource = ""
  agent.state.heartRecipe = initTable[string, int]()
  agent.state.stations = initTable[string, Option[Location]]()
  for key in ["assembler", "chest", "charger"]:
    agent.state.stations[key] = none(Location)
  agent.state.extractors = initTable[string, seq[ExtractorInfo]]()
  for resource in ["carbon", "oxygen", "germanium", "silicon"]:
    agent.state.extractors[resource] = @[]
  agent.state.agentOccupancy = initHashSet[Location]()
  agent.state.occupancy = newSeqWith(
    agent.state.mapHeight,
    newSeqWith(agent.state.mapWidth, cellFree.int)
  )
  agent.state.lastAction = agent.cfg.actions.noop
  agent.state.explorationDirection = ""
  agent.state.explorationDirectionSetStep = 0
  agent.state.explorationEscapeUntilStep = 0
  agent.state.positionHistory = @[]
  agent.state.waitSteps = 0
  agent.state.waitingAtExtractor = none(Location)
  agent.state.usingObjectThisStep = false

proc resourceVibe(resource: string): string =
  ## Map canonical resource names to the actual vibe glyphs used by the engine.
  case resource.toLowerAscii()
  of "carbon":
    "carbon_a"
  of "oxygen":
    "oxygen_a"
  of "germanium":
    "germanium_a"
  of "silicon":
    "silicon_a"
  else:
    "carbon_a"

proc newLadybugAgent*(agentId: int, environmentConfig: string): LadybugAgent {.raises: [].} =
  var config = parseConfig(environmentConfig)
  result = LadybugAgent(agentId: agentId, cfg: config)
  result.random = initRand(agentId)
  initState(result)

proc reset*(agent: LadybugAgent) =
  initState(agent)

proc decodeObservation(
  agent: LadybugAgent,
  numTokens: int,
  sizeToken: int,
  rawObservation: pointer
): Table[Location, seq[FeatureValue]] =
  let observations = cast[ptr UncheckedArray[uint8]](rawObservation)
  for token in 0 ..< numTokens:
    let baseIdx = token * sizeToken
    let locationPacked = observations[baseIdx]
    let featureId = observations[baseIdx + 1]
    let value = observations[baseIdx + 2]
    if locationPacked == 255 and featureId == 255 and value == 255:
      break
    var location: Location
    if locationPacked != 0xFF:
      location.y = (locationPacked shr 4).int - agent.state.obsHalfHeight
      location.x = (locationPacked and 0x0F).int - agent.state.obsHalfWidth
    if location notin result:
      result[location] = @[]
    result[location].add(FeatureValue(featureId: featureId.int, value: value.int))

proc worldLocation(agent: LadybugAgent, local: Location): Option[Location] =
  let worldRow = agent.state.row + local.y
  let worldCol = agent.state.col + local.x
  if worldRow < 0 or worldRow >= agent.state.mapHeight:
    return none(Location)
  if worldCol < 0 or worldCol >= agent.state.mapWidth:
    return none(Location)
  some(Location(x: worldCol, y: worldRow))

proc updateInventory(agent: LadybugAgent, visible: Table[Location, seq[FeatureValue]]) =
  agent.state.energy = agent.cfg.getInventory(visible, agent.cfg.features.invEnergy)
  agent.state.carbon = agent.cfg.getInventory(visible, agent.cfg.features.invCarbon)
  agent.state.oxygen = agent.cfg.getInventory(visible, agent.cfg.features.invOxygen)
  agent.state.germanium = agent.cfg.getInventory(visible, agent.cfg.features.invGermanium)
  agent.state.silicon = agent.cfg.getInventory(visible, agent.cfg.features.invSilicon)
  agent.state.hearts = agent.cfg.getInventory(visible, agent.cfg.features.invHeart)
  agent.state.decoder = agent.cfg.getInventory(visible, agent.cfg.features.invDecoder)
  agent.state.modulator = agent.cfg.getInventory(visible, agent.cfg.features.invModulator)
  agent.state.resonator = agent.cfg.getInventory(visible, agent.cfg.features.invResonator)
  agent.state.scrambler = agent.cfg.getInventory(visible, agent.cfg.features.invScrambler)

proc buildObservedObject(agent: LadybugAgent, features: seq[FeatureValue]): ObservedObject =
  result.protocolInputs = initTable[string, int]()
  result.protocolOutputs = initTable[string, int]()
  for fv in features:
    if fv.featureId == agent.cfg.features.tag:
      result.name = agent.cfg.tagName(fv.value)
    elif fv.featureId == agent.cfg.features.converting:
      result.converting = fv.value > 0
    elif fv.featureId == agent.cfg.features.cooldownRemaining:
      result.cooldownRemaining = fv.value
    elif fv.featureId == agent.cfg.features.clipped:
      result.clipped = fv.value > 0
    elif fv.featureId == agent.cfg.features.remainingUses:
      result.remainingUses = fv.value
    else:
      for resource, id in agent.cfg.features.protocolInputs.pairs:
        if fv.featureId == id:
          result.protocolInputs[resource] = fv.value
      for resource, id in agent.cfg.features.protocolOutputs.pairs:
        if fv.featureId == id:
          result.protocolOutputs[resource] = fv.value

proc discoverStation(agent: LadybugAgent, key: string, loc: Location) =
  if key notin agent.state.stations:
    agent.state.stations[key] = some(loc)
    return
  if agent.state.stations[key].isNone:
    agent.state.stations[key] = some(loc)

proc discoverExtractor(
  agent: LadybugAgent,
  resource: string,
  loc: Location,
  obj: ObservedObject
) =
  var items = agent.state.extractors.getOrDefault(resource, @[])
  var found = false
  for i in 0 ..< items.len:
    if items[i].position == loc:
      items[i].lastSeen = agent.state.stepCount
      items[i].converting = obj.converting
      items[i].cooldownRemaining = obj.cooldownRemaining
      items[i].clipped = obj.clipped
      if obj.remainingUses != 0:
        items[i].remainingUses = obj.remainingUses
      found = true
      break
  if not found:
    var info = ExtractorInfo(
      position: loc,
      resource: resource,
      lastSeen: agent.state.stepCount,
      converting: obj.converting,
      cooldownRemaining: obj.cooldownRemaining,
      clipped: obj.clipped,
      remainingUses: (if obj.remainingUses == 0: 999 else: obj.remainingUses)
    )
    items.add(info)
  agent.state.extractors[resource] = items

proc discoverObjects(agent: LadybugAgent, visible: Table[Location, seq[FeatureValue]]) =
  agent.state.agentOccupancy.clear()
  for dy in -agent.state.obsHalfHeight .. agent.state.obsHalfHeight:
    for dx in -agent.state.obsHalfWidth .. agent.state.obsHalfWidth:
      let local = Location(x: dx, y: dy)
      let world = agent.worldLocation(local)
      if world.isSome:
        agent.state.occupancy[world.get().y][world.get().x] = cellFree

  for local, features in visible.pairs:
    if local.x == 0 and local.y == 0:
      continue
    let worldOpt = agent.worldLocation(local)
    if worldOpt.isNone:
      continue
    let world = worldOpt.get()
    var obj = agent.buildObservedObject(features)
    if obj.name.len == 0:
      continue
    let lowerName = obj.name.toLowerAscii()

    if lowerName == "agent":
      agent.state.agentOccupancy.incl(world)
      continue

    if lowerName.contains("wall"):
      agent.state.occupancy[world.y][world.x] = cellObstacle
      continue

    if lowerName.contains("assembler"):
      agent.state.occupancy[world.y][world.x] = cellObstacle
      discoverStation(agent, "assembler", world)
      if agent.state.heartRecipe.len == 0 and obj.protocolOutputs.getOrDefault("heart", 0) > 0:
        agent.state.heartRecipe = initTable[string, int]()
        for resource, amount in obj.protocolInputs.pairs:
          if resource != "energy":
            agent.state.heartRecipe[resource] = amount
      continue

    if lowerName.contains("chest"):
      agent.state.occupancy[world.y][world.x] = cellObstacle
      discoverStation(agent, "chest", world)
      continue

    if lowerName.contains("charger"):
      agent.state.occupancy[world.y][world.x] = cellObstacle
      discoverStation(agent, "charger", world)
      continue

    if lowerName.contains("extractor"):
      agent.state.occupancy[world.y][world.x] = cellObstacle
      var resource = lowerName
      if resource.startsWith("clipped_"):
        resource = resource[8 .. ^1]
      if resource.endsWith("_extractor"):
        resource = resource[0 ..< resource.len - "_extractor".len]
      discoverExtractor(agent, resource, world, obj)

proc updatePosition(agent: LadybugAgent) =
  if agent.state.usingObjectThisStep:
    agent.state.usingObjectThisStep = false
    return
  if agent.state.lastAction == agent.cfg.actions.moveNorth:
    dec agent.state.row
  elif agent.state.lastAction == agent.cfg.actions.moveSouth:
    inc agent.state.row
  elif agent.state.lastAction == agent.cfg.actions.moveEast:
    inc agent.state.col
  elif agent.state.lastAction == agent.cfg.actions.moveWest:
    dec agent.state.col
  agent.state.positionHistory.add(Location(x: agent.state.col, y: agent.state.row))
  if agent.state.positionHistory.len > positionHistorySize:
    agent.state.positionHistory.delete(0)

proc clearWaiting(agent: LadybugAgent) =
  agent.state.waitingAtExtractor = none(Location)
  agent.state.pendingUseResource = ""
  agent.state.pendingUseAmount = 0
  agent.state.waitSteps = 0

proc getInventoryValue(agent: LadybugAgent, resource: string): int =
  case resource
  of "carbon": result = agent.state.carbon
  of "oxygen": result = agent.state.oxygen
  of "germanium": result = agent.state.germanium
  of "silicon": result = agent.state.silicon
  of "heart": result = agent.state.hearts
  else: result = 0

proc findExtractor(
  agent: LadybugAgent,
  resource: string,
  loc: Location
): Option[ExtractorInfo] =
  if resource notin agent.state.extractors:
    return none(ExtractorInfo)
  for info in agent.state.extractors[resource]:
    if info.position == loc:
      return some(info)
  none(ExtractorInfo)

proc handleWaiting(agent: LadybugAgent): Option[int] =
  if agent.state.pendingUseResource.len == 0 or agent.state.waitingAtExtractor.isNone:
    return none(int)

  let resource = agent.state.pendingUseResource
  let waitingLoc = agent.state.waitingAtExtractor.get()
  if agent.getInventoryValue(resource) > agent.state.pendingUseAmount:
    clearWaiting(agent)
    return none(int)

  var maxWait = 20
  let infoOpt = agent.findExtractor(resource, waitingLoc)
  if infoOpt.isSome():
    maxWait = infoOpt.get().cooldownRemaining + 5

  inc agent.state.waitSteps
  if agent.state.waitSteps > maxWait:
    clearWaiting(agent)
    return none(int)

  some(agent.cfg.actions.noop)

proc calculateDeficits(agent: LadybugAgent): Table[string, int] =
  result = initTable[string, int]()
  for resource in ["carbon", "oxygen", "germanium", "silicon"]:
    let required = agent.state.heartRecipe.getOrDefault(resource, 0)
    let deficit = required - agent.getInventoryValue(resource)
    result[resource] = (if deficit > 0: deficit else: 0)

proc findNearestExtractor(agent: LadybugAgent, resource: string): Option[ExtractorInfo] =
  if resource notin agent.state.extractors:
    return none(ExtractorInfo)
  var bestDist = high(int)
  var best: ExtractorInfo
  for info in agent.state.extractors[resource]:
    if info.clipped or info.remainingUses == 0:
      continue
    let dist = abs(info.position.x - agent.state.col) + abs(info.position.y - agent.state.row)
    if dist < bestDist:
      bestDist = dist
      best = info
  if bestDist == high(int):
    return none(ExtractorInfo)
  some(best)

proc isWithinBounds(agent: LadybugAgent, row: int, col: int): bool =
  row >= 0 and row < agent.state.mapHeight and col >= 0 and col < agent.state.mapWidth

proc isAgentBlocking(agent: LadybugAgent, row: int, col: int): bool =
  let loc = Location(x: col, y: row)
  loc in agent.state.agentOccupancy

proc isTraversable(agent: LadybugAgent, row: int, col: int): bool =
  if not agent.isWithinBounds(row, col):
    return false
  if agent.isAgentBlocking(row, col):
    return false
  agent.state.occupancy[row][col] == cellFree

proc isAdjacent(row1: int, col1: int, row2: int, col2: int): bool =
  let dr = abs(row1 - row2)
  let dc = abs(col1 - col2)
  (dr == 1 and dc == 0) or (dr == 0 and dc == 1)

proc computeGoalCells(
  agent: LadybugAgent,
  targetRow: int,
  targetCol: int,
  reachAdjacent: bool
): seq[(int, int)] =
  if not reachAdjacent:
    return @[(targetRow, targetCol)]
  for (dr, dc) in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
    let nr = targetRow + dr
    let nc = targetCol + dc
    if agent.isTraversable(nr, nc):
      result.add((nr, nc))
  if result.len == 0:
    for (dr, dc) in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
      let nr = targetRow + dr
      let nc = targetCol + dc
      if agent.isWithinBounds(nr, nc):
        result.add((nr, nc))

proc reconstructPath(
  cameFrom: Table[(int, int), (int, int)],
  current: (int, int)
): seq[(int, int)] =
  var cur = current
  while cameFrom.hasKey(cur):
    let prev = cameFrom[cur]
    if prev == parentSentinel:
      break
    result.add(cur)
    cur = prev
  result.reverse()

proc moveTowards(
  agent: LadybugAgent,
  targetRow: int,
  targetCol: int,
  reachAdjacent: bool = false,
  allowGoalBlock: bool = false
): int =
  if not reachAdjacent and agent.state.row == targetRow and agent.state.col == targetCol:
    return agent.cfg.actions.noop
  let goals = agent.computeGoalCells(targetRow, targetCol, reachAdjacent)
  if goals.len == 0:
    return agent.cfg.actions.noop
  var queue = initDeque[(int, int)]()
  var cameFrom = initTable[(int, int), (int, int)]()
  let start = (agent.state.row, agent.state.col)
  queue.addLast(start)
  cameFrom[start] = parentSentinel
  let goalSet = goals.toHashSet
  while queue.len > 0:
    let current = queue.popFirst()
    if current in goalSet:
      let path = reconstructPath(cameFrom, current)
      if path.len == 0:
        return agent.cfg.actions.noop
      let nextStep = path[0]
      let dr = nextStep[0] - agent.state.row
      let dc = nextStep[1] - agent.state.col
      if dr == -1 and dc == 0:
        return agent.cfg.actions.moveNorth
      if dr == 1 and dc == 0:
        return agent.cfg.actions.moveSouth
      if dr == 0 and dc == 1:
        return agent.cfg.actions.moveEast
      if dr == 0 and dc == -1:
        return agent.cfg.actions.moveWest
      return agent.cfg.actions.noop
    for (dr, dc) in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
      let nr = current[0] + dr
      let nc = current[1] + dc
      if cameFrom.hasKey((nr, nc)):
        continue
      var canTraverse = agent.isTraversable(nr, nc)
      if not canTraverse and allowGoalBlock and (nr, nc) in goalSet:
        canTraverse = true
      if not canTraverse:
        continue
      cameFrom[(nr, nc)] = current
      queue.addLast((nr, nc))
  agent.cfg.actions.noop

proc directionDelta(direction: string): (int, int) =
  case direction
  of "north": (-1, 0)
  of "south": (1, 0)
  of "east": (0, 1)
  of "west": (0, -1)
  else: (0, 0)

proc directionAction(agent: LadybugAgent, direction: string): int =
  case direction
  of "north": agent.cfg.actions.moveNorth
  of "south": agent.cfg.actions.moveSouth
  of "east": agent.cfg.actions.moveEast
  of "west": agent.cfg.actions.moveWest
  else: agent.cfg.actions.noop

proc tryRandomDirection(agent: LadybugAgent): int =
  var dirs = @["north", "south", "east", "west"]
  agent.random.shuffle(dirs)
  for dir in dirs:
    let delta = directionDelta(dir)
    let nr = agent.state.row + delta[0]
    let nc = agent.state.col + delta[1]
    if agent.isTraversable(nr, nc):
      return agent.directionAction(dir)
  agent.cfg.actions.noop

proc explore(agent: LadybugAgent): int =
  if agent.state.explorationDirection.len > 0:
    let steps = agent.state.stepCount - agent.state.explorationDirectionSetStep
    if steps >= explorationDirectionPersistence:
      agent.state.explorationDirection = ""

  # Escape mode: navigate toward assembler if stuck recently
  if agent.state.explorationEscapeUntilStep > agent.state.stepCount:
    agent.state.explorationDirection = ""
    if agent.state.stations["assembler"].isSome():
      let assemblerLoc = agent.state.stations["assembler"].get()
      if isAdjacent(agent.state.row, agent.state.col, assemblerLoc.y, assemblerLoc.x):
        agent.state.explorationEscapeUntilStep = 0
      else:
        return agent.moveTowards(assemblerLoc.y, assemblerLoc.x, reachAdjacent = true)
    else:
      agent.state.explorationEscapeUntilStep = 0
  else:
    # Check if we've stayed within a small area recently; if so, trigger escape
    let historyLen = agent.state.positionHistory.len
    if historyLen >= explorationAreaCheckWindow and agent.state.stations["assembler"].isSome():
      var minRow = high(int)
      var maxRow = low(int)
      var minCol = high(int)
      var maxCol = low(int)
      for i in (historyLen - explorationAreaCheckWindow) ..< historyLen:
        let pos = agent.state.positionHistory[i]
        if pos.y < minRow:
          minRow = pos.y
        if pos.y > maxRow:
          maxRow = pos.y
        if pos.x < minCol:
          minCol = pos.x
        if pos.x > maxCol:
          maxCol = pos.x
      let areaHeight = maxRow - minRow + 1
      let areaWidth = maxCol - minCol + 1
      if areaHeight <= explorationAreaSizeThreshold and areaWidth <= explorationAreaSizeThreshold:
        let assemblerLoc = agent.state.stations["assembler"].get()
        let dist = abs(agent.state.row - assemblerLoc.y) + abs(agent.state.col - assemblerLoc.x)
        if dist > explorationAssemblerDistanceThreshold:
          agent.state.explorationEscapeUntilStep = agent.state.stepCount + explorationEscapeDuration
          agent.state.explorationDirection = ""
          return agent.moveTowards(assemblerLoc.y, assemblerLoc.x, reachAdjacent = true)

  if agent.state.explorationDirection.len > 0:
    let delta = directionDelta(agent.state.explorationDirection)
    let nr = agent.state.row + delta[0]
    let nc = agent.state.col + delta[1]
    if agent.isTraversable(nr, nc):
      return agent.directionAction(agent.state.explorationDirection)
    agent.state.explorationDirection = ""

  var dirs = @["north", "south", "east", "west"]
  agent.random.shuffle(dirs)
  for dir in dirs:
    let delta = directionDelta(dir)
    let nr = agent.state.row + delta[0]
    let nc = agent.state.col + delta[1]
    if agent.isTraversable(nr, nc):
      agent.state.explorationDirection = dir
      agent.state.explorationDirectionSetStep = agent.state.stepCount
      return agent.directionAction(dir)
  return agent.tryRandomDirection()

proc navigateToAdjacent(
  agent: LadybugAgent,
  targetRow: int,
  targetCol: int
): Option[int] =
  if isAdjacent(agent.state.row, agent.state.col, targetRow, targetCol):
    return none(int)
  let action = agent.moveTowards(targetRow, targetCol, reachAdjacent = true)
  if action == agent.cfg.actions.noop:
    return some(agent.explore())
  some(action)

proc moveIntoCell(agent: LadybugAgent, targetRow: int, targetCol: int): int =
  agent.state.usingObjectThisStep = true
  return agent.moveTowards(targetRow, targetCol, allowGoalBlock = true)

proc useExtractor(agent: LadybugAgent, extractor: ExtractorInfo): int =
  if extractor.cooldownRemaining > 0 or extractor.converting:
    agent.state.waitingAtExtractor = some(extractor.position)
    inc agent.state.waitSteps
    return agent.cfg.actions.noop
  if extractor.remainingUses == 0 or extractor.clipped:
    clearWaiting(agent)
    return agent.cfg.actions.noop
  agent.state.pendingUseResource = extractor.resource
  agent.state.pendingUseAmount = agent.getInventoryValue(extractor.resource)
  agent.state.waitingAtExtractor = some(extractor.position)
  agent.state.waitSteps = 0
  return agent.moveIntoCell(extractor.position.y, extractor.position.x)

proc updatePhase(agent: LadybugAgent) =
  if agent.state.energy < rechargeThresholdLow:
    agent.state.phase = rechargePhase
    clearWaiting(agent)
    return
  if agent.state.phase == rechargePhase and agent.state.energy < rechargeThresholdHigh:
    return
  if agent.state.hearts > 0:
    agent.state.phase = deliverPhase
    clearWaiting(agent)
    return
  if agent.state.heartRecipe.len > 0:
    var ready = true
    for resource, amount in agent.state.heartRecipe.pairs:
      if agent.getInventoryValue(resource) < amount:
        ready = false
        break
    if ready:
      agent.state.phase = assemblePhase
      clearWaiting(agent)
      return
  agent.state.phase = gatherPhase

proc desiredVibe(agent: LadybugAgent): string =
  case agent.state.phase
  of gatherPhase:
    if agent.state.targetResource.len > 0:
      return resourceVibe(agent.state.targetResource)
    return "carbon_a"
  of assemblePhase:
    return "heart_a"
  of deliverPhase:
    return "default"
  of rechargePhase:
    return "charger"

proc vibeAction(agent: LadybugAgent, vibe: string): int =
  case vibe
  of "carbon_a": agent.cfg.actions.vibeCarbonA
  of "carbon_b": agent.cfg.actions.vibeCarbonB
  of "oxygen_a": agent.cfg.actions.vibeOxygenA
  of "oxygen_b": agent.cfg.actions.vibeOxygenB
  of "germanium_a": agent.cfg.actions.vibeGermaniumA
  of "germanium_b": agent.cfg.actions.vibeGermaniumB
  of "silicon_a": agent.cfg.actions.vibeSiliconA
  of "silicon_b": agent.cfg.actions.vibeSiliconB
  of "heart_a": agent.cfg.actions.vibeHeartA
  of "heart_b": agent.cfg.actions.vibeHeartB
  of "charger": agent.cfg.actions.vibeCharger
  of "gear": agent.cfg.actions.vibeGear
  of "assembler": agent.cfg.actions.vibeAssembler
  of "chest": agent.cfg.actions.vibeChest
  of "wall": agent.cfg.actions.vibeWall
  of "default": agent.cfg.actions.vibeDefault
  else: agent.cfg.actions.vibeDefault

proc doGather(agent: LadybugAgent): int =
  let waitAction = agent.handleWaiting()
  if waitAction.isSome():
    return waitAction.get()
  if agent.state.heartRecipe.len == 0:
    agent.state.targetResource = ""
    return agent.explore()
  let deficits = agent.calculateDeficits()
  var needed: seq[(string, int)]
  for resource, deficit in deficits.pairs:
    if deficit > 0:
      needed.add((resource, deficit))
  if needed.len == 0:
    clearWaiting(agent)
    return agent.explore()
  needed.sort(proc(a, b: (string, int)): int = cmp(b[1], a[1]))
  for (resource, _) in needed:
    let extractorOpt = agent.findNearestExtractor(resource)
    if extractorOpt.isSome():
      agent.state.targetResource = resource
      let extractor = extractorOpt.get()
      let nav = agent.navigateToAdjacent(extractor.position.y, extractor.position.x)
      if nav.isSome():
        return nav.get()
      return agent.useExtractor(extractor)
  agent.state.targetResource = needed[0][0]
  return agent.explore()

proc doAssemble(agent: LadybugAgent): int =
  agent.state.targetResource = ""
  if agent.state.stations["assembler"].isNone:
    return agent.explore()
  if agent.state.currentGlyph != "heart_a":
    agent.state.currentGlyph = "heart_a"
    return agent.vibeAction("heart_a")
  let loc = agent.state.stations["assembler"].get()
  let nav = agent.navigateToAdjacent(loc.y, loc.x)
  if nav.isSome():
    return nav.get()
  return agent.moveIntoCell(loc.y, loc.x)

proc doDeliver(agent: LadybugAgent): int =
  agent.state.targetResource = ""
  if agent.state.stations["chest"].isNone:
    return agent.explore()
  if agent.state.currentGlyph != "default":
    agent.state.currentGlyph = "default"
    return agent.vibeAction("default")
  let loc = agent.state.stations["chest"].get()
  let nav = agent.navigateToAdjacent(loc.y, loc.x)
  if nav.isSome():
    return nav.get()
  return agent.moveIntoCell(loc.y, loc.x)

proc doRecharge(agent: LadybugAgent): int =
  agent.state.targetResource = ""
  if agent.state.stations["charger"].isNone:
    return agent.explore()
  let loc = agent.state.stations["charger"].get()
  let nav = agent.navigateToAdjacent(loc.y, loc.x)
  if nav.isSome():
    return nav.get()
  return agent.moveIntoCell(loc.y, loc.x)

proc executePhase(agent: LadybugAgent): int =
  case agent.state.phase
  of gatherPhase:
    doGather(agent)
  of assemblePhase:
    doAssemble(agent)
  of deliverPhase:
    doDeliver(agent)
  of rechargePhase:
    doRecharge(agent)

proc step*(
  agent: LadybugAgent,
  numAgents: int,
  numTokens: int,
  sizeToken: int,
  rawObservations: pointer,
  numActions: int,
  rawActions: pointer
) {.raises: [].} =
  discard numAgents
  discard numActions
  let observations = cast[ptr UncheckedArray[uint8]](rawObservations)
  let actions = cast[ptr UncheckedArray[int32]](rawActions)
  let offset = agent.agentId * numTokens * sizeToken
  let agentObservation = cast[pointer](observations[offset].addr)
  try:
    inc agent.state.stepCount
    updatePosition(agent)
    let visible = agent.decodeObservation(numTokens, sizeToken, agentObservation)
    agent.updateInventory(visible)
    discoverObjects(agent, visible)
    updatePhase(agent)
    let desired = agent.desiredVibe()
    if agent.state.currentGlyph != desired:
      agent.state.currentGlyph = desired
      let actionId = agent.vibeAction(desired)
      actions[agent.agentId] = actionId.int32
      agent.state.lastAction = actionId
      return
    let actionId = agent.executePhase()
    actions[agent.agentId] = actionId.int32
    agent.state.lastAction = actionId
  except:
    echo getCurrentException().getStackTrace()
    echo getCurrentExceptionMsg()
    actions[agent.agentId] = agent.cfg.actions.noop.int32
    agent.state.lastAction = agent.cfg.actions.noop

proc newLadybugPolicy*(environmentConfig: string): LadybugPolicy =
  let cfg = parseConfig(environmentConfig)
  var agents: seq[LadybugAgent] = @[]
  for id in 0 ..< cfg.config.numAgents:
    agents.add(newLadybugAgent(id, environmentConfig))
  LadybugPolicy(agents: agents)

proc stepBatch*(
    policy: LadybugPolicy,
    agentIds: pointer,
    numAgentIds: int,
    numAgents: int,
    numTokens: int,
    sizeToken: int,
    rawObservations: pointer,
    numActions: int,
    rawActions: pointer
) =
  let ids = cast[ptr UncheckedArray[int32]](agentIds)
  for i in 0 ..< numAgentIds:
    let idx = int(ids[i])
    step(policy.agents[idx], numAgents, numTokens, sizeToken, rawObservations, numActions, rawActions)
