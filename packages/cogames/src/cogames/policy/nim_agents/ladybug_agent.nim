import
  std/[algorithm, deques, options, random, sequtils, sets, strutils, tables],
  common

type
  Phase = enum gatherPhase, assemblePhase, deliverPhase, rechargePhase
  ObservedObject = object
    name: string
    clipped: bool
    cooldownRemaining, remainingUses: int
    protocolInputs, protocolOutputs: Table[string, int]
    agentGroup, agentFrozen: int
  ExtractorInfo = object
    position: Location
    resource: string
    lastSeen, cooldownRemaining, remainingUses: int
    clipped: bool
  LadybugState = object
    row, col, mapHeight, mapWidth, obsHalfWidth, obsHalfHeight: int
    phase: Phase
    stepCount, waitSteps, lastAction: int
    explorationDirectionSetStep, explorationEscapeUntilStep, stuckEscapeStep: int
    positionHistory: seq[Location]
    occupancy: seq[seq[int]]
    agentOccupancy: HashSet[Location]
    energy, carbon, oxygen, germanium, silicon, hearts: int
    decoder, modulator, resonator, scrambler: int
    heartRecipe: Table[string, int]
    heartRecipeKnown: bool
    stations: Table[string, Option[Location]]
    extractors: Table[string, seq[ExtractorInfo]]
    currentGlyph, targetResource, pendingUseResource, explorationDirection: string
    pendingUseAmount: int
    waitingAtExtractor: Option[Location]
    usingObjectThisStep, stuckLoopDetected: bool
    cachedPath: seq[(int, int)]
    cachedPathTarget: Option[Location]
    cachedPathReachAdjacent: bool
  LadybugAgent* = ref object
    agentId*: int
    cfg*: Config
    random*: Rand
    state*: LadybugState
  LadybugPolicy* = ref object
    agents*: seq[LadybugAgent]

proc initHeartRecipeFromConfig(agent: LadybugAgent) {.raises: [].}

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
  ResourceTypes = ["carbon", "oxygen", "germanium", "silicon"]
  StationKeys = ["assembler", "chest", "charger"]
  DirectionNames = ["north", "south", "east", "west"]
  CardinalNeighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
  DefaultHeartRecipe = [
    ("carbon", 2),
    ("oxygen", 2),
    ("germanium", 1),
    ("silicon", 3)
  ]

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
  agent.state.pendingUseResource = ""
  agent.state.pendingUseAmount = 0
  agent.state.heartRecipe = initTable[string, int]()
  agent.state.heartRecipeKnown = false
  agent.initHeartRecipeFromConfig()
  if not agent.state.heartRecipeKnown:
    for (resource, amount) in DefaultHeartRecipe:
      agent.state.heartRecipe[resource] = amount
    agent.state.heartRecipeKnown = true
  agent.state.stations = initTable[string, Option[Location]]()
  for key in StationKeys:
    agent.state.stations[key] = none(Location)
  agent.state.extractors = initTable[string, seq[ExtractorInfo]]()
  for resource in ResourceTypes:
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
  agent.state.stuckLoopDetected = false
  agent.state.stuckEscapeStep = 0
  agent.state.cachedPath = @[]
  agent.state.cachedPathTarget = none(Location)
  agent.state.cachedPathReachAdjacent = false

proc detectLoops(agent: LadybugAgent)
proc tryRandomDirection(agent: LadybugAgent): int

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

proc tagName(cfg: Config, tagId: int): string =
  if tagId >= 0 and tagId < cfg.config.tags.len:
    return cfg.config.tags[tagId]
  ""

proc maybeAssign(
  table: var Table[string, int],
  featureField: int,
  key: string,
  featureId: int,
  value: int
) =
  if featureField != 0 and featureId == featureField:
    table[key] = value

proc clearCachedPath(agent: LadybugAgent) =
  agent.state.cachedPath = @[]
  agent.state.cachedPathTarget = none(Location)
  agent.state.cachedPathReachAdjacent = false

proc stationFromName(lowerName: string): string =
  if lowerName.contains("assembler"):
    "assembler"
  elif lowerName.contains("chest"):
    "chest"
  elif lowerName.contains("charger"):
    "charger"
  else:
    ""

proc initHeartRecipeFromConfig(agent: LadybugAgent) =
  for protocol in agent.cfg.assemblerProtocols:
    if protocol.outputResources.getOrDefault("heart", 0) > 0:
      agent.state.heartRecipe = initTable[string, int]()
      for resource, amount in protocol.inputResources.pairs:
        if resource != "energy":
          agent.state.heartRecipe[resource] = amount
      agent.state.heartRecipeKnown = true
      return

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
  var tagIds: seq[int] = @[]
  result.protocolInputs = initTable[string, int]()
  result.protocolOutputs = initTable[string, int]()
  for fv in features:
    if fv.featureId == agent.cfg.features.tag:
      tagIds.add(fv.value)
      continue
    elif fv.featureId == agent.cfg.features.cooldownRemaining:
      result.cooldownRemaining = fv.value
    elif fv.featureId == agent.cfg.features.clipped:
      result.clipped = fv.value > 0
    elif fv.featureId == agent.cfg.features.remainingUses:
      result.remainingUses = fv.value
    elif fv.featureId == agent.cfg.features.group:
      result.agentGroup = fv.value
    elif fv.featureId == agent.cfg.features.frozen:
      result.agentFrozen = fv.value
    else:
      let fid = fv.featureId
      maybeAssign(result.protocolInputs, agent.cfg.features.protocolInputEnergy, "energy", fid, fv.value)
      maybeAssign(result.protocolInputs, agent.cfg.features.protocolInputCarbon, "carbon", fid, fv.value)
      maybeAssign(result.protocolInputs, agent.cfg.features.protocolInputOxygen, "oxygen", fid, fv.value)
      maybeAssign(result.protocolInputs, agent.cfg.features.protocolInputGermanium, "germanium", fid, fv.value)
      maybeAssign(result.protocolInputs, agent.cfg.features.protocolInputSilicon, "silicon", fid, fv.value)
      maybeAssign(result.protocolInputs, agent.cfg.features.protocolInputHeart, "heart", fid, fv.value)
      maybeAssign(result.protocolInputs, agent.cfg.features.protocolInputDecoder, "decoder", fid, fv.value)
      maybeAssign(result.protocolInputs, agent.cfg.features.protocolInputModulator, "modulator", fid, fv.value)
      maybeAssign(result.protocolInputs, agent.cfg.features.protocolInputResonator, "resonator", fid, fv.value)
      maybeAssign(result.protocolInputs, agent.cfg.features.protocolInputScrambler, "scrambler", fid, fv.value)
      maybeAssign(result.protocolOutputs, agent.cfg.features.protocolOutputEnergy, "energy", fid, fv.value)
      maybeAssign(result.protocolOutputs, agent.cfg.features.protocolOutputCarbon, "carbon", fid, fv.value)
      maybeAssign(result.protocolOutputs, agent.cfg.features.protocolOutputOxygen, "oxygen", fid, fv.value)
      maybeAssign(result.protocolOutputs, agent.cfg.features.protocolOutputGermanium, "germanium", fid, fv.value)
      maybeAssign(result.protocolOutputs, agent.cfg.features.protocolOutputSilicon, "silicon", fid, fv.value)
      maybeAssign(result.protocolOutputs, agent.cfg.features.protocolOutputHeart, "heart", fid, fv.value)
      maybeAssign(result.protocolOutputs, agent.cfg.features.protocolOutputDecoder, "decoder", fid, fv.value)
      maybeAssign(result.protocolOutputs, agent.cfg.features.protocolOutputModulator, "modulator", fid, fv.value)
      maybeAssign(result.protocolOutputs, agent.cfg.features.protocolOutputResonator, "resonator", fid, fv.value)
      maybeAssign(result.protocolOutputs, agent.cfg.features.protocolOutputScrambler, "scrambler", fid, fv.value)

  if tagIds.len > 0:
    result.name = tagName(agent.cfg, tagIds[0])
  elif result.agentGroup != 0 or result.agentFrozen != 0:
    result.name = "agent"
  else:
    result.name = ""

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

    let stationKey = stationFromName(lowerName)
    if stationKey.len > 0:
      agent.state.occupancy[world.y][world.x] = cellObstacle
      discoverStation(agent, stationKey, world)
      if stationKey == "assembler" and (not agent.state.heartRecipeKnown) and
          obj.protocolOutputs.getOrDefault("heart", 0) > 0:
        agent.state.heartRecipe = initTable[string, int]()
        for resource, amount in obj.protocolInputs.pairs:
          if resource != "energy":
            agent.state.heartRecipe[resource] = amount
        agent.state.heartRecipeKnown = true
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
  detectLoops(agent)

proc detectLoops(agent: LadybugAgent) =
  let hist = agent.state.positionHistory
  let lenHist = hist.len
  if lenHist >= 6:
    let a1 = hist[lenHist - 1]
    let a2 = hist[lenHist - 2]
    let a3 = hist[lenHist - 3]
    let a4 = hist[lenHist - 4]
    let a5 = hist[lenHist - 5]
    let a6 = hist[lenHist - 6]
    if a1 == a3 and a3 == a5 and a2 == a4 and a4 == a6 and a1 != a2:
      agent.state.stuckLoopDetected = true
      agent.state.stuckEscapeStep = agent.state.stepCount
      return
  if lenHist >= 9:
    let b1 = hist[lenHist - 1]
    let b2 = hist[lenHist - 2]
    let b3 = hist[lenHist - 3]
    let b4 = hist[lenHist - 4]
    let b5 = hist[lenHist - 5]
    let b6 = hist[lenHist - 6]
    let b7 = hist[lenHist - 7]
    let b8 = hist[lenHist - 8]
    let b9 = hist[lenHist - 9]
    if b1 == b4 and b4 == b7 and b2 == b5 and b5 == b8 and b3 == b6 and b6 == b9:
      agent.state.stuckLoopDetected = true
      agent.state.stuckEscapeStep = agent.state.stepCount

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
  if not agent.state.heartRecipeKnown:
    raise newException(ValueError,
      "Heart recipe not discovered! Agent must observe assembler with correct vibe to learn recipe. Ensure protocol_details_obs=True in game config.")
  result = initTable[string, int]()
  for resource in ResourceTypes:
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

proc findAnyNeededExtractor(agent: LadybugAgent): Option[(ExtractorInfo, string)] =
  let deficits = agent.calculateDeficits()
  var needed = toSeq(deficits.pairs)
  needed.sort(proc(a, b: (string, int)): int = cmp(b[1], a[1]))
  for (resource, deficit) in needed:
    if deficit > 0:
      let extractorOpt = agent.findNearestExtractor(resource)
      if extractorOpt.isSome():
        return some((extractorOpt.get(), resource))
  none((ExtractorInfo, string))

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
  for delta in CardinalNeighbors:
    let nr = targetRow + delta[0]
    let nc = targetCol + delta[1]
    if agent.isTraversable(nr, nc):
      result.add((nr, nc))
  if result.len == 0:
    for delta in CardinalNeighbors:
      let nr = targetRow + delta[0]
      let nc = targetCol + delta[1]
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
  var goalSet = initHashSet[(int, int)]()
  for g in goals:
    goalSet.incl(g)
  let targetLoc = Location(x: targetCol, y: targetRow)
  var pathValid = false
  if agent.state.cachedPath.len > 0 and agent.state.cachedPathTarget.isSome and
      agent.state.cachedPathTarget.get().x == targetCol and
      agent.state.cachedPathTarget.get().y == targetRow and
      agent.state.cachedPathReachAdjacent == reachAdjacent:
    let nextStep = agent.state.cachedPath[0]
    if agent.isTraversable(nextStep[0], nextStep[1]) or (allowGoalBlock and nextStep in goalSet):
      pathValid = true
    else:
      clearCachedPath(agent)
  if not pathValid:
    var queue = initDeque[(int, int)]()
    var cameFrom = initTable[(int, int), (int, int)]()
    let start = (agent.state.row, agent.state.col)
    queue.addLast(start)
    cameFrom[start] = parentSentinel
    var found = false
    var goalReached: (int, int)
    while queue.len > 0 and not found:
      let current = queue.popFirst()
      if current in goalSet:
        found = true
        goalReached = current
        break
      for delta in CardinalNeighbors:
        let nr = current[0] + delta[0]
        let nc = current[1] + delta[1]
        if cameFrom.hasKey((nr, nc)):
          continue
        var canTraverse = agent.isTraversable(nr, nc)
        if not canTraverse and allowGoalBlock and (nr, nc) in goalSet:
          canTraverse = true
        if not canTraverse:
          continue
        cameFrom[(nr, nc)] = current
        queue.addLast((nr, nc))
    if not found:
      clearCachedPath(agent)
      let randomAction = agent.tryRandomDirection()
      return randomAction
    let newPath = reconstructPath(cameFrom, goalReached)
    if newPath.len == 0:
      return agent.cfg.actions.noop
    agent.state.cachedPath = newPath
    agent.state.cachedPathTarget = some(targetLoc)
    agent.state.cachedPathReachAdjacent = reachAdjacent
  let nextStep = agent.state.cachedPath[0]
  agent.state.cachedPath.delete(0)
  if agent.state.cachedPath.len == 0:
    clearCachedPath(agent)
  if allowGoalBlock and agent.isAgentBlocking(nextStep[0], nextStep[1]):
    clearCachedPath(agent)
    let reroute = agent.tryRandomDirection()
    return reroute
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
  agent.cfg.actions.noop

proc directionDelta(direction: string): (int, int) =
  case direction
  of "north": (-1, 0)
  of "south": (1, 0)
  of "east": (0, 1)
  of "west": (0, -1)
  else: (0, 0)

proc stepInDirection(agent: LadybugAgent, direction: string): Option[int] =
  let delta = directionDelta(direction)
  if delta == (0, 0):
    return none(int)
  let nr = agent.state.row + delta[0]
  let nc = agent.state.col + delta[1]
  if not agent.isTraversable(nr, nc):
    return none(int)
  let action = agent.moveTowards(nr, nc)
  if action == agent.cfg.actions.noop:
    return none(int)
  some(action)

proc tryRandomDirection(agent: LadybugAgent): int =
  var dirs = @DirectionNames
  agent.random.shuffle(dirs)
  for dir in dirs:
    let action = agent.stepInDirection(dir)
    if action.isSome():
      clearCachedPath(agent)
      return action.get()
  return agent.cfg.actions.noop

proc explore(agent: LadybugAgent): int =
  clearCachedPath(agent)
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
    let action = agent.stepInDirection(agent.state.explorationDirection)
    if action.isSome():
      return action.get()
    agent.state.explorationDirection = ""

  var dirs = @DirectionNames
  agent.random.shuffle(dirs)
  for dir in dirs:
    let action = agent.stepInDirection(dir)
    if action.isSome():
      agent.state.explorationDirection = dir
      agent.state.explorationDirectionSetStep = agent.state.stepCount
      return action.get()
  return agent.tryRandomDirection()

proc exploreUntil(agent: LadybugAgent, condition: proc (): bool, reason: string): Option[int] =
  if condition():
    return none(int)
  agent.state.explorationEscapeUntilStep = 0
  some(agent.explore())

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
  if agent.state.row == targetRow and agent.state.col == targetCol:
    return agent.cfg.actions.noop
  if agent.isAgentBlocking(targetRow, targetCol):
    clearCachedPath(agent)
    return agent.tryRandomDirection()
  agent.state.usingObjectThisStep = true
  return agent.moveTowards(targetRow, targetCol, allowGoalBlock = true)

proc useExtractor(agent: LadybugAgent, extractor: ExtractorInfo): int =
  if extractor.cooldownRemaining > 0:
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

proc handleStuck(agent: LadybugAgent): Option[int] =
  if not agent.state.stuckLoopDetected:
    return none(int)
  agent.state.stuckLoopDetected = false
  some(agent.tryRandomDirection())

proc updatePhase(agent: LadybugAgent) =
  let previousPhase = agent.state.phase
  if agent.state.energy < rechargeThresholdLow:
    agent.state.phase = rechargePhase
    clearWaiting(agent)
    if previousPhase != agent.state.phase:
      clearCachedPath(agent)
    return
  if agent.state.phase == rechargePhase and agent.state.energy < rechargeThresholdHigh:
    return
  if agent.state.hearts > 0:
    agent.state.phase = deliverPhase
    clearWaiting(agent)
    if previousPhase != agent.state.phase:
      clearCachedPath(agent)
    return
  if agent.state.heartRecipeKnown:
    var ready = true
    for resource, amount in agent.state.heartRecipe.pairs:
      if agent.getInventoryValue(resource) < amount:
        ready = false
        break
    if ready:
      agent.state.phase = assemblePhase
      clearWaiting(agent)
      if previousPhase != agent.state.phase:
        clearCachedPath(agent)
      return
  agent.state.phase = gatherPhase
  if previousPhase != agent.state.phase:
    clearCachedPath(agent)

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
  if not agent.state.heartRecipeKnown:
    agent.state.targetResource = ""
    clearWaiting(agent)
    return agent.explore()
  let deficits = agent.calculateDeficits()
  var needsResources = false
  for deficit in deficits.values:
    if deficit > 0:
      needsResources = true
      break
  if not needsResources:
    clearWaiting(agent)
    return agent.cfg.actions.noop
  let exploreAction = agent.exploreUntil(
    proc (): bool = agent.findAnyNeededExtractor().isSome(),
    "need extractor"
  )
  if exploreAction.isSome():
    return exploreAction.get()
  let extractorChoice = agent.findAnyNeededExtractor()
  if extractorChoice.isNone():
    return agent.explore()
  let (extractor, resource) = extractorChoice.get()
  agent.state.targetResource = resource
  let nav = agent.navigateToAdjacent(extractor.position.y, extractor.position.x)
  if nav.isSome():
    clearWaiting(agent)
    return nav.get()
  return agent.useExtractor(extractor)

proc doAssemble(agent: LadybugAgent): int =
  agent.state.targetResource = ""
  let exploreAction = agent.exploreUntil(
    proc (): bool = agent.state.stations["assembler"].isSome(),
    "need assembler"
  )
  if exploreAction.isSome():
    return exploreAction.get()
  let loc = agent.state.stations["assembler"].get()
  let nav = agent.navigateToAdjacent(loc.y, loc.x)
  if nav.isSome():
    return nav.get()
  return agent.moveIntoCell(loc.y, loc.x)

proc doDeliver(agent: LadybugAgent): int =
  agent.state.targetResource = ""
  let exploreAction = agent.exploreUntil(
    proc (): bool = agent.state.stations["chest"].isSome(),
    "need chest"
  )
  if exploreAction.isSome():
    return exploreAction.get()
  let loc = agent.state.stations["chest"].get()
  let nav = agent.navigateToAdjacent(loc.y, loc.x)
  if nav.isSome():
    return nav.get()
  return agent.moveIntoCell(loc.y, loc.x)

proc doRecharge(agent: LadybugAgent): int =
  agent.state.targetResource = ""
  let exploreAction = agent.exploreUntil(
    proc (): bool = agent.state.stations["charger"].isSome(),
    "need charger"
  )
  if exploreAction.isSome():
    return exploreAction.get()
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
    let stuckAction = handleStuck(agent)
    if stuckAction.isSome():
      let actionId = stuckAction.get()
      actions[agent.agentId] = actionId.int32
      agent.state.lastAction = actionId
      return
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
