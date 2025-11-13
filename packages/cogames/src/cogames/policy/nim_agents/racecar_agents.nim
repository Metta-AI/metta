import
  std/[tables, random, sets, options],
  common

type
  ResourceKind = enum
    rkCarbon,
    rkOxygen,
    rkGermanium,
    rkSilicon

  RaceCarAgent* = ref object
    agentId*: int
    map: Table[Location, seq[FeatureValue]]
    seen: HashSet[Location]
    cfg: Config
    random: Rand
    location: Location
    lastActions: seq[int]
    resourceFocus: Option[ResourceKind]

  RaceCarPolicy* = ref object
    agents*: seq[RaceCarAgent]

const
  EnergyReserve = 40
  ChestDepositThreshold = 80
  MaxHistory = 8

proc resourceQuota(kind: ResourceKind): int =
  case kind
  of rkSilicon:
    50
  else:
    30

proc vibeFor(agent: RaceCarAgent, kind: ResourceKind): int32 =
  case kind
  of rkCarbon:
    agent.cfg.actions.vibeCarbon.int32
  of rkOxygen:
    agent.cfg.actions.vibeOxygen.int32
  of rkGermanium:
    agent.cfg.actions.vibeGermanium.int32
  of rkSilicon:
    agent.cfg.actions.vibeSilicon.int32

template inventoryFeature(agent: RaceCarAgent, kind: ResourceKind): int =
  case kind
  of rkCarbon:
    agent.cfg.features.invCarbon
  of rkOxygen:
    agent.cfg.features.invOxygen
  of rkGermanium:
    agent.cfg.features.invGermanium
  of rkSilicon:
    agent.cfg.features.invSilicon

template extractorTag(agent: RaceCarAgent, kind: ResourceKind): int =
  case kind
  of rkCarbon:
    agent.cfg.tags.carbonExtractor
  of rkOxygen:
    agent.cfg.tags.oxygenExtractor
  of rkGermanium:
    agent.cfg.tags.germaniumExtractor
  of rkSilicon:
    agent.cfg.tags.siliconExtractor

proc sampleInventory(agent: RaceCarAgent, visible: Table[Location, seq[FeatureValue]]): Table[ResourceKind, int] =
  result = initTable[ResourceKind, int]()
  for kind in ResourceKind:
    let value = agent.cfg.getInventory(visible, agent.inventoryFeature(kind))
    result[kind] = value

proc hasHeartIngredients(inventory: Table[ResourceKind, int]): bool =
  for kind in ResourceKind:
    if inventory.getOrDefault(kind, 0) < resourceQuota(kind):
      return false
  true

proc shouldDeposit(inventory: Table[ResourceKind, int]): bool =
  var total = 0
  var readyKinds = 0
  for kind in ResourceKind:
    let amount = inventory.getOrDefault(kind, 0)
    total += amount
    if amount >= resourceQuota(kind):
      inc readyKinds
  total >= ChestDepositThreshold or readyKinds >= 2

proc currentFocus(agent: RaceCarAgent, inventory: Table[ResourceKind, int]): Option[ResourceKind] =
  if agent.resourceFocus.isSome:
    let kind = agent.resourceFocus.get()
    if inventory.getOrDefault(kind, 0) >= resourceQuota(kind):
      agent.resourceFocus = none(ResourceKind)
    else:
      return agent.resourceFocus
  var best: Option[ResourceKind] = none(ResourceKind)
  var lowest = high(int)
  for kind in ResourceKind:
    let amount = inventory.getOrDefault(kind, 0)
    if amount < resourceQuota(kind) and amount < lowest:
      lowest = amount
      best = some(kind)
  agent.resourceFocus = best
  agent.resourceFocus

proc randomMove(agent: RaceCarAgent): int32 =
  case agent.random.rand(0 .. 3)
  of 0:
    agent.cfg.actions.moveNorth.int32
  of 1:
    agent.cfg.actions.moveSouth.int32
  of 2:
    agent.cfg.actions.moveWest.int32
  else:
    agent.cfg.actions.moveEast.int32

proc moveGreedy(agent: RaceCarAgent, target: Location): int32 =
  let dx = target.x - agent.location.x
  let dy = target.y - agent.location.y
  if dx == 0 and dy == 0:
    return agent.cfg.actions.noop.int32
  if abs(dx) >= abs(dy):
    if dx > 0:
      return agent.cfg.actions.moveEast.int32
    return agent.cfg.actions.moveWest.int32
  if dy > 0:
    return agent.cfg.actions.moveSouth.int32
  agent.cfg.actions.moveNorth.int32

proc moveTowards(agent: RaceCarAgent, target: Location): int32 =
  if target == agent.location:
    return agent.cfg.actions.noop.int32
  let action = agent.cfg.aStar(agent.location, target, agent.map)
  if action.isSome():
    return action.get().int32
  agent.moveGreedy(target)

proc exploreAction(agent: RaceCarAgent): int32 =
  let unseen = agent.cfg.getNearbyUnseen(agent.location, agent.map, agent.seen)
  if unseen.isSome():
    let action = agent.cfg.aStar(agent.location, unseen.get(), agent.map)
    if action.isSome():
      return action.get().int32

  var candidates: seq[(Location, int32)]
  let north = Location(x: agent.location.x, y: agent.location.y - 1)
  let south = Location(x: agent.location.x, y: agent.location.y + 1)
  let west = Location(x: agent.location.x - 1, y: agent.location.y)
  let east = Location(x: agent.location.x + 1, y: agent.location.y)
  if north notin agent.seen:
    candidates.add((north, agent.cfg.actions.moveNorth.int32))
  if south notin agent.seen:
    candidates.add((south, agent.cfg.actions.moveSouth.int32))
  if west notin agent.seen:
    candidates.add((west, agent.cfg.actions.moveWest.int32))
  if east notin agent.seen:
    candidates.add((east, agent.cfg.actions.moveEast.int32))
  if candidates.len == 0:
    return agent.randomMove()
  let idx = agent.random.rand(0 ..< candidates.len)
  candidates[idx][1]

proc stabilizeAction(agent: RaceCarAgent, action: int32): int32 =
  var candidate = action.int
  if agent.lastActions.len >= 2:
    if candidate == agent.cfg.actions.moveWest and
        agent.lastActions[^1] == agent.cfg.actions.moveEast and
        agent.lastActions[^2] == agent.cfg.actions.moveWest and
        agent.random.rand(1 .. 2) == 1:
      candidate = agent.cfg.actions.noop
    elif candidate == agent.cfg.actions.moveEast and
        agent.lastActions[^1] == agent.cfg.actions.moveWest and
        agent.lastActions[^2] == agent.cfg.actions.moveEast and
        agent.random.rand(1 .. 2) == 1:
      candidate = agent.cfg.actions.noop
    elif candidate == agent.cfg.actions.moveNorth and
        agent.lastActions[^1] == agent.cfg.actions.moveSouth and
        agent.lastActions[^2] == agent.cfg.actions.moveNorth and
        agent.random.rand(1 .. 2) == 1:
      candidate = agent.cfg.actions.noop
    elif candidate == agent.cfg.actions.moveSouth and
        agent.lastActions[^1] == agent.cfg.actions.moveNorth and
        agent.lastActions[^2] == agent.cfg.actions.moveSouth and
        agent.random.rand(1 .. 2) == 1:
      candidate = agent.cfg.actions.noop
  agent.lastActions.add(candidate)
  if agent.lastActions.len > MaxHistory:
    agent.lastActions.delete(0)
  candidate.int32

proc updateMap(agent: RaceCarAgent, visible: Table[Location, seq[FeatureValue]]) =
  if agent.map.len == 0:
    agent.map = visible
    agent.location = Location(x: 0, y: 0)
    for location in keys(visible):
      agent.seen.incl(location)
    return

  var
    bestScore = 0
    bestLocation = agent.location
    possibleOffsets: seq[Location]

  let lastAction = agent.cfg.getLastAction(visible)
  if lastAction == agent.cfg.actions.moveNorth or lastAction == -1:
    possibleOffsets.add(Location(x: 0, y: -1))
  if lastAction == agent.cfg.actions.moveSouth or lastAction == -1:
    possibleOffsets.add(Location(x: 0, y: 1))
  if lastAction == agent.cfg.actions.moveWest or lastAction == -1:
    possibleOffsets.add(Location(x: -1, y: 0))
  if lastAction == agent.cfg.actions.moveEast or lastAction == -1:
    possibleOffsets.add(Location(x: 1, y: 0))
  possibleOffsets.add(Location(x: 0, y: 0))

  for offset in possibleOffsets:
    var score = 0
    let location = Location(x: agent.location.x + offset.x, y: agent.location.y + offset.y)
    for x in -5 .. 5:
      for y in -5 .. 5:
        let visibleTag = agent.cfg.getTag(visible, Location(x: x, y: y))
        let mapTag = agent.cfg.getTag(agent.map, Location(x: x + location.x, y: y + location.y))
        if visibleTag == mapTag:
          if visibleTag == agent.cfg.tags.agent or visibleTag == -1:
            discard
          elif visibleTag == agent.cfg.tags.assembler:
            score += 100
          elif visibleTag == agent.cfg.tags.wall:
            score += 1
          else:
            score += 10
    if score > bestScore:
      bestScore = score
      bestLocation = location

  if bestScore >= 2:
    agent.location = bestLocation
    for x in -5 .. 5:
      for y in -5 .. 5:
        let visibleLocation = Location(x: x, y: y)
        let location = Location(x: x + agent.location.x, y: y + agent.location.y)
        if visibleLocation in visible:
          agent.map[location] = visible[visibleLocation]
        else:
          agent.map[location] = @[]
        agent.seen.incl(location)

proc decideAction(agent: RaceCarAgent, visible: Table[Location, seq[FeatureValue]]): int32 =
  let vibe = agent.cfg.getVibe(visible)
  let invEnergy = agent.cfg.getInventory(visible, agent.cfg.features.invEnergy)
  let invHeart = agent.cfg.getInventory(visible, agent.cfg.features.invHeart)
  let inventory = agent.sampleInventory(visible)
  let charger = agent.cfg.getNearby(agent.location, agent.map, agent.cfg.tags.charger)
  let chest = agent.cfg.getNearby(agent.location, agent.map, agent.cfg.tags.chest)
  let assembler = agent.cfg.getNearby(agent.location, agent.map, agent.cfg.tags.assembler)

  if invEnergy < EnergyReserve and charger.isSome():
    return agent.moveTowards(charger.get())

  if invHeart > 0:
    if vibe != agent.cfg.vibes.default:
      return agent.cfg.actions.vibeDefault.int32
    if chest.isSome():
      return agent.moveTowards(chest.get())

  if hasHeartIngredients(inventory):
    if vibe != agent.cfg.vibes.heart:
      return agent.cfg.actions.vibeHeart.int32
    if assembler.isSome():
      return agent.moveTowards(assembler.get())

  if shouldDeposit(inventory) and chest.isSome():
    if chest.get() == agent.location:
      return agent.cfg.actions.vibeChest.int32
    return agent.moveTowards(chest.get())

  let focus = agent.currentFocus(inventory)
  if focus.isSome():
    let kind = focus.get()
    if vibe != agent.vibeFor(kind):
      return agent.vibeFor(kind)
    let extractor = agent.cfg.getNearby(agent.location, agent.map, agent.extractorTag(kind))
    if extractor.isSome():
      return agent.moveTowards(extractor.get())

  agent.exploreAction()

proc step*(
  agent: RaceCarAgent,
  numAgents: int,
  numTokens: int,
  sizeToken: int,
  rawObservation: pointer,
  numActions: int,
  agentAction: ptr int32
) {.raises: [].} =
  try:
    discard numAgents
    discard numActions
    let observations = cast[ptr UncheckedArray[uint8]](rawObservation)
    var visible = initTable[Location, seq[FeatureValue]]()
    for token in 0 ..< numTokens:
      let base = token * sizeToken
      let locationPacked = observations[base]
      let featureId = observations[base + 1]
      let value = observations[base + 2]
      if locationPacked == 255 and featureId == 255 and value == 255:
        break
      var location: Location
      if locationPacked != 0xFF:
        location.y = (locationPacked shr 4).int - 5
        location.x = (locationPacked and 0x0F).int - 5
      if location notin visible:
        visible[location] = @[]
      visible[location].add(FeatureValue(featureId: featureId.int, value: value.int))

    agent.updateMap(visible)
    let action = agent.decideAction(visible)
    let stabilized = agent.stabilizeAction(action)
    agentAction[] = stabilized
  except:
    echo getCurrentException().getStackTrace()
    echo getCurrentExceptionMsg()
    quit()

proc newRaceCarAgent*(agentId: int, environmentConfig: string): RaceCarAgent =
  var config = parseConfig(environmentConfig)
  result = RaceCarAgent(agentId: agentId, cfg: config)
  result.random = initRand(agentId)
  result.map = initTable[Location, seq[FeatureValue]]()
  result.seen = initHashSet[Location]()
  result.location = Location(x: 0, y: 0)
  result.lastActions = @[]
  result.resourceFocus = none(ResourceKind)

proc newRaceCarPolicy*(environmentConfig: string): RaceCarPolicy =
  let cfg = parseConfig(environmentConfig)
  var agents: seq[RaceCarAgent] = @[]
  for id in 0 ..< cfg.config.numAgents:
    agents.add(newRaceCarAgent(id, environmentConfig))
  RaceCarPolicy(agents: agents)

proc stepBatch*(
    policy: RaceCarPolicy,
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
  let obsArray = cast[ptr UncheckedArray[uint8]](rawObservations)
  let actionArray = cast[ptr UncheckedArray[int32]](rawActions)
  let obsStride = numTokens * sizeToken

  for i in 0 ..< numAgentIds:
    let idx = int(ids[i])
    if idx < 0 or idx >= policy.agents.len:
      continue
    let obsPtr = cast[pointer](obsArray[idx * obsStride].addr)
    let actPtr = cast[ptr int32](actionArray[idx].addr)
    step(policy.agents[idx], numAgents, numTokens, sizeToken, obsPtr, numActions, actPtr)
