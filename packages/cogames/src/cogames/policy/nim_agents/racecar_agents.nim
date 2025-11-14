import
  std/[tables, random, sets, options],
  common

when defined(racecar_debug):
  template rcEcho(args: varargs[untyped]) =
    {.push stackTrace: off.}
    system.echo(args)
    {.pop.}
else:
  template rcEcho(args: varargs[untyped]) = discard

template echo(args: varargs[untyped]) = rcEcho(args)

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
  EnergyReserve = 50
  ResourceQuotaCarbon = 30
  ResourceQuotaOxygen = 30
  ResourceQuotaGermanium = 30
  ResourceQuotaSilicon = 50

proc resourceQuota(kind: ResourceKind): int =
  case kind
  of rkCarbon: ResourceQuotaCarbon
  of rkOxygen: ResourceQuotaOxygen
  of rkGermanium: ResourceQuotaGermanium
  of rkSilicon: ResourceQuotaSilicon

proc vibeFor(agent: RaceCarAgent, kind: ResourceKind): int32 =
  case kind
  of rkCarbon: agent.cfg.actions.vibeCarbon.int32
  of rkOxygen: agent.cfg.actions.vibeOxygen.int32
  of rkGermanium: agent.cfg.actions.vibeGermanium.int32
  of rkSilicon: agent.cfg.actions.vibeSilicon.int32

proc extractorTag(agent: RaceCarAgent, kind: ResourceKind): int =
  case kind
  of rkCarbon: agent.cfg.tags.carbonExtractor
  of rkOxygen: agent.cfg.tags.oxygenExtractor
  of rkGermanium: agent.cfg.tags.germaniumExtractor
  of rkSilicon: agent.cfg.tags.siliconExtractor

proc sampleInventory(agent: RaceCarAgent, visible: Table[Location, seq[FeatureValue]]): Table[ResourceKind, int] =
  result = initTable[ResourceKind, int]()
  result[rkCarbon] = agent.cfg.getInventory(visible, agent.cfg.features.invCarbon)
  result[rkOxygen] = agent.cfg.getInventory(visible, agent.cfg.features.invOxygen)
  result[rkGermanium] = agent.cfg.getInventory(visible, agent.cfg.features.invGermanium)
  result[rkSilicon] = agent.cfg.getInventory(visible, agent.cfg.features.invSilicon)

proc hasHeartIngredients(inventory: Table[ResourceKind, int]): bool =
  for kind in ResourceKind:
    if inventory.getOrDefault(kind, 0) < resourceQuota(kind):
      return false
  true

proc updateFocus(agent: RaceCarAgent, inventory: Table[ResourceKind, int]) =
  if agent.resourceFocus.isSome:
    let current = agent.resourceFocus.get()
    if inventory.getOrDefault(current, 0) >= resourceQuota(current):
      agent.resourceFocus = none(ResourceKind)

  if agent.resourceFocus.isSome:
    return

  var bestKind: Option[ResourceKind]
  var lowest = high(int)
  for kind in ResourceKind:
    let amount = inventory.getOrDefault(kind, 0)
    if amount < resourceQuota(kind) and amount < lowest:
      lowest = amount
      bestKind = some(kind)

  agent.resourceFocus = bestKind

proc randomMove(agent: RaceCarAgent): int32 =
  case agent.random.rand(1 .. 4)
  of 1: agent.cfg.actions.moveNorth.int32
  of 2: agent.cfg.actions.moveSouth.int32
  of 3: agent.cfg.actions.moveWest.int32
  else: agent.cfg.actions.moveEast.int32

proc stabilizeAction(agent: RaceCarAgent, action: int): int32 =
  var candidate = action
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
  candidate.int32

proc updateMap(agent: RaceCarAgent, visible: Table[Location, seq[FeatureValue]]) =
  if agent.map.len == 0:
    agent.map = visible
    agent.location = Location(x: 0, y: 0)
    return

  var
    bestScore = 0
    bestLocation = agent.location
    possibleOffsets: seq[Location]

  var lastAction = agent.cfg.getLastAction(visible)
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

  if bestScore < 2:
    discard
  else:
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

proc seekExtractor(agent: RaceCarAgent, kind: ResourceKind): Option[int32] =
  let tag = agent.extractorTag(kind)
  let nearby = agent.cfg.getNearby(agent.location, agent.map, tag)
  if nearby.isSome():
    let action = agent.cfg.aStar(agent.location, nearby.get(), agent.map)
    if action.isSome():
      return some(action.get().int32)
  none(int32)

proc step*(
  agent: RaceCarAgent,
  numAgents: int,
  numTokens: int,
  sizeToken: int,
  rawObservation: pointer,
  numActions: int,
  agentAction: ptr int32
) =
  try:
    discard numAgents
    discard numActions
    let observations = cast[ptr UncheckedArray[uint8]](rawObservation)

    proc doAction(action: int32) =
      let stabilized = agent.stabilizeAction(action.int)
      agentAction[] = stabilized

    var map: Table[Location, seq[FeatureValue]]
    for token in 0 ..< numTokens:
      let locationPacked = observations[token * sizeToken]
      let featureId = observations[token * sizeToken + 1]
      let value = observations[token * sizeToken + 2]
      if locationPacked == 255 and featureId == 255 and value == 255:
        break
      var location: Location
      if locationPacked != 0xFF:
        location.y = (locationPacked shr 4).int - 5
        location.x = (locationPacked and 0x0F).int - 5
      if location notin map:
        map[location] = @[]
      map[location].add(FeatureValue(featureId: featureId.int, value: value.int))

    updateMap(agent, map)

    let vibe = agent.cfg.getVibe(map)
    let invEnergy = agent.cfg.getInventory(map, agent.cfg.features.invEnergy)
    let invHeart = agent.cfg.getInventory(map, agent.cfg.features.invHeart)
    let invCarbon = agent.cfg.getInventory(map, agent.cfg.features.invCarbon)
    let invOxygen = agent.cfg.getInventory(map, agent.cfg.features.invOxygen)
    let invGermanium = agent.cfg.getInventory(map, agent.cfg.features.invGermanium)
    let invSilicon = agent.cfg.getInventory(map, agent.cfg.features.invSilicon)

    if invEnergy < EnergyReserve:
      let chargerNearby = agent.cfg.getNearby(agent.location, agent.map, agent.cfg.tags.charger)
      if chargerNearby.isSome():
        let action = agent.cfg.aStar(agent.location, chargerNearby.get(), agent.map)
        if action.isSome():
          doAction(action.get().int32)
          return

    if invHeart > 0:
      if vibe != agent.cfg.vibes.default:
        doAction(agent.cfg.actions.vibeDefault.int32)
        return
      let chestNearby = agent.cfg.getNearby(agent.location, agent.map, agent.cfg.tags.chest)
      if chestNearby.isSome():
        let action = agent.cfg.aStar(agent.location, chestNearby.get(), agent.map)
        if action.isSome():
          doAction(action.get().int32)
          return

    let keyLocations = [
      Location(x: -10, y: -10),
      Location(x: -10, y: +10),
      Location(x: +10, y: -10),
      Location(x: +10, y: +10),
    ]
    let assemblerNearby = agent.cfg.getNearby(agent.location, agent.map, agent.cfg.tags.assembler)
    if assemblerNearby.isSome():
      for keyLocation in keyLocations:
        let location = assemblerNearby.get() + keyLocation
        if location notin agent.seen:
          let action = agent.cfg.aStar(agent.location, location, agent.map)
          if action.isSome():
            doAction(action.get().int32)
            return

    if invCarbon == 0:
      let action = agent.seekExtractor(rkCarbon)
      if action.isSome():
        doAction(action.get())
        return
    if invOxygen == 0:
      let action = agent.seekExtractor(rkOxygen)
      if action.isSome():
        doAction(action.get())
        return
    if invGermanium == 0:
      let action = agent.seekExtractor(rkGermanium)
      if action.isSome():
        doAction(action.get())
        return
    if invSilicon == 0:
      let action = agent.seekExtractor(rkSilicon)
      if action.isSome():
        doAction(action.get())
        return

    let inventory = agent.sampleInventory(map)
    agent.updateFocus(inventory)

    if hasHeartIngredients(inventory):
      if vibe != agent.cfg.vibes.heart:
        doAction(agent.cfg.actions.vibeHeart.int32)
        return
      if assemblerNearby.isSome():
        let action = agent.cfg.aStar(agent.location, assemblerNearby.get(), agent.map)
        if action.isSome():
          doAction(action.get().int32)
          return

    if agent.resourceFocus.isSome():
      let focusKind = agent.resourceFocus.get()
      if inventory.getOrDefault(focusKind, 0) < resourceQuota(focusKind):
        if vibe != agent.vibeFor(focusKind):
          doAction(agent.vibeFor(focusKind))
          return
        let action = agent.seekExtractor(focusKind)
        if action.isSome():
          doAction(action.get())
          return

    let unseenNearby = agent.cfg.getNearbyUnseen(agent.location, agent.map, agent.seen)
    if unseenNearby.isSome():
      let action = agent.cfg.aStar(agent.location, unseenNearby.get(), agent.map)
      if action.isSome():
        doAction(action.get().int32)
        return

    doAction(agent.randomMove())
  except:
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
