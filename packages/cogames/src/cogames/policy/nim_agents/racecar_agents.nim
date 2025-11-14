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
    heartRequirements: Table[ResourceKind, int]
    energyRequirement: int
    resourceOrder: seq[ResourceKind]
    assemblerLocation: Option[Location]

  RaceCarPolicy* = ref object
    agents*: seq[RaceCarAgent]

const
  EnergyReserve = 50
  EnergyBuffer = 20
  RequirementBuffer = 1

var sharedExtractorOffsets: Table[ResourceKind, HashSet[Location]] = initTable[ResourceKind, HashSet[Location]]()

proc buildResourceOrder(agentId: int): seq[ResourceKind] =
  let base = @[rkCarbon, rkOxygen, rkGermanium, rkSilicon]
  result = @[]
  if base.len == 0:
    return
  let offset = agentId mod base.len
  for i in 0 ..< base.len:
    result.add(base[(offset + i) mod base.len])

proc recipeRequirement(agent: RaceCarAgent, kind: ResourceKind): int =
  if agent.heartRequirements.len > 0 and agent.heartRequirements.hasKey(kind):
    return agent.heartRequirements[kind]
  1

proc resourceRequirement(agent: RaceCarAgent, kind: ResourceKind): int =
  var base = recipeRequirement(agent, kind)
  if agent.heartRequirements.len > 0 and agent.heartRequirements.hasKey(kind) and
      agent.heartRequirements[kind] <= 0:
    return 0
  if base <= 0:
    base = 1
  base + RequirementBuffer

proc targetEnergy(agent: RaceCarAgent): int =
  if agent.energyRequirement > 0:
    return max(agent.energyRequirement + EnergyBuffer, EnergyReserve)
  EnergyReserve

proc resourceVibe(agent: RaceCarAgent, kind: ResourceKind): tuple[action: int, vibe: int] =
  case kind
  of rkCarbon:
    if agent.cfg.actions.vibeCarbonA != 0:
      return (agent.cfg.actions.vibeCarbonA, agent.cfg.vibes.carbonA)
    elif agent.cfg.actions.vibeCarbonB != 0:
      return (agent.cfg.actions.vibeCarbonB, agent.cfg.vibes.carbonB)
  of rkOxygen:
    if agent.cfg.actions.vibeOxygenA != 0:
      return (agent.cfg.actions.vibeOxygenA, agent.cfg.vibes.oxygenA)
    elif agent.cfg.actions.vibeOxygenB != 0:
      return (agent.cfg.actions.vibeOxygenB, agent.cfg.vibes.oxygenB)
  of rkGermanium:
    if agent.cfg.actions.vibeGermaniumA != 0:
      return (agent.cfg.actions.vibeGermaniumA, agent.cfg.vibes.germaniumA)
    elif agent.cfg.actions.vibeGermaniumB != 0:
      return (agent.cfg.actions.vibeGermaniumB, agent.cfg.vibes.germaniumB)
  of rkSilicon:
    if agent.cfg.actions.vibeSiliconA != 0:
      return (agent.cfg.actions.vibeSiliconA, agent.cfg.vibes.siliconA)
    elif agent.cfg.actions.vibeSiliconB != 0:
      return (agent.cfg.actions.vibeSiliconB, agent.cfg.vibes.siliconB)
  (0, agent.cfg.vibes.default)

proc heartAssembleVibe(agent: RaceCarAgent): tuple[action: int, vibe: int] =
  if agent.cfg.actions.vibeHeartA != 0:
    return (agent.cfg.actions.vibeHeartA, agent.cfg.vibes.heartA)
  elif agent.cfg.actions.vibeHeartB != 0:
    return (agent.cfg.actions.vibeHeartB, agent.cfg.vibes.heartB)
  (0, agent.cfg.vibes.default)

proc heartDepositVibe(agent: RaceCarAgent): tuple[action: int, vibe: int] =
  if agent.cfg.actions.vibeHeartB != 0:
    return (agent.cfg.actions.vibeHeartB, agent.cfg.vibes.heartB)
  elif agent.cfg.actions.vibeHeartA != 0:
    return (agent.cfg.actions.vibeHeartA, agent.cfg.vibes.heartA)
  (0, agent.cfg.vibes.default)

proc rememberExtractorOffset(kind: ResourceKind, offset: Location) =
  if not sharedExtractorOffsets.hasKey(kind):
    sharedExtractorOffsets[kind] = initHashSet[Location]()
  sharedExtractorOffsets[kind].incl(offset)

proc chooseExtractorTarget(agent: RaceCarAgent, kind: ResourceKind): Option[Location] =
  if agent.assemblerLocation.isNone():
    return none(Location)
  if sharedExtractorOffsets.hasKey(kind):
    var found = false
    var bestLocation = Location(x: 0, y: 0)
    var bestDistance = high(int)
    let assemblerLoc = agent.assemblerLocation.get()
    for offset in sharedExtractorOffsets[kind]:
      let candidate = assemblerLoc + offset
      let distance = manhattan(candidate, agent.location)
      if not found or distance < bestDistance:
        bestDistance = distance
        bestLocation = candidate
        found = true
    if found:
      return some(bestLocation)
  none(Location)

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

proc updateExtractorOffsets(agent: RaceCarAgent, visible: Table[Location, seq[FeatureValue]]) =
  if agent.assemblerLocation.isNone():
    return
  let assemblerLoc = agent.assemblerLocation.get()
  for relativeLocation, featureValues in visible:
    for featureValue in featureValues:
      if featureValue.featureId == agent.cfg.features.tag:
        for kind in ResourceKind:
          if featureValue.value == agent.extractorTag(kind):
            let globalLocation = agent.location + relativeLocation
            let offset = globalLocation - assemblerLoc
            rememberExtractorOffset(kind, offset)

proc protocolFeatureValue(features: seq[FeatureValue], featureId: int): int =
  ## Return the positive value associated with a protocol feature id, if present.
  if featureId == 0:
    return 0
  for featureValue in features:
    if featureValue.featureId == featureId and featureValue.value > 0:
      return featureValue.value
  0

proc updateHeartRequirements(agent: RaceCarAgent) =
  if agent.map.len == 0:
    return
  let assemblerNearby = agent.cfg.getNearby(agent.location, agent.map, agent.cfg.tags.assembler)
  if assemblerNearby.isNone():
    return
  let assemblerLocation = assemblerNearby.get()
  if assemblerLocation notin agent.map:
    return
  let features = agent.map[assemblerLocation]
  let heartAmount = protocolFeatureValue(features, agent.cfg.features.protocolOutputHeart)
  if heartAmount == 0:
    return

  var updated = initTable[ResourceKind, int]()
  var energyRequirement = agent.energyRequirement
  let energyCost = protocolFeatureValue(features, agent.cfg.features.protocolInputEnergy)
  if energyCost > 0:
    energyRequirement = max(energyRequirement, energyCost)

  proc recordRequirement(kind: ResourceKind, featureId: int) =
    let value = protocolFeatureValue(features, featureId)
    if value > 0:
      updated[kind] = value

  recordRequirement(rkCarbon, agent.cfg.features.protocolInputCarbon)
  recordRequirement(rkOxygen, agent.cfg.features.protocolInputOxygen)
  recordRequirement(rkGermanium, agent.cfg.features.protocolInputGermanium)
  recordRequirement(rkSilicon, agent.cfg.features.protocolInputSilicon)

  if updated.len > 0:
    agent.heartRequirements = updated
  agent.energyRequirement = max(agent.energyRequirement, energyRequirement)

proc hasHeartIngredients(agent: RaceCarAgent, inventory: Table[ResourceKind, int], invEnergy: int): bool =
  for kind in ResourceKind:
    let required = recipeRequirement(agent, kind)
    if required <= 0:
      continue
    if inventory.getOrDefault(kind, 0) < required:
      return false
  invEnergy >= (if agent.energyRequirement > 0: agent.energyRequirement else: EnergyReserve)

proc updateFocus(agent: RaceCarAgent, inventory: Table[ResourceKind, int]) =
  if agent.resourceFocus.isSome:
    let current = agent.resourceFocus.get()
    if inventory.getOrDefault(current, 0) >= resourceRequirement(agent, current):
      agent.resourceFocus = none(ResourceKind)

  if agent.resourceFocus.isSome:
    return

  let priorityKinds = @[rkGermanium, rkSilicon]
  for kind in priorityKinds:
    let required = resourceRequirement(agent, kind)
    if required <= 0:
      continue
    if inventory.getOrDefault(kind, 0) < required:
      agent.resourceFocus = some(kind)
      return

  if agent.resourceOrder.len > 0:
    for kind in agent.resourceOrder:
      let required = resourceRequirement(agent, kind)
      if required <= 0:
        continue
      if inventory.getOrDefault(kind, 0) < required:
        agent.resourceFocus = some(kind)
        return
  else:
    for kind in ResourceKind:
      let required = resourceRequirement(agent, kind)
      if required <= 0:
        continue
      if inventory.getOrDefault(kind, 0) < required:
        agent.resourceFocus = some(kind)
        return

  agent.resourceFocus = none(ResourceKind)

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

type ExtractorPlan = tuple[action: Option[int32], waiting: bool]

proc seekExtractor(agent: RaceCarAgent, kind: ResourceKind): ExtractorPlan =
  var target = agent.chooseExtractorTarget(kind)
  if target.isNone():
    let tag = agent.extractorTag(kind)
    let nearby = agent.cfg.getNearby(agent.location, agent.map, tag)
    if nearby.isSome():
      target = nearby
      if agent.assemblerLocation.isSome():
        let offset = nearby.get() - agent.assemblerLocation.get()
        rememberExtractorOffset(kind, offset)
  if target.isSome():
    let location = target.get()
    let action = agent.cfg.aStar(agent.location, location, agent.map)
    if action.isSome():
      return (some(action.get().int32), false)
    elif agent.location == location:
      return (none(int32), true)
  (none(int32), false)

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
    agent.updateExtractorOffsets(map)
    agent.updateHeartRequirements()

    let vibe = agent.cfg.getVibe(map, Location(x: 0, y: 0))
    let invEnergy = agent.cfg.getInventory(map, agent.cfg.features.invEnergy)
    let invHeart = agent.cfg.getInventory(map, agent.cfg.features.invHeart)
    let inventory = agent.sampleInventory(map)

    let chargerNearby = agent.cfg.getNearby(agent.location, agent.map, agent.cfg.tags.charger)
    let assemblerNearby = agent.cfg.getNearby(agent.location, agent.map, agent.cfg.tags.assembler)

    if invEnergy < targetEnergy(agent) and chargerNearby.isSome():
      let action = agent.cfg.aStar(agent.location, chargerNearby.get(), agent.map)
      if action.isSome():
        doAction(action.get().int32)
        return

    if invHeart > 0:
      let (depositAction, depositVibe) = agent.heartDepositVibe()
      if depositAction != 0 and vibe != depositVibe:
        doAction(depositAction.int32)
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
    if assemblerNearby.isSome():
      agent.assemblerLocation = some(assemblerNearby.get())
      for keyLocation in keyLocations:
        let location = assemblerNearby.get() + keyLocation
        if location notin agent.seen:
          let action = agent.cfg.aStar(agent.location, location, agent.map)
          if action.isSome():
            doAction(action.get().int32)
            return

    agent.updateFocus(inventory)

    if agent.hasHeartIngredients(inventory, invEnergy):
      let (assembleAction, assembleVibe) = agent.heartAssembleVibe()
      if assembleAction != 0 and vibe != assembleVibe:
        doAction(assembleAction.int32)
        return
      if assemblerNearby.isSome():
        let action = agent.cfg.aStar(agent.location, assemblerNearby.get(), agent.map)
        if action.isSome():
          doAction(action.get().int32)
          return

    if agent.resourceFocus.isSome():
      let focusKind = agent.resourceFocus.get()
      if inventory.getOrDefault(focusKind, 0) < resourceRequirement(agent, focusKind):
        let (resourceAction, resourceVibeValue) = agent.resourceVibe(focusKind)
        if resourceAction != 0 and vibe != resourceVibeValue:
          doAction(resourceAction.int32)
          return
        let plan = agent.seekExtractor(focusKind)
        if plan.action.isSome():
          doAction(plan.action.get())
          return
        if plan.waiting:
          doAction(agent.cfg.actions.noop.int32)
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
  result.heartRequirements = initTable[ResourceKind, int]()
  result.energyRequirement = 0
  result.resourceOrder = buildResourceOrder(agentId)
  result.assemblerLocation = none(Location)

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
