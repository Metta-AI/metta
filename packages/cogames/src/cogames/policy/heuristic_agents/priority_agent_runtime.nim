const
  defaultInventoryCapacity = 100
  maxStackSizePerResource = 50
  defaultChestTarget = 100

proc toCoord(location: Location): Coord =
  Coord(x: location.x, y: location.y)

proc locationOptionToCoord(loc: Option[Location]): Option[Coord] =
  if loc.isSome:
    return some(loc.get().toCoord())
  none(Coord)

proc chebyshevDistance(a, b: Location): int =
  max(abs(a.x - b.x), abs(a.y - b.y))

proc initPriorityFields(agent: HeuristicAgent) =
  agent.planner = newPriorityAgent()
  agent.chestInventory = initTable[ResourceKind, int]()
  agent.chestTargets = initTable[ResourceKind, int]()
  for resource in HeartResources:
    agent.chestTargets[resource] = defaultChestTarget
  agent.assemblerLocation = none(Location)
  agent.chestLocation = none(Location)
  agent.homeLocation = none(Location)

proc cloneResourceTable(source: Table[ResourceKind, int]): Table[ResourceKind, int] =
  result = initTable[ResourceKind, int]()
  for key, value in source.pairs:
    result[key] = value

proc updateStructureKnowledge(agent: HeuristicAgent) =
  ## Remember assembler/chest locations once seen.
  for location, featureValues in agent.map:
    for featureValue in featureValues:
      if featureValue.featureId != agent.features.typeId:
        continue
      if featureValue.value == agent.types.assembler:
        agent.assemblerLocation = some(location)
        if agent.homeLocation.isNone:
          agent.homeLocation = agent.assemblerLocation
      elif featureValue.value == agent.types.chest:
        agent.chestLocation = some(location)

proc buildInventory(agent: HeuristicAgent, visible: Table[Location, seq[FeatureValue]]): Table[ResourceKind, int] =
  result = initTable[ResourceKind, int]()
  result[rkEnergy] = agent.getInventory(visible, agent.features.invEnergy)
  result[rkCarbon] = agent.getInventory(visible, agent.features.invCarbon)
  result[rkOxygen] = agent.getInventory(visible, agent.features.invOxygen)
  result[rkGermanium] = agent.getInventory(visible, agent.features.invGermanium)
  result[rkSilicon] = agent.getInventory(visible, agent.features.invSilicon)

proc countVisibleFriends(agent: HeuristicAgent, visible: Table[Location, seq[FeatureValue]]): int =
  for location, _ in visible:
    if location.x == 0 and location.y == 0:
      continue
    if agent.getTypeId(visible, location) == agent.types.agent:
      inc result

proc countFriendsAtAssembler(agent: HeuristicAgent): int =
  if agent.assemblerLocation.isNone:
    return 0
  let assemblerLoc = agent.assemblerLocation.get()
  for location, _ in agent.map:
    if agent.getTypeId(agent.map, location) == agent.types.agent and location != assemblerLoc:
      if chebyshevDistance(location, assemblerLoc) <= 1:
        inc result

proc readyToCraft(inventory: Table[ResourceKind, int]): bool =
  var total = 0
  var minResource = int.high
  for resource in HeartResources:
    let amount = inventory.getOrDefault(resource, 0)
    total += amount
    if amount < minResource:
      minResource = amount
  total >= 50 and minResource >= 10

proc buildSnapshot(agent: HeuristicAgent, visible: Table[Location, seq[FeatureValue]]): AgentSnapshot =
  result.inventory = buildInventory(agent, visible)
  result.inventoryCapacity = defaultInventoryCapacity
  result.maxStackPerResource = maxStackSizePerResource
  result.chest = cloneResourceTable(agent.chestInventory)
  result.chestTarget = cloneResourceTable(agent.chestTargets)
  result.assemblerLocation = locationOptionToCoord(agent.assemblerLocation)
  result.chestLocation = locationOptionToCoord(agent.chestLocation)
  result.homeLocation = locationOptionToCoord(agent.homeLocation)
  result.friendsNearby = agent.countVisibleFriends(visible)
  result.friendsAtAssembler = agent.countFriendsAtAssembler()
  result.scavengedDropNearby = false
  result.readyToCraft = readyToCraft(result.inventory)
  result.location = agent.location.toCoord()
  result.seenChest = agent.chestLocation.isSome
  result.seenAssembler = agent.assemblerLocation.isSome

proc resourceTypeId(agent: HeuristicAgent, resource: ResourceKind): int =
  case resource:
  of rkCarbon:
    agent.types.carbonExtractor
  of rkOxygen:
    agent.types.oxygenExtractor
  of rkGermanium:
    agent.types.germaniumExtractor
  of rkSilicon:
    agent.types.siliconExtractor
  of rkEnergy:
    agent.types.charger

proc findStructure(agent: HeuristicAgent, typeId: int): Option[Coord] =
  for location, _ in agent.map:
    if agent.getTypeId(agent.map, location) == typeId:
      return some(location.toCoord())
  none(Coord)

proc targetForResource(agent: HeuristicAgent, resource: ResourceKind): Option[Coord] =
  findStructure(agent, agent.resourceTypeId(resource))

proc randomMove(agent: HeuristicAgent): int32 =
  case agent.random.rand(0 .. 3)
  of 0:
    agent.actions.moveNorth.int32
  of 1:
    agent.actions.moveSouth.int32
  of 2:
    agent.actions.moveWest.int32
  else:
    agent.actions.moveEast.int32

proc moveTowards(agent: HeuristicAgent, target: Coord): int32 =
  let dx = target.x - agent.location.x
  let dy = target.y - agent.location.y
  if dx == 0 and dy == 0:
    return agent.actions.noop.int32
  if abs(dx) >= abs(dy):
    if dx > 0:
      return agent.actions.moveEast.int32
    else:
      return agent.actions.moveWest.int32
  else:
    if dy > 0:
      return agent.actions.moveSouth.int32
    else:
      return agent.actions.moveNorth.int32

proc exploreAction(agent: HeuristicAgent): int32 =
  ## Bias toward unseen tiles; fallback to random movement.
  var candidates: seq[(Location, int32)]
  let north = Location(x: agent.location.x, y: agent.location.y - 1)
  let south = Location(x: agent.location.x, y: agent.location.y + 1)
  let west = Location(x: agent.location.x - 1, y: agent.location.y)
  let east = Location(x: agent.location.x + 1, y: agent.location.y)
  if north notin agent.seen:
    candidates.add((north, agent.actions.moveNorth.int32))
  if south notin agent.seen:
    candidates.add((south, agent.actions.moveSouth.int32))
  if west notin agent.seen:
    candidates.add((west, agent.actions.moveWest.int32))
  if east notin agent.seen:
    candidates.add((east, agent.actions.moveEast.int32))
  if candidates.len == 0:
    return agent.randomMove()
  let idx = agent.random.rand(0 ..< candidates.len)
  candidates[idx][1]

proc chooseAction(agent: HeuristicAgent, taskOpt: Option[PriorityTask]): int32 =
  if taskOpt.isNone:
    return agent.exploreAction()
  let task = taskOpt.get()
  let targetCoord = task.payload.target
  case task.kind
  of tkCoordinateHeart:
    if targetCoord.isSome and targetCoord.get() == agent.location.toCoord():
      return agent.actions.vibeHeart.int32
    elif targetCoord.isSome:
      return agent.moveTowards(targetCoord.get())
    else:
      return agent.actions.vibeHeart.int32
  of tkHoldAssembler:
    if targetCoord.isSome:
      return agent.moveTowards(targetCoord.get())
    return agent.actions.vibeAssembler.int32
  of tkReturnInventory:
    if targetCoord.isSome:
      return agent.moveTowards(targetCoord.get())
    return agent.exploreAction()
  of tkDepositToChest:
    if targetCoord.isSome:
      return agent.moveTowards(targetCoord.get())
    return agent.actions.vibeChest.int32
  of tkCollectScarce:
    if task.payload.resource.isSome:
      let locus = agent.targetForResource(task.payload.resource.get())
      if locus.isSome:
        return agent.moveTowards(locus.get())
    return agent.exploreAction()
  of tkCollect:
    return agent.randomMove()
  of tkExplore:
    return agent.exploreAction()

proc planAndSelectAction(agent: HeuristicAgent, visible: Table[Location, seq[FeatureValue]]): (Option[PriorityTask], int32) =
  updateStructureKnowledge(agent)
  let snapshot = agent.buildSnapshot(visible)
  agent.planner.plan(snapshot)
  let plannedTask = agent.planner.nextTask()
  let chosenAction = agent.chooseAction(plannedTask)
  (plannedTask, chosenAction)
