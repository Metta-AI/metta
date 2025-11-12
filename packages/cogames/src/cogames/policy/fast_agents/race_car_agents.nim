import
  std/[heapqueue, options, random, sets, strformat, strutils, tables],
  genny, jsony,
  common

type
  ResourceKind = enum
    rkEnergy,
    rkCarbon,
    rkOxygen,
    rkGermanium,
    rkSilicon

  TaskKind = enum
    tkExplore,
    tkCollect,
    tkCollectScarce,
    tkReturnInventory,
    tkCoordinateHeart,
    tkHoldAssembler,
    tkDepositToChest,
    tkWithdrawResources,
    tkDepositHeart

  TaskPayload = object
    target: Option[Location]
    resource: Option[ResourceKind]
    stashAmount: int

  PriorityTask = object
    priority: int
    kind: TaskKind
    payload: TaskPayload
    note: string

  TaskQueue = object
    heap: HeapQueue[PriorityTask]

  AgentSnapshot = object
    inventory: Table[ResourceKind, int]
    inventoryCapacity: int
    maxStackPerResource: int
    chest: Table[ResourceKind, int]
    chestTarget: Table[ResourceKind, int]
    assemblerLocation: Option[Location]
    chestLocation: Option[Location]
    homeLocation: Option[Location]
    friendsNearby: int
    friendsAtAssembler: int
    scavengedDropNearby: bool
    readyToCraft: bool
    location: Location
    seenChest: bool
    seenAssembler: bool
    heartInventory: int

  PriorityAgent = ref object
    queue: TaskQueue
    lastSnapshot: Option[AgentSnapshot]
    randomizer: Rand

  SignalActionKind = enum
    sakVibeCarbon,
    sakVibeOxygen,
    sakVibeGermanium,
    sakVibeSilicon,
    sakVibeWall,
    sakVibeAssembler,
    sakVibeChest,
    sakVibeGear

  InitSignalKind = enum
    iskResource,
    iskDirection

  InitSignal = object
    name: string
    kind: InitSignalKind
    actionKind: SignalActionKind
    resource: Option[ResourceKind]
    heading: Option[Location]

  RaceCarAgent* = ref object
    agentId*: int
    map: Table[Location, seq[FeatureValue]]
    seen: HashSet[Location]
    cfg: Config
    random: Rand
    location: Location
    planner: PriorityAgent
    chestInventory: Table[ResourceKind, int]
    chestTargets: Table[ResourceKind, int]
    chestHearts: int
    assemblerLocation: Option[Location]
    chestLocation: Option[Location]
    homeLocation: Option[Location]
    lastInventorySample: Table[ResourceKind, int]
    lastHeartInventory: int
    lastActions: seq[int]
    initSignalName: string
    initHeading: Option[Location]
    headingObjective: Option[Location]
    initResourceFocus: Option[ResourceKind]
    initPhaseTicks: int
    initPhaseDone: bool
    lastActions: seq[int]

const
  defaultInventoryCapacity = 100
  maxStackSizePerResource = 50
  defaultChestTarget = 100
  HeartResources = [rkCarbon, rkOxygen, rkGermanium, rkSilicon]
  InitPhaseDuration = 6

const InitSignals: array[0 .. 7, InitSignal] = [
  InitSignal(
    name: "resource_carbon",
    kind: iskResource,
    actionKind: sakVibeCarbon,
    resource: some(rkCarbon),
    heading: none(Location)
  ),
  InitSignal(
    name: "resource_oxygen",
    kind: iskResource,
    actionKind: sakVibeOxygen,
    resource: some(rkOxygen),
    heading: none(Location)
  ),
  InitSignal(
    name: "resource_germanium",
    kind: iskResource,
    actionKind: sakVibeGermanium,
    resource: some(rkGermanium),
    heading: none(Location)
  ),
  InitSignal(
    name: "resource_silicon",
    kind: iskResource,
    actionKind: sakVibeSilicon,
    resource: some(rkSilicon),
    heading: none(Location)
  ),
  InitSignal(
    name: "direction_north",
    kind: iskDirection,
    actionKind: sakVibeWall,
    resource: none(ResourceKind),
    heading: some(Location(x: 0, y: -1))
  ),
  InitSignal(
    name: "direction_south",
    kind: iskDirection,
    actionKind: sakVibeAssembler,
    resource: none(ResourceKind),
    heading: some(Location(x: 0, y: 1))
  ),
  InitSignal(
    name: "direction_east",
    kind: iskDirection,
    actionKind: sakVibeChest,
    resource: none(ResourceKind),
    heading: some(Location(x: 1, y: 0))
  ),
  InitSignal(
    name: "direction_west",
    kind: iskDirection,
    actionKind: sakVibeGear,
    resource: none(ResourceKind),
    heading: some(Location(x: -1, y: 0))
  )
]

var initSignalOwners = initTable[string, int]()
var initSignalAssignments = initTable[int, string]()

proc resourceQuota(resource: ResourceKind): int =
  case resource
  of rkSilicon:
    50
  of rkCarbon, rkOxygen, rkGermanium:
    30
  else:
    0

proc chestHasBundle(snapshot: AgentSnapshot): bool =
  for resource in HeartResources:
    if snapshot.chest.getOrDefault(resource, 0) < resourceQuota(resource):
      return false
  true

proc inventoryReadyForHeart(snapshot: AgentSnapshot): bool =
  for resource in HeartResources:
    if snapshot.inventory.getOrDefault(resource, 0) < resourceQuota(resource):
      return false
  true

proc currentCarry(snapshot: AgentSnapshot): Option[(ResourceKind, int)] =
  var bestAmount = 0
  var bestResource = rkCarbon
  for resource in HeartResources:
    let amount = snapshot.inventory.getOrDefault(resource, 0)
    if amount > bestAmount:
      bestAmount = amount
      bestResource = resource
  if bestAmount > 0:
    return some((bestResource, bestAmount))
  none((ResourceKind, int))

proc missingResources(snapshot: AgentSnapshot): seq[ResourceKind] =
  for resource in HeartResources:
    if snapshot.chest.getOrDefault(resource, 0) < resourceQuota(resource):
      result.add(resource)

proc chooseMissingResource(planner: PriorityAgent, snapshot: AgentSnapshot): ResourceKind =
  let missing = snapshot.missingResources()
  if missing.len == 0:
    return rkCarbon
  if missing.len == 1:
    return missing[0]
  let idx = planner.randomizer.rand(0 ..< missing.len)
  missing[idx]

proc `<`(a, b: PriorityTask): bool =
  if a.priority == b.priority:
    return ord(a.kind) > ord(b.kind)
  a.priority < b.priority

proc emptyQueue(): TaskQueue =
  TaskQueue(heap: initHeapQueue[PriorityTask]())

proc pushTask(queue: var TaskQueue, task: PriorityTask) =
  queue.heap.push(task)

proc popTask(queue: var TaskQueue): Option[PriorityTask] =
  if queue.heap.len == 0:
    return none(PriorityTask)
  some(queue.heap.pop())

proc newPriorityAgent(seed: int): PriorityAgent =
  PriorityAgent(queue: emptyQueue(), randomizer: initRand(seed))

proc enqueueTask(tasks: var seq[PriorityTask], kind: TaskKind, priority: int, payload: TaskPayload, note: string) =
  tasks.add(PriorityTask(priority: priority, kind: kind, payload: payload, note: note))

proc buildPriorityPlan(planner: PriorityAgent, snapshot: AgentSnapshot): seq[PriorityTask] =
  var tasks: seq[PriorityTask]

  if snapshot.heartInventory > 0:
    tasks.enqueueTask(
      tkDepositHeart,
      priority = 120,
      payload = TaskPayload(target: snapshot.chestLocation),
      note = "Carrying a heart; deposit to chest"
    )
    return tasks

  if inventoryReadyForHeart(snapshot) and snapshot.seenAssembler:
    tasks.enqueueTask(
      tkCoordinateHeart,
      priority = 110,
      payload = TaskPayload(target: snapshot.assemblerLocation),
      note = "Inventory stocked; craft a heart"
    )
    return tasks

  if chestHasBundle(snapshot) and snapshot.seenChest:
    tasks.enqueueTask(
      tkWithdrawResources,
      priority = 100,
      payload = TaskPayload(target: snapshot.chestLocation),
      note = "Chest stocked; withdraw ingredients"
    )
    return tasks

  let carrying = snapshot.currentCarry()
  if carrying.isSome:
    let (resource, amount) = carrying.get()
    if amount >= resourceQuota(resource) and snapshot.seenChest:
      tasks.enqueueTask(
        tkDepositToChest,
        priority = 90,
        payload = TaskPayload(target: snapshot.chestLocation, resource: some(resource), stashAmount: amount),
        note = &"Delivering {amount} units of {resource}"
      )
      return tasks
    tasks.enqueueTask(
      tkCollectScarce,
      priority = 80,
      payload = TaskPayload(resource: some(resource), stashAmount: resourceQuota(resource)),
      note = &"Continue gathering {resource}"
    )
    return tasks

  let missing = snapshot.missingResources()
  if missing.len > 0:
    let chosen = planner.chooseMissingResource(snapshot)
    tasks.enqueueTask(
      tkCollectScarce,
      priority = 70,
      payload = TaskPayload(resource: some(chosen), stashAmount: resourceQuota(chosen)),
      note = &"Seeking extractor for {chosen}"
    )
    return tasks

  tasks.enqueueTask(
    tkExplore,
    priority = 10,
    payload = TaskPayload(),
    note = "Fallback exploration"
  )

  tasks

proc plan(planner: PriorityAgent, snapshot: AgentSnapshot) =
  planner.lastSnapshot = some(snapshot)
  planner.queue = emptyQueue()
  for task in planner.buildPriorityPlan(snapshot):
    planner.queue.pushTask(task)

proc nextTask(planner: PriorityAgent): Option[PriorityTask] =
  planner.queue.popTask()

proc cloneResourceTable(source: Table[ResourceKind, int]): Table[ResourceKind, int] =
  result = initTable[ResourceKind, int]()
  for key, value in source.pairs:
    result[key] = value

proc chebyshevDistance(a, b: Location): int =
  max(abs(a.x - b.x), abs(a.y - b.y))

proc signalInfo(name: string): InitSignal =
  for signal in InitSignals:
    if signal.name == name:
      return signal
  InitSignals[0]

proc resolveSignalAction(cfg: Config, actionKind: SignalActionKind): int =
  case actionKind
  of sakVibeCarbon:
    cfg.actions.vibeCarbon
  of sakVibeOxygen:
    cfg.actions.vibeOxygen
  of sakVibeGermanium:
    cfg.actions.vibeGermanium
  of sakVibeSilicon:
    cfg.actions.vibeSilicon
  of sakVibeWall:
    cfg.actions.vibeWall
  of sakVibeAssembler:
    cfg.actions.vibeAssembler
  of sakVibeChest:
    cfg.actions.vibeChest
  of sakVibeGear:
    cfg.actions.vibeGear

proc releaseInitSignal(agent: RaceCarAgent) =
  if agent.initSignalName.len > 0:
    if initSignalOwners.getOrDefault(agent.initSignalName, agent.agentId) == agent.agentId:
      initSignalOwners.del(agent.initSignalName)
    initSignalAssignments.del(agent.agentId)
    agent.initSignalName = ""

proc assignInitSignal(agent: RaceCarAgent) =
  var available: seq[InitSignal]
  for signal in InitSignals:
    if not initSignalOwners.hasKey(signal.name):
      available.add(signal)
  if available.len == 0:
    available = @InitSignals
  let idx = agent.random.rand(0 ..< available.len)
  let choice = available[idx]
  initSignalAssignments[agent.agentId] = choice.name
  initSignalOwners[choice.name] = agent.agentId
  agent.initSignalName = choice.name
  agent.initHeading = choice.heading
  if choice.kind == iskResource:
    agent.initResourceFocus = choice.resource
  else:
    agent.initResourceFocus = none(ResourceKind)
  agent.headingObjective = none(Location)

proc ensureUniqueSignal(agent: RaceCarAgent) =
  if agent.initSignalName.len == 0 or not initSignalAssignments.hasKey(agent.agentId):
    assignInitSignal(agent)
    return
  let owner = initSignalOwners.getOrDefault(agent.initSignalName, agent.agentId)
  if owner != agent.agentId:
    assignInitSignal(agent)

proc initPriorityFields(agent: RaceCarAgent) =
  agent.planner = newPriorityAgent(agent.agentId)
  agent.chestInventory = initTable[ResourceKind, int]()
  agent.chestTargets = initTable[ResourceKind, int]()
  for resource in HeartResources:
    agent.chestTargets[resource] = defaultChestTarget
  agent.assemblerLocation = none(Location)
  agent.chestLocation = none(Location)
  agent.homeLocation = none(Location)
  agent.chestHearts = 0
  agent.lastInventorySample = initTable[ResourceKind, int]()
  agent.lastHeartInventory = 0
  agent.initPhaseTicks = 0
  agent.initPhaseDone = false
  agent.headingObjective = none(Location)
  agent.initHeading = none(Location)
  agent.initResourceFocus = none(ResourceKind)

proc recordInventorySnapshot(agent: RaceCarAgent, snapshot: var AgentSnapshot) =
  if agent.lastInventorySample.len == 0:
    agent.lastInventorySample = cloneResourceTable(snapshot.inventory)
    agent.lastHeartInventory = snapshot.heartInventory
    snapshot.chest = cloneResourceTable(agent.chestInventory)
    return

  let atChest = snapshot.seenChest and snapshot.chestLocation.isSome and snapshot.location == snapshot.chestLocation.get()
  if atChest:
    for resource in ResourceKind:
      let previous = agent.lastInventorySample.getOrDefault(resource, 0)
      let current = snapshot.inventory.getOrDefault(resource, 0)
      if current < previous:
        agent.chestInventory[resource] = agent.chestInventory.getOrDefault(resource, 0) + (previous - current)
      elif current > previous:
        let newValue = agent.chestInventory.getOrDefault(resource, 0) - (current - previous)
        agent.chestInventory[resource] = max(0, newValue)

    let prevHeart = agent.lastHeartInventory
    let currHeart = snapshot.heartInventory
    if currHeart < prevHeart:
      agent.chestHearts += prevHeart - currHeart
    elif currHeart > prevHeart:
      agent.chestHearts = max(0, agent.chestHearts - (currHeart - prevHeart))

  agent.lastInventorySample = cloneResourceTable(snapshot.inventory)
  agent.lastHeartInventory = snapshot.heartInventory
  snapshot.chest = cloneResourceTable(agent.chestInventory)

proc updateStructureKnowledge(agent: RaceCarAgent) =
  for location, _ in agent.map:
    let tag = agent.cfg.getTag(agent.map, location)
    if tag == agent.cfg.tags.assembler:
      agent.assemblerLocation = some(location)
      if agent.homeLocation.isNone:
        agent.homeLocation = agent.assemblerLocation
    elif tag == agent.cfg.tags.chest:
      agent.chestLocation = some(location)

proc buildInventory(agent: RaceCarAgent, visible: Table[Location, seq[FeatureValue]]): Table[ResourceKind, int] =
  result = initTable[ResourceKind, int]()
  result[rkEnergy] = agent.cfg.getInventory(visible, agent.cfg.features.invEnergy)
  result[rkCarbon] = agent.cfg.getInventory(visible, agent.cfg.features.invCarbon)
  result[rkOxygen] = agent.cfg.getInventory(visible, agent.cfg.features.invOxygen)
  result[rkGermanium] = agent.cfg.getInventory(visible, agent.cfg.features.invGermanium)
  result[rkSilicon] = agent.cfg.getInventory(visible, agent.cfg.features.invSilicon)

proc countVisibleFriends(agent: RaceCarAgent, visible: Table[Location, seq[FeatureValue]]): int =
  for location, _ in visible:
    if location.x == 0 and location.y == 0:
      continue
    if agent.cfg.getTag(visible, location) == agent.cfg.tags.agent:
      inc result

proc countFriendsAtAssembler(agent: RaceCarAgent): int =
  if agent.assemblerLocation.isNone:
    return 0
  let assemblerLoc = agent.assemblerLocation.get()
  for location, _ in agent.map:
    if agent.cfg.getTag(agent.map, location) == agent.cfg.tags.agent and location != assemblerLoc:
      if chebyshevDistance(location, assemblerLoc) <= 1:
        inc result

proc readyToCraft(inventory: Table[ResourceKind, int]): bool =
  for resource in HeartResources:
    if inventory.getOrDefault(resource, 0) < resourceQuota(resource):
      return false
  true

proc buildSnapshot(agent: RaceCarAgent, visible: Table[Location, seq[FeatureValue]]): AgentSnapshot =
  result.inventory = agent.buildInventory(visible)
  result.inventoryCapacity = defaultInventoryCapacity
  result.maxStackPerResource = maxStackSizePerResource
  result.chest = cloneResourceTable(agent.chestInventory)
  result.chestTarget = cloneResourceTable(agent.chestTargets)
  result.assemblerLocation = agent.assemblerLocation
  result.chestLocation = agent.chestLocation
  result.homeLocation = agent.homeLocation
  result.friendsNearby = agent.countVisibleFriends(visible)
  result.friendsAtAssembler = agent.countFriendsAtAssembler()
  result.scavengedDropNearby = false
  result.readyToCraft = readyToCraft(result.inventory)
  result.location = agent.location
  result.seenChest = agent.chestLocation.isSome
  result.seenAssembler = agent.assemblerLocation.isSome
  result.heartInventory = agent.cfg.getInventory(visible, agent.cfg.features.invHeart)

proc resourceTypeId(agent: RaceCarAgent, resource: ResourceKind): int =
  case resource
  of rkCarbon:
    agent.cfg.tags.carbonExtractor
  of rkOxygen:
    agent.cfg.tags.oxygenExtractor
  of rkGermanium:
    agent.cfg.tags.germaniumExtractor
  of rkSilicon:
    agent.cfg.tags.siliconExtractor
  of rkEnergy:
    agent.cfg.tags.charger

proc findStructure(agent: RaceCarAgent, typeId: int): Option[Location] =
  for location, _ in agent.map:
    if agent.cfg.getTag(agent.map, location) == typeId:
      return some(location)
  none(Location)

proc targetForResource(agent: RaceCarAgent, resource: ResourceKind): Option[Location] =
  let typeId = agent.resourceTypeId(resource)
  var bestScore = int.low
  var best: Option[Location] = none(Location)
  let baseLocation = if agent.homeLocation.isSome: agent.homeLocation.get() else: agent.location
  for location, _ in agent.map:
    if agent.cfg.getTag(agent.map, location) == typeId:
      let distFromBase = chebyshevDistance(location, baseLocation)
      let distFromAgent = chebyshevDistance(location, agent.location)
      let score = distFromBase * 10 - distFromAgent
      if score > bestScore:
        bestScore = score
        best = some(location)
  if best.isSome:
    return best
  agent.findStructure(typeId)

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
    else:
      return agent.cfg.actions.moveWest.int32
  else:
    if dy > 0:
      return agent.cfg.actions.moveSouth.int32
    else:
      return agent.cfg.actions.moveNorth.int32

proc moveTowards(agent: RaceCarAgent, target: Location): int32 =
  if target == agent.location:
    return agent.cfg.actions.noop.int32
  let pathAction = agent.cfg.aStar(agent.location, target, agent.map)
  if pathAction.isSome:
    return pathAction.get().int32
  agent.moveGreedy(target)

proc exploreAction(agent: RaceCarAgent): int32 =
  let unseen = agent.cfg.getNearbyUnseen(agent.location, agent.map, agent.seen)
  if unseen.isSome:
    let action = agent.cfg.aStar(agent.location, unseen.get(), agent.map)
    if action.isSome:
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
  while true:
    var replaced = false
    if agent.lastActions.len >= 2:
      if candidate == agent.cfg.actions.moveWest and
          agent.lastActions[^1] == agent.cfg.actions.moveEast and
          agent.lastActions[^2] == agent.cfg.actions.moveWest and
          agent.random.rand(1 .. 2) == 1:
        candidate = agent.cfg.actions.noop
        replaced = true
      elif candidate == agent.cfg.actions.moveEast and
          agent.lastActions[^1] == agent.cfg.actions.moveWest and
          agent.lastActions[^2] == agent.cfg.actions.moveEast and
          agent.random.rand(1 .. 2) == 1:
        candidate = agent.cfg.actions.noop
        replaced = true
      elif candidate == agent.cfg.actions.moveNorth and
          agent.lastActions[^1] == agent.cfg.actions.moveSouth and
          agent.lastActions[^2] == agent.cfg.actions.moveNorth and
          agent.random.rand(1 .. 2) == 1:
        candidate = agent.cfg.actions.noop
        replaced = true
      elif candidate == agent.cfg.actions.moveSouth and
          agent.lastActions[^1] == agent.cfg.actions.moveNorth and
          agent.lastActions[^2] == agent.cfg.actions.moveSouth and
          agent.random.rand(1 .. 2) == 1:
        candidate = agent.cfg.actions.noop
        replaced = true
    if not replaced:
      break

  agent.lastActions.add(candidate)
  if agent.lastActions.len > 16:
    agent.lastActions.delete(0)
  candidate.int32

proc interactWithChest(agent: RaceCarAgent, target: Option[Location]): int32 =
  if target.isSome:
    let coord = target.get()
    if coord == agent.location:
      return agent.cfg.actions.vibeChest.int32
    return agent.moveTowards(coord)
  agent.exploreAction()

proc finalizeInitHeading(agent: RaceCarAgent) =
  if agent.initHeading.isSome and agent.headingObjective.isNone:
    let heading = agent.initHeading.get()
    let objective = Location(x: agent.location.x + heading.x * 4, y: agent.location.y + heading.y * 4)
    agent.headingObjective = some(objective)

proc handleInitPhase(
  agent: RaceCarAgent,
  actions: ptr UncheckedArray[int32],
  actionIndex: int
): bool =
  if agent.initPhaseDone:
    return false

  ensureUniqueSignal(agent)
  let signal = signalInfo(agent.initSignalName)
  let vibeAction = resolveSignalAction(agent.cfg, signal.actionKind)
  let stabilized = agent.stabilizeAction(vibeAction.int32)
  actions[actionIndex] = stabilized
  inc agent.initPhaseTicks

  let owner = initSignalOwners.getOrDefault(agent.initSignalName, agent.agentId)
  if agent.initPhaseTicks >= InitPhaseDuration and owner == agent.agentId:
    agent.initPhaseDone = true
    finalizeInitHeading(agent)
  return true

proc chooseAction(agent: RaceCarAgent, taskOpt: Option[PriorityTask]): int32 =
  if taskOpt.isNone:
    return agent.exploreAction()
  let task = taskOpt.get()
  let targetCoord = task.payload.target
  case task.kind
  of tkCoordinateHeart:
    if targetCoord.isSome and targetCoord.get() == agent.location:
      return agent.cfg.actions.vibeHeart.int32
    elif targetCoord.isSome:
      return agent.moveTowards(targetCoord.get())
    else:
      return agent.cfg.actions.vibeHeart.int32
  of tkHoldAssembler:
    if targetCoord.isSome:
      return agent.moveTowards(targetCoord.get())
    return agent.cfg.actions.vibeAssembler.int32
  of tkReturnInventory:
    if targetCoord.isSome:
      return agent.moveTowards(targetCoord.get())
    return agent.exploreAction()
  of tkDepositToChest:
    return agent.interactWithChest(targetCoord)
  of tkWithdrawResources:
    return agent.interactWithChest(targetCoord)
  of tkDepositHeart:
    return agent.interactWithChest(targetCoord)
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

proc planAndSelectAction(agent: RaceCarAgent, visible: Table[Location, seq[FeatureValue]]): (Option[PriorityTask], int32) =
  agent.updateStructureKnowledge()
  var snapshot = agent.buildSnapshot(visible)
  agent.recordInventorySnapshot(snapshot)
  agent.planner.plan(snapshot)
  let plannedTask = agent.planner.nextTask()
  let chosenAction = agent.chooseAction(plannedTask)
  (plannedTask, chosenAction)

proc newRaceCarAgent*(agentId: int, environmentConfig: string): RaceCarAgent {.raises: [].} =
  echo "Creating new race car agent ", agentId
  var config = parseConfig(environmentConfig)
  result = RaceCarAgent(agentId: agentId, cfg: config)
  result.random = initRand(agentId)
  result.initPriorityFields()
  result.lastActions = @[]

proc reset*(agent: RaceCarAgent) =
  echo "Resetting race car agent ", agent.agentId
  releaseInitSignal(agent)
  agent.map.clear()
  agent.seen.clear()
  agent.location = Location(x: 0, y: 0)
  agent.initPriorityFields()
  agent.lastActions.setLen(0)

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
    echo "Looks like we are lost?"
    echo "  current location: ", agent.location.x, ", ", agent.location.y
    echo "  best location: ", bestLocation.x, ", ", bestLocation.y
  else:
    agent.location =  bestLocation
    for x in -5 .. 5:
      for y in -5 .. 5:
        let visibleLocation = Location(x: x, y: y)
        let location = Location(x: x + agent.location.x, y: y + agent.location.y)
        if visibleLocation in visible:
          agent.map[location] = visible[visibleLocation]
        else:
          agent.map[location] = @[]
        agent.seen.incl(location)

proc raceCarStepInternal(
  agent: RaceCarAgent,
  numTokens: int,
  sizeToken: int,
  rawObservation: pointer,
  actions: ptr UncheckedArray[int32],
  actionIndex: int
) {.raises: [].} =
  try:
    echo "Prioritizing race car agent ", agent.agentId
    let observations = cast[ptr UncheckedArray[uint8]](rawObservation)

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

    echo "current location: ", agent.location.x, ", ", agent.location.y
    echo "visible map:"
    agent.cfg.drawMap(map, initHashSet[Location]())
    updateMap(agent, map)
    echo "updated map:"
    agent.cfg.drawMap(agent.map, agent.seen)

    if handleInitPhase(agent, actions, actionIndex):
      return

    let vibe = agent.cfg.getVibe(map)
    echo "vibe: ", vibe

    let (plannedTask, chosenAction) = agent.planAndSelectAction(map)
    if plannedTask.isSome:
      let task = plannedTask.get()
      echo &"selected task {task.kind} priority {task.priority} note: {task.note}"

    let stabilized = agent.stabilizeAction(chosenAction)
    actions[actionIndex] = stabilized
    echo "taking action ", stabilized

  except:
    echo getCurrentException().getStackTrace()
    echo getCurrentExceptionMsg()
    quit()

proc step*(
  agent: RaceCarAgent,
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
  let agentObservation = cast[pointer](observations[agent.agentId * numTokens * sizeToken].addr)
  raceCarStepInternal(agent, numTokens, sizeToken, agentObservation, actions, agent.agentId)
