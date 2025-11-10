# Priority-task scaffolding for Nim agents.
#
# This module sketches the desired behavior queue for Nim heuristic agents
# without modifying the existing `heuristic_agents.nim` implementation yet.
# We compose the already-exported `HeuristicAgent` type so that we can feed
# observations into this planner while we gradually migrate shared utilities
# into a common module.

import std/[heapqueue, options, random, strformat, tables]

type
  ## Simplified coordinate used by the planner. We intentionally duplicate this
  ## instead of importing from `heuristic_agents` to avoid relying on private
  ## fields that will be extracted later.
  Coord* = object
    x*: int
    y*: int

  ResourceKind* = enum
    rkEnergy,
    rkCarbon,
    rkOxygen,
    rkGermanium,
    rkSilicon

  TaskKind* = enum
    tkExplore,
    tkCollect,
    tkCollectScarce,
    tkReturnInventory,
    tkCoordinateHeart,
    tkHoldAssembler,
    tkDepositToChest,
    tkWithdrawResources,
    tkDepositHeart

  TaskPayload* = object
    ## Optional payload used by downstream action-selection code. For now the
    ## planner only records minimal intent (targets and resource preference).
    target*: Option[Coord]
    resource*: Option[ResourceKind]
    stashAmount*: int

  PriorityTask* = object
    priority*: int
    kind*: TaskKind
    payload*: TaskPayload
    note*: string

  TaskQueue* = object
    heap: HeapQueue[PriorityTask]

  AgentSnapshot* = object
    ## Lightweight view of the world state required to evaluate predicates.
    inventory*: Table[ResourceKind, int]
    inventoryCapacity*: int
    maxStackPerResource*: int
    chest*: Table[ResourceKind, int]
    chestTarget*: Table[ResourceKind, int]
    assemblerLocation*: Option[Coord]
    chestLocation*: Option[Coord]
    homeLocation*: Option[Coord]
    friendsNearby*: int
    friendsAtAssembler*: int
    scavengedDropNearby*: bool
    readyToCraft*: bool
    location*: Coord
    seenChest*: bool
    seenAssembler*: bool
    heartInventory*: int

  PriorityAgent* = ref object
    queue: TaskQueue
    lastSnapshot: Option[AgentSnapshot]
    randomizer: Rand

const
  ## Resources that contribute directly to a heart recipe.
  HeartResources* = [rkCarbon, rkOxygen, rkGermanium, rkSilicon]

proc resourceQuota*(resource: ResourceKind): int =
  ## Required inventory stack before switching to deposit/craft behaviors.
  case resource
  of rkSilicon:
    50
  of rkCarbon, rkOxygen, rkGermanium:
    30
  else:
    0

proc chestHasBundle(snapshot: AgentSnapshot): bool =
  ## True when the chest holds enough of every ingredient to assemble one heart.
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
  ## heapqueue is a min-heap, so invert the comparison to treat a higher
  ## numerical priority as "larger".
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

proc newPriorityAgent*(seed: int): PriorityAgent =
  PriorityAgent(queue: emptyQueue(), randomizer: initRand(seed))

proc enqueueTask(tasks: var seq[PriorityTask], kind: TaskKind, priority: int, payload: TaskPayload, note: string) =
  tasks.add(PriorityTask(priority: priority, kind: kind, payload: payload, note: note))

proc buildPriorityPlan(planner: PriorityAgent, snapshot: AgentSnapshot): seq[PriorityTask] =
  ## Evaluate predicates and emit desired tasks ordered by priority.
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

proc plan*(planner: PriorityAgent, snapshot: AgentSnapshot) =
  planner.lastSnapshot = some(snapshot)
  planner.queue = emptyQueue()
  for task in planner.buildPriorityPlan(snapshot):
    planner.queue.pushTask(task)

proc nextTask*(planner: PriorityAgent): Option[PriorityTask] =
  planner.queue.popTask()

proc debugPlan*(tasks: seq[PriorityTask]): string =
  ## Handy helper for unit tests or logging.
  result = ""
  for idx, task in tasks:
    result.add(&"#{idx:02d} priority={task.priority} kind={task.kind} note={task.note}\n")
