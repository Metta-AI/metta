# Priority-task scaffolding for Nim agents.
#
# This module sketches the desired behavior queue for Nim heuristic agents
# without modifying the existing `heuristic_agents.nim` implementation yet.
# We compose the already-exported `HeuristicAgent` type so that we can feed
# observations into this planner while we gradually migrate shared utilities
# into a common module.

import std/[heapqueue, options, strformat, tables]

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
    tkDepositToChest

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

  PriorityAgent* = ref object
    queue: TaskQueue
    lastSnapshot: Option[AgentSnapshot]

const
  ## Resources that contribute directly to a heart recipe.
  HeartResources* = [rkCarbon, rkOxygen, rkGermanium, rkSilicon]

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

proc totalInventory(snapshot: AgentSnapshot): int =
  var total = 0
  for resource in ResourceKind:
    total += snapshot.inventory.getOrDefault(resource, 0)
  total

proc chestNeed(snapshot: AgentSnapshot, kind: ResourceKind): int =
  let target = snapshot.chestTarget.getOrDefault(kind, 0)
  let have = snapshot.chest.getOrDefault(kind, 0)
  result = target - have
  if result < 0:
    result = 0

proc leastSatisfiedResource(snapshot: AgentSnapshot): ResourceKind =
  var bestKind = rkCarbon
  var bestNeed = -1
  for resource in HeartResources:
    let need = chestNeed(snapshot, resource)
    if need > bestNeed:
      bestNeed = need
      bestKind = resource
  bestKind

proc chestAlmostReady(snapshot: AgentSnapshot): bool =
  var unmet = 0
  for resource in HeartResources:
    unmet += chestNeed(snapshot, resource)
  unmet <= 10 # Tunable buffer before declaring "almost ready".

proc halfInventory(snapshot: AgentSnapshot): bool =
  totalInventory(snapshot) >= (snapshot.inventoryCapacity div 2)

proc saturatedStack(snapshot: AgentSnapshot): bool =
  for resource in HeartResources:
    if snapshot.inventory.getOrDefault(resource, 0) >= snapshot.maxStackPerResource:
      return true
  false

proc newPriorityAgent*(): PriorityAgent =
  PriorityAgent(queue: emptyQueue())

proc enqueueTask(tasks: var seq[PriorityTask], kind: TaskKind, priority: int, payload: TaskPayload, note: string) =
  tasks.add(PriorityTask(priority: priority, kind: kind, payload: payload, note: note))

proc buildPriorityPlan(snapshot: AgentSnapshot): seq[PriorityTask] =
  ## Evaluate predicates and emit desired tasks ordered by priority.
  var tasks: seq[PriorityTask]

  if snapshot.readyToCraft and snapshot.friendsAtAssembler > 0 and snapshot.seenAssembler:
    tasks.enqueueTask(
      tkCoordinateHeart,
      priority = 100,
      payload = TaskPayload(target: snapshot.assemblerLocation),
      note = "Friends ready at assembler; coordinate heart"
    )
    tasks.enqueueTask(
      tkHoldAssembler,
      priority = 95,
      payload = TaskPayload(target: snapshot.assemblerLocation),
      note = "Hold assembler perimeter while waiting"
    )
    return tasks

  if snapshot.seenAssembler and chestAlmostReady(snapshot):
    tasks.enqueueTask(
      tkHoldAssembler,
      priority = 90,
      payload = TaskPayload(target: snapshot.assemblerLocation),
      note = "Chest nearly ready; stage near assembler"
    )

  if (halfInventory(snapshot) or saturatedStack(snapshot)) and snapshot.homeLocation.isSome:
    tasks.enqueueTask(
      tkReturnInventory,
      priority = 80,
      payload = TaskPayload(target: snapshot.homeLocation),
      note = "Inventory threshold reached; return home"
    )
    if snapshot.seenChest:
      tasks.enqueueTask(
        tkDepositToChest,
        priority = 70,
        payload = TaskPayload(target: snapshot.chestLocation),
        note = "No partners home; deposit to chest"
      )

  if snapshot.scavengedDropNearby and not saturatedStack(snapshot):
    tasks.enqueueTask(
      tkCollect,
      priority = 65,
      payload = TaskPayload(target: none(Coord)),
      note = "Loose resources nearby"
    )

  let scarce = leastSatisfiedResource(snapshot)
  tasks.enqueueTask(
    tkCollectScarce,
    priority = 60,
    payload = TaskPayload(resource: some(scarce)),
    note = &"Hunting scarce resource {scarce}"
  )

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
  for task in buildPriorityPlan(snapshot):
    planner.queue.pushTask(task)

proc nextTask*(planner: PriorityAgent): Option[PriorityTask] =
  planner.queue.popTask()

proc debugPlan*(tasks: seq[PriorityTask]): string =
  ## Handy helper for unit tests or logging.
  result = ""
  for idx, task in tasks:
    result.add(&"#{idx:02d} priority={task.priority} kind={task.kind} note={task.note}\n")
