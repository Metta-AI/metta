import
  std/[algorithm, tables, sets, heapqueue],
  vmath,
  common, replays

type
  PathNode* = object
    pos*: IVec2
    gCost*: int
    hCost*: int
    parent*: int

proc fCost*(node: PathNode): int =
  ## Calculate the total cost of a node.
  node.gCost + node.hCost

proc `<`(a, b: PathNode): bool =
  ## Comparison for heap (min-heap based on fCost).
  a.fCost < b.fCost

proc heuristic*(a, b: IVec2): int =
  ## Calculate the Manhattan distance heuristic.
  abs(a.x - b.x) + abs(a.y - b.y)

proc isWalkablePos*(pos: IVec2): bool =
  ## Check if a position is walkable.
  if pos.x < 0 or pos.x >= replay.mapSize[0] or pos.y < 0 or pos.y >= replay.mapSize[1]:
    return false
  let obj = getObjectAtLocation(pos)
  if obj != nil:
    let typeName = obj.typeName
    if typeName != "agent":
      return false
  return true

proc findPath*(start, goal: IVec2): seq[IVec2] =
  ## Find a path from start to goal using A* pathfinding.
  if start == goal:
    return @[]
  if not isWalkablePos(goal):
    return @[]
  var openHeap = initHeapQueue[PathNode]()
  openHeap.push(PathNode(pos: start, gCost: 0, hCost: heuristic(start, goal), parent: -1))
  var closedSet = initHashSet[IVec2]()
  var allNodes: seq[PathNode] = @[]
  while openHeap.len > 0:
    let current = openHeap.pop()
    if current.pos in closedSet:
      continue
    closedSet.incl(current.pos)
    let currentNodeIdx = allNodes.len
    allNodes.add(current)
    if current.pos == goal:
      var path: seq[IVec2] = @[]
      var node = current
      var nodeIdx = currentNodeIdx
      while nodeIdx >= 0 and node.parent >= 0:
        path.add(node.pos)
        nodeIdx = node.parent
        if nodeIdx >= 0:
          node = allNodes[nodeIdx]
        else:
          break
      path.add(start)
      path.reverse()
      return path
    const directions = [ivec2(0, -1), ivec2(0, 1), ivec2(-1, 0), ivec2(1, 0)]
    for dir in directions:
      let neighborPos = ivec2(current.pos.x + dir.x, current.pos.y + dir.y)
      if neighborPos.x < 0 or neighborPos.x >= replay.mapSize[0] or neighborPos.y < 0 or neighborPos.y >= replay.mapSize[1]:
        continue
      if neighborPos in closedSet:
        continue
      if not isWalkablePos(neighborPos):
        continue
      let newGCost = current.gCost + 1
      openHeap.push(PathNode(
        pos: neighborPos,
        gCost: newGCost,
        hCost: heuristic(neighborPos, goal),
        parent: currentNodeIdx
      ))
  return @[]

proc clearPath*(agentId: int) =
  ## Clear the path and objectives for an agent.
  agentPaths.del(agentId)
  agentObjectives.del(agentId)


proc recomputePath*(agentId: int, currentPos: IVec2) =
  ## Recompute the path for an agent through all their queued objectives.
  if not agentObjectives.hasKey(agentId) or agentObjectives[agentId].len == 0:
    agentPaths.del(agentId)
    return

  # Compute path actions through all path-based objectives (Move/Bump).
  var pathActions: seq[PathAction] = @[]
  var lastPos = currentPos

  for objIdx, objective in agentObjectives[agentId]:
    case objective.kind
    of Move:
      # For moving, path directly to the objective.
      let movePath = findPath(lastPos, objective.pos)
      if movePath.len == 0:
        clearPath(agentId)
        return
      # Convert positions to move actions.
      for pos in movePath:
        if pos != lastPos:
          pathActions.add(PathAction(kind: Move, pos: pos))
      lastPos = objective.pos
    of Bump:
      # For bumping, path to the specified approach position.
      let approachPos = ivec2(objective.pos.x + objective.approachDir.x, objective.pos.y + objective.approachDir.y)
      if not isWalkablePos(approachPos):
        # Approach position is not walkable, clear path.
        clearPath(agentId)
        return
      if lastPos != approachPos:
        # Path to the approach position.
        let movePath = findPath(lastPos, approachPos)
        if movePath.len == 0:
          clearPath(agentId)
          return
        for pos in movePath:
          if pos != lastPos:
            pathActions.add(PathAction(kind: Move, pos: pos))
      # Add the bump action.
      pathActions.add(PathAction(
        kind: Bump,
        bumpPos: objective.pos,
        bumpDir: ivec2(objective.pos.x - approachPos.x, objective.pos.y - approachPos.y),
      ))
      lastPos = approachPos
    of Vibe:
      # Add vibe as a path action to maintain synchronization.
      pathActions.add(PathAction(
        kind: Vibe,
        vibeActionId: objective.vibeActionId
      ))

  if pathActions.len > 0:
    agentPaths[agentId] = pathActions
  else:
    clearPath(agentId)