import
  std/[algorithm, tables],
  vmath,
  common, replays

type
  PathNode = object
    pos: IVec2
    gCost: int
    hCost: int
    parent: int

proc fCost(node: PathNode): int =
  ## Calculate the total cost of a node.
  node.gCost + node.hCost

proc heuristic(a, b: IVec2): int =
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
  var openList: seq[PathNode] = @[PathNode(pos: start, gCost: 0, hCost: heuristic(start, goal), parent: -1)]
  var closedSet: seq[IVec2] = @[]
  var allNodes: seq[PathNode] = @[]
  while openList.len > 0:
    var currentIdx = 0
    for i in 1 ..< openList.len:
      if openList[i].fCost < openList[currentIdx].fCost:
        currentIdx = i
    let current = openList[currentIdx]
    openList.delete(currentIdx)
    closedSet.add(current.pos)
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
      var foundInOpen = false
      for i in 0 ..< openList.len:
        if openList[i].pos == neighborPos:
          foundInOpen = true
          if newGCost < openList[i].gCost:
            openList[i].gCost = newGCost
            openList[i].parent = currentNodeIdx
          break
      if not foundInOpen:
        openList.add(PathNode(
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

  for objIdx, obj in agentObjectives[agentId]:
    case obj.objectiveType
    of ObjMove:
      # For moving, path directly to the objective.
      let movePath = findPath(lastPos, obj.pos)
      if movePath.len == 0:
        clearPath(agentId)
        return
      # Convert positions to PathMove actions.
      for pos in movePath:
        if pos != lastPos:
          pathActions.add(PathAction(actionType: PathMove, pos: pos))
      lastPos = obj.pos
    of ObjBump:
      # For bumping, path to the specified approach position.
      let approachPos = ivec2(obj.pos.x + obj.approachDir.x, obj.pos.y + obj.approachDir.y)
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
            pathActions.add(PathAction(actionType: PathMove, pos: pos))
      # Add the bump action.
      pathActions.add(PathAction(
        actionType: PathBump,
        bumpPos: obj.pos,
        bumpDir: ivec2(obj.pos.x - approachPos.x, obj.pos.y - approachPos.y),
      ))
      lastPos = approachPos
    of ObjAction:
      # Add action as a path action to maintain synchronization.
      pathActions.add(PathAction(
        actionType: PathDoAction,
        actionId: obj.actionId,
        argument: obj.argument
      ))

  if pathActions.len > 0:
    agentPaths[agentId] = pathActions
  else:
    clearPath(agentId)

