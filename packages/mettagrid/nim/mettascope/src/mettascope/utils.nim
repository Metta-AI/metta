import
  std/[algorithm],
  boxy, vmath, windy, fidget2/hybridrender

var
  typeface*: Typeface

proc drawText*(
  bxy: Boxy,
  imageKey: string,
  transform: Mat3,
  typeface: Typeface,
  text: string,
  size: float32,
  color: Color
) =
  ## Draw text on the screen.
  var font = newFont(typeface)
  font.size = size
  font.paint = color
  let
    arrangement = typeset(@[newSpan(text, font)], bounds = vec2(1280, 800))
    globalBounds = arrangement.computeBounds(transform).snapToPixels()
    textImage = newImage(globalBounds.w.int, globalBounds.h.int)
    imageSpace = translate(-globalBounds.xy) * transform
  textImage.fillText(arrangement, imageSpace)

  bxy.addImage(imageKey, textImage)
  bxy.drawImage(imageKey, globalBounds.xy)

proc measureText*(
  text: string,
  size: float32,
  typeface: Typeface
): Vec2 =
  var font = newFont(typeface)
  font.size = size
  let arrangement = typeset(@[newSpan(text, font)], bounds = vec2(1280, 800))
  let transform = translate(vec2(0, 0))
  let bounds = arrangement.computeBounds(transform).snapToPixels()
  return vec2(bounds.w, bounds.h)

proc newSeq2D*[T](width: int, height: int): seq[seq[T]] =
  result = newSeq[seq[T]](width)
  for i in 0 ..< width:
    result[i] = newSeq[T](height)

type
  PathNode* = object
    pos*: IVec2
    gCost*: int
    hCost*: int
    parent*: int

proc fCost(node: PathNode): int =
  ## Calculate the total cost of a node.
  node.gCost + node.hCost

proc heuristic(a, b: IVec2): int =
  ## Calculate the Manhattan distance heuristic.
  abs(a.x - b.x) + abs(a.y - b.y)

proc findPath*(start, goal: IVec2, mapWidth, mapHeight: int, isWalkable: proc(pos: IVec2): bool): seq[IVec2] =
  ## Find a path from start to goal using A* pathfinding.
  if start == goal:
    return @[]

  if not isWalkable(goal):
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

      if neighborPos.x < 0 or neighborPos.x >= mapWidth or neighborPos.y < 0 or neighborPos.y >= mapHeight:
        continue

      if neighborPos in closedSet:
        continue

      if not isWalkable(neighborPos):
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
