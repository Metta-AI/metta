import
  std/[unittest, strutils],
  vmath,
  ../src/mettascope/[pathfinding, replays, common]

type
  TestMap = object
    width: int
    height: int
    walkable: seq[bool]

proc parseAsciiMap(asciiMap: string): TestMap =
  ## Parse an ASCII map into a TestMap structure.
  let lines = asciiMap.strip().split('\n')
  result.height = lines.len
  result.width = if lines.len > 0: lines[0].len else: 0
  result.walkable = newSeq[bool](result.width * result.height)
  for y, line in lines:
    for x, c in line:
      result.walkable[y * result.width + x] = c != '#'

proc setupTestMap(testMap: TestMap) =
  ## Setup global replay state for testing.

  replay = Replay(
    version: 2,
    numAgents: 0,
    maxSteps: 1,
    mapSize: (testMap.width, testMap.height),
    fileName: "test",
    objects: @[],
  )

  for y in 0 ..< testMap.height:
    for x in 0 ..< testMap.width:
      if not testMap.walkable[y * testMap.width + x]:
        let obj = Entity(
          id: replay.objects.len,
          typeName: "wall",
          location: @[ivec3(x.int32, y.int32, 0)],
        )
        replay.objects.add(obj)

suite "Pathfinding Basic Tests":
  test "heuristic calculates Manhattan distance":
    let a = ivec2(0, 0)
    let b = ivec2(3, 4)
    check heuristic(a, b) == 7

  test "heuristic with same position":
    let a = ivec2(5, 5)
    check heuristic(a, a) == 0

  test "heuristic with negative coordinates":
    let a = ivec2(-2, -3)
    let b = ivec2(1, 2)
    check heuristic(a, b) == 8

  test "PathNode fCost calculation":
    let node = PathNode(
      pos: ivec2(0, 0),
      gCost: 5,
      hCost: 10,
      parent: -1
    )
    check node.fCost() == 15

suite "Pathfinding with Maps":
  test "simple straight line path":
    let testMap = parseAsciiMap("""
#####
#...#
#...#
#...#
#####""")
    setupTestMap(testMap)
    let path = findPath(ivec2(1, 1), ivec2(1, 3))
    check path.len == 3
    check path[0] == ivec2(1, 1)
    check path[1] == ivec2(1, 2)
    check path[2] == ivec2(1, 3)

  test "path around obstacle":
    let testMap = parseAsciiMap("""
#######
#.....#
#.###.#
#.....#
#######""")
    setupTestMap(testMap)
    let path = findPath(ivec2(1, 1), ivec2(5, 1))
    check path.len > 0
    check path[0] == ivec2(1, 1)
    check path[^1] == ivec2(5, 1)

  test "no path through walls":
    let testMap = parseAsciiMap("""
#######
#..#..#
#..#..#
#..#..#
#######""")
    setupTestMap(testMap)
    let path = findPath(ivec2(1, 1), ivec2(5, 1))
    check path.len == 0

  test "same start and goal returns empty path":
    let testMap = parseAsciiMap("""
#####
#...#
#...#
#####""")
    setupTestMap(testMap)
    let path = findPath(ivec2(2, 1), ivec2(2, 1))
    check path.len == 0

  test "path to unwalkable position returns empty":
    let testMap = parseAsciiMap("""
#####
#...#
#...#
#####""")
    setupTestMap(testMap)
    let path = findPath(ivec2(1, 1), ivec2(0, 0))
    check path.len == 0
