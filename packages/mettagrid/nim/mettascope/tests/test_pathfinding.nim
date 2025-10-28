import
  std/[unittest, strutils, times, random],
  vmath,
  ../src/mettascope/[pathfinding, replays, common]

type
  TestMap = object
    width: int
    height: int
    walkable: seq[bool]

proc createEmptyMap(width, height: int): TestMap =
  ## Create an empty map with all walkable tiles.
  result.width = width
  result.height = height
  result.walkable = newSeq[bool](width * height)
  for i in 0 ..< result.walkable.len:
    result.walkable[i] = true

proc createMapWithBorder(width, height: int): TestMap =
  ## Create a map with walls around the border.
  result = createEmptyMap(width, height)
  for y in 0 ..< height:
    for x in 0 ..< width:
      if x == 0 or x == width - 1 or y == 0 or y == height - 1:
        result.walkable[y * width + x] = false

proc createMapWithRandomWalls(width, height: int, wallDensity: float): TestMap =
  ## Create a map with randomly placed walls.
  result = createMapWithBorder(width, height)
  var rng = initRand(42)
  for y in 1 ..< height - 1:
    for x in 1 ..< width - 1:
      if rng.rand(1.0) < wallDensity:
        result.walkable[y * width + x] = false

proc createMapWithCorridor(width, height: int): TestMap =
  ## Create a map with a long winding corridor.
  result = createEmptyMap(width, height)
  for i in 0 ..< result.walkable.len:
    result.walkable[i] = false
  for y in 1 ..< height - 1:
    result.walkable[y * width + width div 2] = true
  for x in 1 ..< width - 1:
    result.walkable[(height div 2) * width + x] = true

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

suite "Large Scale Pathfinding":
  test "50x50 empty map diagonal path":
    echo "  Testing 50x50 empty map..."
    let testMap = createEmptyMap(50, 50)
    setupTestMap(testMap)
    let start = epochTime()
    let path = findPath(ivec2(0, 0), ivec2(49, 49))
    let elapsed = epochTime() - start
    echo "  Path found in ", elapsed.formatFloat(ffDecimal, 3), "s"
    check path.len == 99
    check path[0] == ivec2(0, 0)
    check path[^1] == ivec2(49, 49)

  test "50x50 with border path":
    echo "  Testing 50x50 with border..."
    let testMap = createMapWithBorder(50, 50)
    setupTestMap(testMap)
    let start = epochTime()
    let path = findPath(ivec2(1, 1), ivec2(48, 48))
    let elapsed = epochTime() - start
    echo "  Path found in ", elapsed.formatFloat(ffDecimal, 3), "s"
    check path.len > 0
    check path[0] == ivec2(1, 1)
    check path[^1] == ivec2(48, 48)

  test "32x32 with random walls":
    echo "  Testing 32x32 with random walls..."
    let testMap = createMapWithRandomWalls(32, 32, 0.2)
    setupTestMap(testMap)
    let start = epochTime()
    let path = findPath(ivec2(1, 1), ivec2(30, 30))
    let elapsed = epochTime() - start
    echo "  Path search completed in ", elapsed.formatFloat(ffDecimal, 3), "s"
    if path.len > 0:
      echo "  Path length: ", path.len
      check path[0] == ivec2(1, 1)
      check path[^1] == ivec2(30, 30)
    else:
      echo "  No path found (expected with random walls)"

  test "50x50 long corridor":
    echo "  Testing 50x50 long corridor..."
    let testMap = createMapWithCorridor(50, 50)
    setupTestMap(testMap)
    let start = epochTime()
    let path = findPath(ivec2(1, 25), ivec2(48, 25))
    let elapsed = epochTime() - start
    echo "  Path found in ", elapsed.formatFloat(ffDecimal, 3), "s"
    check path.len == 48
    check path[0] == ivec2(1, 25)
    check path[^1] == ivec2(48, 25)

  test "100x100 corner to corner":
    echo "  Testing 100x100 corner to corner..."
    let testMap = createMapWithBorder(100, 100)
    setupTestMap(testMap)
    let start = epochTime()
    let path = findPath(ivec2(1, 1), ivec2(98, 98))
    let elapsed = epochTime() - start
    echo "  Path found in ", elapsed.formatFloat(ffDecimal, 3), "s"
    echo "  Path length: ", path.len
    check path.len > 0
    check path[0] == ivec2(1, 1)
    check path[^1] == ivec2(98, 98)
