import
  std/[strutils, random],
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
          location: @[ivec2(x.int32, y.int32)],
        )
        replay.objects.add(obj)

block basic_tests:
  block heuristic_calculation:
    let pos1 = ivec2(0, 0)
    let pos2 = ivec2(3, 4)
    doAssert heuristic(pos1, pos2) == 7, "heuristic should calculate Manhattan distance"

    let same_pos = ivec2(5, 5)
    doAssert heuristic(same_pos, same_pos) == 0, "heuristic should be 0 for same position"

    let neg_pos1 = ivec2(-2, -3)
    let neg_pos2 = ivec2(1, 2)
    doAssert heuristic(neg_pos1, neg_pos2) == 8, "heuristic should work with negative coordinates"

  block pathnode_fcost:
    let node = PathNode(
      pos: ivec2(0, 0),
      gCost: 5,
      hCost: 10,
      parent: -1
    )
    doAssert node.fCost() == 15, "fCost should be gCost + hCost"

block map_tests:
  block straight_line_path:
    let map = parseAsciiMap("""
#####
#...#
#...#
#...#
#####""")
    setupTestMap(map)
    let path = findPath(ivec2(1, 1), ivec2(1, 3))
    doAssert path.len == 3, "straight line path should have 3 points"
    doAssert path[0] == ivec2(1, 1), "path should start at correct position"
    doAssert path[1] == ivec2(1, 2), "path should go through middle"
    doAssert path[2] == ivec2(1, 3), "path should end at goal"

  block path_around_obstacle:
    let map = parseAsciiMap("""
#######
#.....#
#.###.#
#.....#
#######""")
    setupTestMap(map)
    let path = findPath(ivec2(1, 1), ivec2(5, 1))
    doAssert path.len > 0, "path around obstacle should be found"
    doAssert path[0] == ivec2(1, 1), "path should start at correct position"
    doAssert path[^1] == ivec2(5, 1), "path should end at goal"

  block no_path_through_walls:
    let map = parseAsciiMap("""
#######
#..#..#
#..#..#
#..#..#
#######""")
    setupTestMap(map)
    let path = findPath(ivec2(1, 1), ivec2(5, 1))
    doAssert path.len == 0, "no path should be found through walls"

  block same_start_and_goal:
    let map = parseAsciiMap("""
#####
#...#
#...#
#####""")
    setupTestMap(map)
    let path = findPath(ivec2(2, 1), ivec2(2, 1))
    doAssert path.len == 0, "same start and goal should return empty path"

  block unwalkable_goal:
    let map = parseAsciiMap("""
#####
#...#
#...#
#####""")
    setupTestMap(map)
    let path = findPath(ivec2(1, 1), ivec2(0, 0))
    doAssert path.len == 0, "path to unwalkable position should be empty"

block large_scale_tests:
  block empty_50x50_diagonal:
    let map = createEmptyMap(50, 50)
    setupTestMap(map)
    let path = findPath(ivec2(0, 0), ivec2(49, 49))
    doAssert path.len == 99, "diagonal path should have 99 steps"
    doAssert path[0] == ivec2(0, 0), "path should start at origin"
    doAssert path[^1] == ivec2(49, 49), "path should end at goal"

  block border_50x50_path:
    let map = createMapWithBorder(50, 50)
    setupTestMap(map)
    let path = findPath(ivec2(1, 1), ivec2(48, 48))
    doAssert path.len > 0, "path with border should be found"
    doAssert path[0] == ivec2(1, 1), "path should start correctly"
    doAssert path[^1] == ivec2(48, 48), "path should end at goal"

  block random_walls_32x32:
    let map = createMapWithRandomWalls(32, 32, 0.2)
    setupTestMap(map)
    let path = findPath(ivec2(1, 1), ivec2(30, 30))
    doAssert path.len > 0, "path should be found in random walls"
    doAssert path[0] == ivec2(1, 1), "path should start correctly"
    doAssert path[^1] == ivec2(30, 30), "path should end at goal"

  block corridor_50x50:
    let map = createMapWithCorridor(50, 50)
    setupTestMap(map)
    let path = findPath(ivec2(1, 25), ivec2(48, 25))
    doAssert path.len == 48, "corridor path should have 48 steps"
    doAssert path[0] == ivec2(1, 25), "path should start in corridor"
    doAssert path[^1] == ivec2(48, 25), "path should end at goal"

  block border_100x100:
    let map = createMapWithBorder(100, 100)
    setupTestMap(map)
    let path = findPath(ivec2(1, 1), ivec2(98, 98))
    doAssert path.len > 0, "large map path should be found"
    doAssert path[0] == ivec2(1, 1), "path should start correctly"
    doAssert path[^1] == ivec2(98, 98), "path should end at goal"

when isMainModule:
  discard
