import
  std/[random],
  benchy, vmath,
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

proc setupTestMap(testMap: TestMap) =
  ## Setup global replay state for benchmarking.
  replay = Replay(
    version: 2,
    numAgents: 0,
    maxSteps: 1,
    mapSize: (testMap.width, testMap.height),
    fileName: "bench",
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

timeIt "50x50 empty diagonal":
  let testMap = createEmptyMap(50, 50)
  setupTestMap(testMap)
  let path = findPath(ivec2(0, 0), ivec2(49, 49))
  doAssert path.len == 99

timeIt "50x50 with border":
  let testMap = createMapWithBorder(50, 50)
  setupTestMap(testMap)
  let path = findPath(ivec2(1, 1), ivec2(48, 48))
  doAssert path.len > 0

timeIt "50x50 corridor":
  let testMap = createMapWithCorridor(50, 50)
  setupTestMap(testMap)
  let path = findPath(ivec2(1, 25), ivec2(48, 25))
  doAssert path.len == 48

timeIt "32x32 random walls 20%":
  let testMap = createMapWithRandomWalls(32, 32, 0.2)
  setupTestMap(testMap)
  discard findPath(ivec2(1, 1), ivec2(30, 30))

timeIt "100x100 empty diagonal":
  let testMap = createEmptyMap(100, 100)
  setupTestMap(testMap)
  let path = findPath(ivec2(0, 0), ivec2(99, 99))
  doAssert path.len == 199

timeIt "100x100 with border":
  let testMap = createMapWithBorder(100, 100)
  setupTestMap(testMap)
  let path = findPath(ivec2(1, 1), ivec2(98, 98))
  doAssert path.len > 0

timeIt "100x100 corridor":
  let testMap = createMapWithCorridor(100, 100)
  setupTestMap(testMap)
  let path = findPath(ivec2(1, 50), ivec2(98, 50))
  doAssert path.len == 98

timeIt "64x64 random walls 20%":
  let testMap = createMapWithRandomWalls(64, 64, 0.2)
  setupTestMap(testMap)
  discard findPath(ivec2(1, 1), ivec2(62, 62))

