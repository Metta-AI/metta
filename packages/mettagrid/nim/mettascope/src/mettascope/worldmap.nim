import
  std/[math, os, strutils, tables, strformat, random, times],
  boxy, vmath, windy, fidget2/[hybridrender, common, measure],
  common, panels, actions, utils, replays, objectinfo,
  pathfinding, tilemap, pixelator, shaderquad

const TS = 1.0 / 64.0 # Tile scale.
const TILE_SIZE = 64

proc centerAt*(panel: Panel, entity: Entity)

var
  terrainMap*: TileMap
  visibilityMapStep*: int = -1
  visibilityMapSelectionId*: int = -1
  visibilityMapLockFocus*: bool = false
  visibilityMap*: TileMap
  px*: Pixelator
  sq*: ShaderQuad
  previousPanelSize*: Vec2 = vec2(0, 0)

proc weightedRandomInt*(weights: seq[int]): int =
  ## Return a random integer between 0 and 7, with a weighted distribution.
  var r = rand(sum(weights))
  var acc = 0
  for i, w in weights:
    acc += w
    if r <= acc:
      return i
  doAssert false, "should not happen"

const patternToTile = @[
  18, 17, 4, 4, 12, 22, 4, 4, 30, 13, 41, 41, 30, 13, 41, 41, 19, 23, 5, 5, 37,
  9, 5, 5, 30, 13, 41, 41, 30, 13, 41, 41, 24, 43, 39, 39, 44, 45, 39, 39, 48,
  32, 46, 46, 48, 32, 46, 46, 24, 43, 39, 39, 44, 45, 39, 39, 48, 32, 46, 46,
  48, 32, 46, 46, 36, 10, 3, 3, 16, 40, 3, 3, 20, 27, 6, 6, 20, 27, 6, 6, 25,
  15, 2, 2, 26, 38, 2, 2, 20, 27, 6, 6, 20, 27, 6, 6, 24, 43, 39, 39, 44, 45,
  39, 39, 48, 32, 46, 46, 48, 32, 46, 46, 24, 43, 39, 39, 44, 45, 39, 39, 48,
  32, 46, 46, 48, 32, 46, 46, 28, 28, 8, 8, 21, 21, 8, 8, 33, 33, 7, 7, 33, 33,
  7, 7, 35, 35, 31, 31, 14, 14, 31, 31, 33, 33, 7, 7, 33, 33, 7, 7, 47, 47, 1,
  1, 42, 42, 1, 1, 34, 34, 0, 0, 34, 34, 0, 0, 47, 47, 1, 1, 42, 42, 1, 1,
  34, 34, 0, 0, 34, 34, 0, 0, 28, 28, 8, 8, 21, 21, 8, 8, 33, 33, 7, 7, 33,
  33, 7, 7, 35, 35, 31, 31, 14, 14, 31, 31, 33, 33, 7, 7, 33, 33, 7, 7, 47, 47,
  1, 1, 42, 42, 1, 1, 34, 34, 0, 0, 34, 34, 0, 0, 47, 47, 1, 1, 42, 42, 1,
  1, 34, 34, 0, 0, 34, 34, 0, 0
]

proc generateTerrainMap(): TileMap {.measure.} =
  ## Generate a 1024x1024 texture where each pixel is a byte index into the 16x16 tile map.
  let
    width = ceil(replay.mapSize[0].float32 / 32.0f).int * 32
    height = ceil(replay.mapSize[1].float32 / 32.0f).int * 32

  echo "Real map size: ", replay.mapSize[0], "x", replay.mapSize[1]
  echo "Tile map size: ", width, "x", height, " (multiples of 32)"

  var terrainMap = newTileMap(
    width = width,
    height = height,
    tileSize = 64,
    atlasPath = dataDir & "/blob7x8.png"
  )

  var asteroidMap: seq[bool] = newSeq[bool](width * height)
  # Fill the asteroid map with ground (true).
  for y in 0 ..< replay.mapSize[1]:
    for x in 0 ..< replay.mapSize[0]:
      asteroidMap[y * width + x] = true

  # Walk the walls and generate a map of which tiles are present.
  for obj in replay.objects:
    if obj.typeName == "wall":
      let pos = obj.location.at(0)
      asteroidMap[pos.y * width + pos.x] = false

  # Generate the tile edges.
  for i in 0 ..< terrainMap.indexData.len:
    let x = i mod width
    let y = i div width

    proc get(map: seq[bool], x: int, y: int): int =
      if x < 0 or y < 0 or x >= width or y >= height:
        return 0
      if map[y * width + x]:
        return 1
      return 0

    var tile: uint8 = 0
    if asteroidMap[y * width + x]:
      tile = (49 + weightedRandomInt(@[100, 50, 25, 10, 5, 2, 1])).uint8
    else:
      let
        pattern = (
          1 * asteroidMap.get(x-1, y-1) + # NW
          2 * asteroidMap.get(x, y-1) + # N
          4 * asteroidMap.get(x+1, y-1) + # NE
          8 * asteroidMap.get(x+1, y) + # E
          16 * asteroidMap.get(x+1, y+1) + # SE
          32 * asteroidMap.get(x, y+1) + # S
          64 * asteroidMap.get(x-1, y+1) + # SW
          128 * asteroidMap.get(x-1, y) # W
        )
      tile = patternToTile[pattern].uint8
    terrainMap.indexData[i] = tile

  terrainMap.setupGPU()
  return terrainMap

proc rebuildVisibilityMap*(visibilityMap: TileMap) {.measure.} =
  ## Rebuild the visibility map.
  let
    width = visibilityMap.width
    height = visibilityMap.height

  var fogOfWarMap: seq[bool] = newSeq[bool](width * height)
  for y in 0 ..< replay.mapSize[1]:
    for x in 0 ..< replay.mapSize[0]:
      fogOfWarMap[y * width + x] = true

  # Walk the agents and clear the visibility map.
  # If lockFocus is on with an agent selected, only show that agent's vision.
  let agentsToProcess = if settings.lockFocus and selection != nil and selection.isAgent:
    @[selection]
  else:
    replay.agents

  for obj in agentsToProcess:
    let center = ivec2(int32(obj.visionSize div 2), int32(obj.visionSize div 2))
    let pos = obj.location.at
    for i in 0 ..< obj.visionSize:
      for j in 0 ..< obj.visionSize:
        let gridPos = pos.xy + ivec2(int32(i), int32(j)) - center
        if gridPos.x >= 0 and gridPos.x < width and
          gridPos.y >= 0 and gridPos.y < height:
          fogOfWarMap[gridPos.y * width + gridPos.x] = false

  # Generate the tile edges.
  for i in 0 ..< visibilityMap.indexData.len:
    let x = i mod width
    let y = i div width

    proc get(map: seq[bool], x: int, y: int): int =
      if x < 0 or y < 0 or x >= width or y >= height:
        return 0
      if map[y * width + x]:
        return 1
      return 0

    var tile: uint8 = 0
    if fogOfWarMap[y * width + x]:
      tile = 49
    else:
      let
        pattern = (
          1 * fogOfWarMap.get(x-1, y-1) + # NW
          2 * fogOfWarMap.get(x, y-1) + # N
          4 * fogOfWarMap.get(x+1, y-1) + # NE
          8 * fogOfWarMap.get(x+1, y) + # E
          16 * fogOfWarMap.get(x+1, y+1) + # SE
          32 * fogOfWarMap.get(x, y+1) + # S
          64 * fogOfWarMap.get(x-1, y+1) + # SW
          128 * fogOfWarMap.get(x-1, y) # W
        )
      tile = patternToTile[pattern].uint8
    visibilityMap.indexData[i] = tile

proc generateVisibilityMap(): TileMap {.measure.} =
  ## Generate a 1024x1024 texture where each pixel is a byte index into the 16x16 tile map.
  let
    width = ceil(replay.mapSize[0].float32 / 32.0f).int * 32
    height = ceil(replay.mapSize[1].float32 / 32.0f).int * 32

  echo "Real map size: ", replay.mapSize[0], "x", replay.mapSize[1]
  echo "Tile map size: ", width, "x", height, " (multiples of 32)"

  var visibilityMap = newTileMap(
    width = width,
    height = height,
    tileSize = 64,
    atlasPath = dataDir & "/fog7x8.png"
  )
  visibilityMap.rebuildVisibilityMap()
  visibilityMap.setupGPU()
  return visibilityMap

proc updateVisibilityMap*(visibilityMap: TileMap) {.measure.} =
  ## Update the visibility map.
  visibilityMap.rebuildVisibilityMap()
  visibilityMap.updateGPU()

proc buildAtlas*() {.measure.} =
  ## Build the atlas.
  bxy.addImage("minimapPip", readImage(dataDir & "/minimapPip.png"))
  bxy.addImage("selection", readImage(dataDir & "/selection.png"))
  bxy.addImage("agents/path", readImage(dataDir & "/agents/path.png"))
  bxy.addImage("agents/footprints", readImage(dataDir & "/agents/footprints.png"))
  bxy.addImage("actions/thoughts_lightning", readImage(dataDir & "/actions/thoughts_lightning.png"))
  bxy.addImage("actions/icons/unknown", readImage(dataDir & "/actions/icons/unknown.png"))
  bxy.addImage("actions/arrow", readImage(dataDir & "/actions/arrow.png"))
  bxy.addImage("actions/thoughts", readImage(dataDir & "/actions/thoughts.png"))

  bxy.addImage("minimap/agent", readImage(dataDir & "/minimap/agent.png"))
  bxy.addImage("minimap/assembler", readImage(dataDir & "/minimap/assembler.png"))
  bxy.addImage("minimap/carbon_extractor", readImage(dataDir & "/minimap/carbon_extractor.png"))
  bxy.addImage("minimap/charger", readImage(dataDir & "/minimap/charger.png"))
  bxy.addImage("minimap/germanium_extractor", readImage(dataDir & "/minimap/germanium_extractor.png"))
  bxy.addImage("minimap/silicon_extractor", readImage(dataDir & "/minimap/silicon_extractor.png"))
  bxy.addImage("minimap/oxygen_extractor", readImage(dataDir & "/minimap/oxygen_extractor.png"))
  bxy.addImage("minimap/chest", readImage(dataDir & "/minimap/chest.png"))

  proc addDir(rootDir: string, dir: string) =
    for path in walkDirRec(rootDir / dir):
      if path.endsWith(".png") and "fidget" notin path:
        let name = path.replace(rootDir & "/", "").replace(".png", "")
        bxy.addImage(name, readImage(path))

  addDir(dataDir, "resources")

proc getProjectionView*(): Mat4 {.measure.} =
  ## Get the projection and view matrix.
  let m = bxy.getTransform()
  let view = mat4(
    m[0, 0], m[0, 1], m[0, 2], 0,
    m[1, 0], m[1, 1], m[1, 2], 0,
    0, 0, 0, 1,
    m[2, 0], m[2, 1], m[2, 2], 1
  )
  let projection = ortho(0.0f, window.size.x.float32, window.size.y.float32, 0.0f, -1.0f, 1.0f)
  projection * view

proc useSelections*(panel: Panel) {.measure.} =
  ## Reads the mouse position and selects the thing under it.
  let modifierDown = when defined(macosx):
    window.buttonDown[KeyLeftSuper] or window.buttonDown[KeyRightSuper]
  else:
    window.buttonDown[KeyLeftControl] or window.buttonDown[KeyRightControl]

  let shiftDown = window.buttonDown[KeyLeftShift] or window.buttonDown[KeyRightShift]
  let rDown = window.buttonDown[KeyR]

  # Track mouse down position to distinguish clicks from drags.
  if window.buttonPressed[MouseLeft] and not modifierDown:
    mouseDownPos = window.mousePos.vec2

  # Focus agent on double-click.
  if window.buttonPressed[DoubleClick] and not modifierDown:
    settings.lockFocus = not settings.lockFocus
    if settings.lockFocus and selection != nil:
      centerAt(panel, selection)

  # Only select on mouse up, and only if we didn't drag much.
  if window.buttonReleased[MouseLeft] and not modifierDown:
    let mouseDragDistance = (window.mousePos.vec2 - mouseDownPos).length
    const maxClickDragDistance = 5.0
    if mouseDragDistance < maxClickDragDistance:
      selection = nil
      let
        mousePos = bxy.getTransform().inverse * window.mousePos.vec2
        gridPos = (mousePos + vec2(0.5, 0.5)).ivec2
      if gridPos.x >= 0 and gridPos.x < replay.mapSize[0] and
        gridPos.y >= 0 and gridPos.y < replay.mapSize[1]:
        let obj = getObjectAtLocation(gridPos)
        if obj != nil:
          selectObject(obj)

  if window.buttonPressed[MouseRight] or (window.buttonPressed[MouseLeft] and modifierDown):
    if selection != nil and selection.isAgent:
      let
        mousePos = bxy.getTransform().inverse * window.mousePos.vec2
        gridPos = (mousePos + vec2(0.5, 0.5)).ivec2
      if gridPos.x >= 0 and gridPos.x < replay.mapSize[0] and
        gridPos.y >= 0 and gridPos.y < replay.mapSize[1]:
        let startPos = selection.location.at(step).xy

        # Determine if this is a Bump or Move objective.
        let targetObj = getObjectAtLocation(gridPos)
        var objective: Objective
        if targetObj != nil:
          let typeName = targetObj.typeName
          if typeName != "agent" and typeName != "wall":
            # Calculate which quadrant of the tile was clicked.
            # The tile center is at gridPos, and mousePos has fractional parts.
            let
              tileCenterX = gridPos.x.float32
              tileCenterY = gridPos.y.float32
              offsetX = mousePos.x - tileCenterX
              offsetY = mousePos.y - tileCenterY
            # Divide the tile into 4 quadrants at 45-degree angles (diamond shape).
            # If the click is more horizontal than vertical, use left/right approach.
            # If the click is more vertical than horizontal, use top/bottom approach.
            var approachDir: IVec2
            if abs(offsetX) > abs(offsetY):
              # Left or right quadrant.
              if offsetX > 0:
                approachDir = ivec2(1, 0)   # Clicked right, approach from right.
              else:
                approachDir = ivec2(-1, 0)  # Clicked left, approach from left.
            else:
              # Top or bottom quadrant.
              if offsetY > 0:
                approachDir = ivec2(0, 1)   # Clicked bottom, approach from bottom.
              else:
                approachDir = ivec2(0, -1)  # Clicked top, approach from top.
            objective = Objective(kind: Bump, pos: gridPos, approachDir: approachDir, repeat: rDown)
          else:
            objective = Objective(kind: Move, pos: gridPos, approachDir: ivec2(0, 0), repeat: rDown)
        else:
          objective = Objective(kind: Move, pos: gridPos, approachDir: ivec2(0, 0), repeat: rDown)

        if shiftDown:
          # Queue up additional objectives.
          if not agentObjectives.hasKey(selection.agentId) or agentObjectives[selection.agentId].len == 0:
            # No existing objectives, start fresh.
            agentObjectives[selection.agentId] = @[objective]
            recomputePath(selection.agentId, startPos)
          else:
            # Append to existing objectives.
            agentObjectives[selection.agentId].add(objective)
            # Recompute path to include all objectives.
            recomputePath(selection.agentId, startPos)
        else:
          # Replace the entire objective queue.
          agentObjectives[selection.agentId] = @[objective]
          recomputePath(selection.agentId, startPos)

proc drawObjects*() {.measure.} =
  ## Draw the objects on the map.
  for thing in replay.objects:
    let typeName = thing.typeName
    let pos = thing.location.at().xy
    case typeName
    of "wall":
      discard
      # bxy.drawImage("objects/wall",  pos.vec2, angle = 0, scale = TS)
    of "agent":
      let agent = thing
      var agentImage = case agent.orientation.at:
        of 0: "agents/agent.n"
        of 1: "agents/agent.s"
        of 2: "agents/agent.w"
        of 3: "agents/agent.e"
        else:
          echo "Unknown orientation: ", agent.orientation.at
          "agents/agent.n"
      px.drawSprite(
        agentImage,
        pos * TILE_SIZE
      )
    else:
      let spriteName =
        if "objects/" & thing.typeName in px:
          "objects/" & thing.typeName
        else:
          "objects/unknown"
      if thing.isClipped.at:
        px.drawSprite(
          spriteName & ".clipped",
          pos * TILE_SIZE
        )
      else:
        px.drawSprite(
          spriteName,
          pos * TILE_SIZE,
        )

proc drawVisualRanges*(alpha = 0.5) {.measure.} =
  ## Draw the visual ranges of the selected agent.

  bxy.enterRawOpenGLMode()

  if visibilityMap == nil:
    visibilityMapStep = step
    visibilityMapSelectionId = if selection != nil: selection.id else: -1
    visibilityMapLockFocus = settings.lockFocus
    visibilityMap = generateVisibilityMap()

  let
    currentSelectionId = if selection != nil: selection.id else: -1
    needsRebuild = visibilityMapStep != step or visibilityMapLockFocus != settings.lockFocus or
      (settings.lockFocus and visibilityMapSelectionId != currentSelectionId)

  if needsRebuild:
    visibilityMapStep = step
    visibilityMapSelectionId = currentSelectionId
    visibilityMapLockFocus = settings.lockFocus
    visibilityMap.updateVisibilityMap()

  visibilityMap.draw(
    getProjectionView(),
    zoom = 2.0f,
    zoomThreshold = 1.5f,
    tint = color(0, 0, 0, alpha)
  )

  bxy.exitRawOpenGLMode()


proc drawFogOfWar*() {.measure.} =
  ## Draw the fog of war.
  drawVisualRanges(alpha = 1.0)

proc drawTrajectory*() {.measure.} =
  ## Draw the trajectory of the selected object, with footprints or a future arrow.
  if selection != nil and selection.location.len > 1:
    for i in 1 ..< replay.maxSteps:
      let
        loc0 = selection.location.at(i - 1)
        loc1 = selection.location.at(i)
        cx0 = loc0.x.int
        cy0 = loc0.y.int
        cx1 = loc1.x.int
        cy1 = loc1.y.int

      if cx0 != cx1 or cy0 != cy1:
        let a = 1.0f - abs(i - step).float32 / 200.0f
        if a > 0:
          var
            tint = color(0, 0, 0, a)
            image = ""

          let isAgent = selection.typeName == "agent"
          if step >= i:
            # Past trajectory is black.
            tint = color(0, 0, 0, a)
            if isAgent:
              image = "agents/footprints"
            else:
              image = "agents/past_arrow"
          else:
            # Future trajectory is white.
            tint = color(1, 1, 1, a)
            if isAgent:
              image = "agents/path"
            else:
              image = "agents/future_arrow"

          let
            dx = cx1 - cx0
            dy = cy1 - cy0
          var
            rotation: float32 = 0
            diagScale: float32 = 1

          if dx > 0 and dy == 0:
            rotation = 0
          elif dx < 0 and dy == 0:
            rotation = Pi
          elif dx == 0 and dy > 0:
            rotation = -Pi / 2
          elif dx == 0 and dy < 0:
            rotation = Pi / 2
          elif dx > 0 and dy > 0:
            rotation = -Pi / 4
            diagScale = sqrt(2.0f)
          elif dx > 0 and dy < 0:
            rotation = Pi / 4
            diagScale = sqrt(2.0f)
          elif dx < 0 and dy > 0:
            rotation = -3 * Pi / 4
            diagScale = sqrt(2.0f)
          elif dx < 0 and dy < 0:
            rotation = 3 * Pi / 4
            diagScale = sqrt(2.0f)

          # Draw centered at the tile with rotation. Use a slightly larger scale on diagonals.
          bxy.drawImage(
            image,
            vec2(cx0.float32, cy0.float32),
            angle = rotation,
            scale = (1.0f / 200.0f) * diagScale,
            tint = tint
          )

proc drawActions*() {.measure.} =
  ## Draw the actions of the selected agent.
  for obj in replay.objects:
    # Do agent actions.
    if obj.isAgent:
      let actionId = obj.actionId.at
      if (replay.drawnAgentActionMask and (1'u64 shl actionId)) != 0 and
          obj.actionSuccess.at and
          actionId >= 0 and actionId < replay.actionImages.len:
        bxy.drawImage(
          if actionId != replay.attackActionId:
            replay.actionImages[actionId]
          else:
            let attackParam = obj.actionParameter.at
            if attackParam >= 1 and attackParam <= 9:
              replay.actionAttackImages[attackParam - 1]
            else:
              continue,
          obj.location.at.xy.vec2,
          angle = case obj.orientation.at:
          of 0: PI / 2 # North
          of 1: -PI / 2 # South
          of 2: PI # West
          of 3: 0 # East
          else: 0, # East
        scale = 1/200)
    elif obj.productionProgress.at > 0:
      bxy.drawImage(
        "actions/converting",
        obj.location.at.xy.vec2,
        angle = 0,
        scale = 1/200
      )

proc drawAgentDecorations*() {.measure.} =
  # Draw energy bars, shield and frozen status.
  for agent in replay.agents:
    if agent.isFrozen.at:
      bxy.drawImage(
        "agents/frozen",
        agent.location.at.xy.vec2,
        angle = 0,
        scale = TS
      )

proc drawGrid*() {.measure.} =
  # Draw the grid using a single quad and shader-based lines.
  bxy.enterRawOpenGLMode()
  if sq == nil:
    sq = newGridQuad(dataDir & "/view/grid10.png", 10, 10)
  let
    mvp = getProjectionView()
    mapSize = vec2(replay.mapSize[0].float32, replay.mapSize[1].float32)
    tileSize = vec2(1.0f, 1.0f) # world units per tile
    gridColor = vec4(1.0f, 1.0f, 1.0f, 1.0f) # subtle white grid
  sq.draw(mvp, mapSize, tileSize, gridColor, 1.0f)
  bxy.exitRawOpenGLMode()

proc drawPlannedPath*() {.measure.} =
  ## Draw the planned paths for all agents.
  ## Only show paths when in realtime mode and viewing the latest step.
  if playMode != Realtime or step != replay.maxSteps - 1:
    return
  for agentId, pathActions in agentPaths:
    if pathActions.len == 0:
      continue

    # Get agent's current position.
    let agent = getAgentById(agentId)
    var currentPos = agent.location.at(step).xy

    for action in pathActions:
      if action.kind != Move:
        continue
      # Draw arrow from current position to target position.
      let
        pos0 = currentPos
        pos1 = action.pos
        dx = pos1.x - pos0.x
        dy = pos1.y - pos0.y

      var rotation: float32 = 0
      if dx > 0 and dy == 0:
        rotation = 0
      elif dx < 0 and dy == 0:
        rotation = Pi
      elif dx == 0 and dy > 0:
        rotation = -Pi / 2
      elif dx == 0 and dy < 0:
        rotation = Pi / 2

      let alpha = 0.6
      bxy.drawImage(
        "agents/path",
        pos0.vec2,
        angle = rotation,
        scale = 1/200,
        tint = color(1, 1, 1, alpha)
      )
      currentPos = action.pos

    # Draw final queued objective.
    if agentObjectives.hasKey(agentId):
      let objectives = agentObjectives[agentId]
      if objectives.len > 0:
        let objective = objectives[^1]
        if objective.kind in {Move, Bump}:
          bxy.drawImage(
            "selection",
            objective.pos.vec2,
            angle = 0,
            scale = 1.0 / 200.0,
            tint = color(1, 1, 1, 0.5)
          )

      # Draw approach arrows for bump objectives.
      for objective in objectives:
        if objective.kind == Bump and (objective.approachDir.x != 0 or objective.approachDir.y != 0):
          let approachPos = ivec2(objective.pos.x + objective.approachDir.x, objective.pos.y + objective.approachDir.y)
          let offset = vec2(-objective.approachDir.x.float32 * 0.35, -objective.approachDir.y.float32 * 0.35)
          var rotation: float32 = 0
          if objective.approachDir.x > 0:
            rotation = Pi / 2
          elif objective.approachDir.x < 0:
            rotation = -Pi / 2
          elif objective.approachDir.y > 0:
            rotation = 0
          elif objective.approachDir.y < 0:
            rotation = Pi
          bxy.drawImage(
            "actions/arrow",
            approachPos.vec2 + offset,
            angle = rotation,
            scale = 1/200,
            tint = color(1, 1, 1, 0.7)
          )

proc drawSelection*() {.measure.} =
  # Draw selection.
  if selection != nil:
    bxy.drawImage(
      "selection",
      selection.location.at.xy.vec2,
      angle = 0,
      scale = 1/200
    )

proc applyOrientationOffset*(x: int, y: int, orientation: int): (int, int) =
  case orientation
  of 0:
    return (x, y - 1)
  of 1:
    return (x, y + 1)
  of 2:
    return (x - 1, y)
  of 3:
    return (x + 1, y)
  else:
    return (x, y)

proc drawThoughtBubbles*() {.measure.} =
  # Draw the thought bubbles of the selected agent.
  # The idea behind thought bubbles is to show what an agent is thinking.
  # We don't have this directly from the policy yet, so the next best thing
  # is to show a future "key action."
  # It should be a good proxy for what the agent is thinking about.
  if selection == nil or not selection.isAgent:
    return

  var keyAction = -1
  var keyParam = -1
  var keyStep = -1
  var actionHasTarget = false
  let actionStepEnd = min(step + 20, replay.maxSteps)
  for actionStep in step ..< actionStepEnd:
    # We need to find a key action in the future.
    # A key action is a successful action that is not a no-op, rotate, or move.
    # It must not be more than 20 steps in the future.
    let actionId = selection.actionId.at(actionStep)
    if actionId == -1:
      continue
    let actionParam = selection.actionParameter.at(actionStep)
    if actionParam == -1:
      continue
    let actionSuccess = selection.actionSuccess.at(actionStep)
    if not actionSuccess:
      continue
    let actionName = replay.actionNames[actionId]
    if actionName == "noop" or
    actionName == "rotate" or
    actionName == "move" or
    actionName == "move_cardinal" or
    actionName == "move_8way":
      continue
    keyAction = actionId
    keyParam = actionParam
    keyStep = actionStep
    actionHasTarget = not (actionName == "attack" or actionName == "attack_nearest")
    break

  if keyAction != -1 and keyParam != -1:
    let loc = selection.location.at(step).xy.vec2
    if actionHasTarget and keyStep != step:
      # Draw an arrow on a circle around the target, pointing at it.
      let targetLoc = selection.location.at(keyStep).xy
      let (targetX, targetY) = applyOrientationOffset(targetLoc.x, targetLoc.y,
          selection.orientation.at(keyStep))
      let angle = arctan2(targetX.float32 - loc.x, targetY.float32 - loc.y)
      let r = 1.0f / 3.0f
      let tX = targetX.float32 - sin(angle) * r
      let tY = targetY.float32 - cos(angle) * r
      bxy.drawImage(
        "actions/arrow",
        vec2(tX, tY),
        angle = angle + PI,
        scale = 1/200
      )
    let pos = loc.vec2 + vec2(0.5, -0.5)
    # We have a key action, so draw the thought bubble.
    # Draw the key action icon with gained or lost resources.
    bxy.drawImage(
      if step == keyStep: "actions/thoughts_lightning" else: "actions/thoughts",
      pos,
      angle = 0,
      scale = 1/200
    )
    # Draw the action icon.
    bxy.drawImage(
      if keyAction < replay.actionIconImages.len:
        replay.actionIconImages[keyAction] else: "actions/icons/unknown",
      pos,
      angle = 0,
      scale = 1/200/4
    )

    # Draw the resources lost on the left and gained on the right.
    var gainX = pos.x + 32/200
    var lossX = pos.x - 32/200
    let gainMap = selection.gainMap.at(keyStep)
    for item in gainMap:
      var drawX = 0.0f
      if item.count > 0:
        drawX = gainX
        gainX += 8/200
      else:
        drawX = lossX
        lossX -= 8/200
      bxy.drawImage(
        replay.itemImages[item.itemId],
        vec2(drawX, pos.y),
        angle = 0,
        scale = 1/200/8
      )

proc drawTerrain*() {.measure.} =
  ## Draw the terrain, space and asteroid tiles using the terrainMap tilemap.
  bxy.enterRawOpenGLMode()

  if terrainMap == nil:
    terrainMap = generateTerrainMap()
    px = newPixelator(
      dataDir & "/atlas.png",
      dataDir & "/atlas.json"
    )

  terrainMap.draw(getProjectionView(), 2.0f, 1.5f)

  bxy.exitRawOpenGLMode()

proc drawWorldMini*() {.measure.} =
  const wallTypeName = "wall"
  const agentTypeName = "agent"

  drawTerrain()

  # Overlays
  if settings.showVisualRange:
    drawVisualRanges()
  elif settings.showFogOfWar:
    drawFogOfWar()

  # Agents
  let scale = 3.0
  bxy.saveTransform()
  bxy.scale(vec2(scale, scale))

  for obj in replay.objects:
    let pipName = "minimap/" & obj.typeName
    if pipName in bxy:
      let loc = obj.location.at(step).xy
      let rect = rect(
        (loc.x.float32) / scale - 0.5,
        (loc.y.float32) / scale - 0.5,
        1,
        1
      )
      bxy.drawImage(
        pipName,
        rect,
        color(1, 1, 1, 1)
      )

  bxy.restoreTransform()



proc centerAt*(panel: Panel, entity: Entity) {.measure.} =
  ## Center the map on the given entity.
  if entity.isNil:
    return
  let location = entity.location.at(step).xy
  let rectW = panel.rect.w.float32
  let rectH = panel.rect.h.float32
  if rectW <= 0 or rectH <= 0:
    return
  let z = panel.zoom * panel.zoom
  panel.pos.x = rectW / 2.0f - location.x.float32 * z
  panel.pos.y = rectH / 2.0f - location.y.float32 * z

proc drawWorldMain*() {.measure.} =
  ## Draw the world map.
  drawTerrain()
  drawTrajectory()
  drawObjects()

  measurePush("px.flush")
  bxy.enterRawOpenGLMode()
  px.flush(getProjectionView() * scale(vec3(TS, TS, 1.0f)))
  bxy.exitRawOpenGLMode()
  measurePop()

  #drawActions()
  drawAgentDecorations()
  drawSelection()
  drawPlannedPath()

  if settings.showVisualRange:
    drawVisualRanges()
  if settings.showFogOfWar:
    drawFogOfWar()
  if settings.showGrid:
    drawGrid()

  drawThoughtBubbles()

proc fitFullMap*(panel: Panel) {.measure.} =
  ## Set zoom and pan so the full map fits in the panel.
  if replay.isNil:
    return
  let rectW = panel.rect.w.float32
  let rectH = panel.rect.h.float32
  if rectW <= 0 or rectH <= 0:
    return
  let
    mapMinX = -0.5f
    mapMinY = -0.5f
    mapMaxX = replay.mapSize[0].float32 - 0.5f
    mapMaxY = replay.mapSize[1].float32 - 0.5f
    mapW = max(0.001f, mapMaxX - mapMinX)
    mapH = max(0.001f, mapMaxY - mapMinY)
  let zoomScale = min(rectW / mapW, rectH / mapH)
  panel.zoom = clamp(sqrt(zoomScale), panel.minZoom, panel.maxZoom)
  let
    cx = (mapMinX + mapMaxX) / 2.0f
    cy = (mapMinY + mapMaxY) / 2.0f
    z = panel.zoom * panel.zoom
  panel.pos.x = rectW / 2.0f - cx * z
  panel.pos.y = rectH / 2.0f - cy * z

proc fitVisibleMap*(panel: Panel) {.measure.} =
  ## Set zoom and pan so the visible area (union of all agent vision ranges) fits in the panel.
  if replay.isNil:
    return
  
  if replay.agents.len == 0:
    fitFullMap(panel)
    return

  let rectSize = vec2(panel.rect.w.float32, panel.rect.h.float32)

  # Calculate the union of all agent vision areas.
  var
    minPos = vec2(float32.high, float32.high)
    maxPos = vec2(float32.low, float32.low)

  for agent in replay.agents:
    if agent.location.len == 0:
      continue
    let
      pos = agent.location.at(step).xy.vec2
      visionRadius = agent.visionSize.float32 / 2.0f
      agentMin = pos - vec2(visionRadius, visionRadius)
      agentMax = pos + vec2(visionRadius, visionRadius)

    minPos = min(minPos, agentMin)
    maxPos = max(maxPos, agentMax)

  # Ensure we have valid bounds with reasonable size, otherwise fall back to full map
  let size = maxPos - minPos
  if size.x < 1.0f or size.y < 1.0f:
    fitFullMap(panel)
    return

  let
    visibleSize = maxPos - minPos
    zoomScale = min(rectSize.x / visibleSize.x, rectSize.y / visibleSize.y)
    center = (minPos + maxPos) / 2.0f
    zoom = clamp(sqrt(zoomScale), panel.minZoom, panel.maxZoom)

  panel.zoom = zoom
  panel.pos = rectSize / 2.0f - center * (zoom * zoom)

proc adjustPanelForResize*(panel: Panel) {.measure.} =
  ## Adjust pan and zoom when panel resizes to show the same portion of the map.
  let currentSize = vec2(panel.rect.w.float32, panel.rect.h.float32)

  # Skip if this is the first time or no change
  if previousPanelSize.x <= 0 or previousPanelSize.y <= 0 or currentSize == previousPanelSize:
    previousPanelSize = currentSize
    return

  # Calculate current center point in world coordinates using previous panel size
  let
    oldRectW = previousPanelSize.x
    oldRectH = previousPanelSize.y
    rectW = panel.rect.w.float32
    rectH = panel.rect.h.float32
    z = panel.zoom * panel.zoom
    centerX = (oldRectW / 2.0f - panel.pos.x) / z
    centerY = (oldRectH / 2.0f - panel.pos.y) / z

  # Adjust zoom with square root of proportional scaling - moderate the zoom increase
  # when panel gets bigger to keep map elements reasonably sized
  let
    oldDiagonal = sqrt(oldRectW * oldRectW + oldRectH * oldRectH)
    newDiagonal = sqrt(rectW * rectW + rectH * rectH)
    zoomFactor = sqrt(newDiagonal / oldDiagonal)

  panel.zoom = clamp(panel.zoom * zoomFactor, panel.minZoom, panel.maxZoom)

  # Recalculate pan to keep the same center point
  let newZ = panel.zoom * panel.zoom
  panel.pos.x = rectW / 2.0f - centerX * newZ
  panel.pos.y = rectH / 2.0f - centerY * newZ

  # Update previous size
  previousPanelSize = currentSize

proc drawWorldMap*(panel: Panel) {.measure.} =
  ## Draw the world map.

  if replay == nil or replay.mapSize[0] == 0 or replay.mapSize[1] == 0:
    # Replay has not been loaded yet.
    return

  ## Draw the world map.
  if settings.lockFocus:
    centerAt(panel, selection)

  panel.beginPanAndZoom()

  if panel.hasMouse:
    useSelections(panel)

  agentControls()

  if panel.zoom < 3:
    drawWorldMini()
  else:
    drawWorldMain()

  panel.endPanAndZoom()
