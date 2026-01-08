import
  std/[math, os, strutils, tables, strformat, random, times],
  vmath, windy, boxy,
  common, actions, utils, replays,
  pathfinding, tilemap, pixelator, shaderquad,
  panels, objectinfo

proc foo() =
  echo window.size.x, "x", window.size.y

const
  TILE_SIZE = 128
  TS = 1.0 / TILE_SIZE.float32 # Tile scale.

proc centerAt*(zoomInfo: ZoomInfo, entity: Entity)

var
  terrainMap*: TileMap
  visibilityMapStep*: int = -1
  visibilityMapSelectionId*: int = -1
  visibilityMapLockFocus*: bool = false
  visibilityMap*: TileMap
  px*: Pixelator
  sq*: ShaderQuad
  previousPanelSize*: Vec2 = vec2(0, 0)
  needsInitialFit*: bool = true

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

proc generateTerrainMap(): TileMap =
  ## Generate a 1024x1024 texture where each pixel is a byte index into the 16x16 tile map.
  let
    width = ceil(replay.mapSize[0].float32 / 32.0f).int * 32
    height = ceil(replay.mapSize[1].float32 / 32.0f).int * 32

  var terrainMap = newTileMap(
    width = width,
    height = height,
    tileSize = 128,
    atlasPath = dataDir & "/terrain/blob7x8.png"
  )

  var asteroidMap: seq[bool] = newSeq[bool](width * height)
  # Fill the asteroid map with ground (true).
  for y in 0 ..< height:
    for x in 0 ..< width:
      if x >= replay.mapSize[0] or y >= replay.mapSize[1]:
        # Clear the margins.
        asteroidMap[y * width + x] = true
      else:
        asteroidMap[y * width + x] = false

  # Walk the walls and generate a map of which tiles are present.
  for obj in replay.objects:
    if obj.typeName == "wall":
      let pos = obj.location.at(0)
      asteroidMap[pos.y * width + pos.x] = true


  # Generate the tile edges.
  for i in 0 ..< terrainMap.indexData.len:
    let x = i mod width
    let y = i div width

    proc get(map: seq[bool], x: int, y: int): int =
      if x < 0 or y < 0 or x >= width or y >= height:
        return 1
      if map[y * width + x]:
        return 1
      return 0

    var tile: uint8 = 0
    if asteroidMap[y * width + x]:
      tile = 49
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

    # Randomize the solid tiles:
    for i in 0 ..< terrainMap.indexData.len:
      if terrainMap.indexData[i] == 29 or terrainMap.indexData[i] == 18:
        terrainMap.indexData[i] = (50 + weightedRandomInt(@[100, 50, 25, 10, 5, 2])).uint8

  terrainMap.setupGPU()
  return terrainMap

proc rebuildVisibilityMap*(visibilityMap: TileMap) =
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

proc generateVisibilityMap(): TileMap =
  ## Generate a 1024x1024 texture where each pixel is a byte index into the 16x16 tile map.
  let
    width = ceil(replay.mapSize[0].float32 / 32.0f).int * 32
    height = ceil(replay.mapSize[1].float32 / 32.0f).int * 32

  var visibilityMap = newTileMap(
    width = width,
    height = height,
    tileSize = 64,
    atlasPath = dataDir & "/fog7x8.png"
  )
  visibilityMap.rebuildVisibilityMap()
  visibilityMap.setupGPU()
  return visibilityMap

proc updateVisibilityMap*(visibilityMap: TileMap) =
  ## Update the visibility map.
  visibilityMap.rebuildVisibilityMap()
  visibilityMap.updateGPU()

proc getProjectionView*(): Mat4 =
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

proc useSelections*(zoomInfo: ZoomInfo) =
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
      centerAt(zoomInfo, selection)

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

proc getAgentOrientation*(agent: Entity, step: int): Orientation =
  ## Get the orientation of the agent.
  for i in countdown(step, 0):
    let actionId = agent.actionId.at(i)
    if actionId >= 0 and actionId < replay.actionNames.len:
      let actionName = replay.actionNames[actionId]
      if actionName == "move_north":
        return N
      elif actionName == "move_south":
        return S
      elif actionName == "move_west":
        return W
      elif actionName == "move_east":
        return E
      break
  return S

proc drawObjects*() =
  ## Draw the objects on the map.
  for thing in replay.objects:
    let typeName = thing.typeName
    let pos = thing.location.at().xy
    case typeName
    of "wall":
      discard
    of "agent":
      let agent = thing
      # Agents don't do orientation anymore.
      # var agentImage = case agent.orientation.at:
      #   of 0: "agents/agent.n"
      #   of 1: "agents/agent.s"
      #   of 2: "agents/agent.w"
      #   of 3: "agents/agent.e"
      #   else:
      #     echo "Unknown orientation: ", agent.orientation.at
      #     "agents/agent.n"

      # Find last orientation action.
      var agentImage = "agents/agent." & getAgentOrientation(agent, step).char

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

proc drawVisualRanges*(alpha = 0.5) =
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


proc drawFogOfWar*() =
  ## Draw the fog of war.
  drawVisualRanges(alpha = 1.0)

proc drawTrajectory*() =
  ## Draw the trajectory of the selected object, with footprints or a future arrow.
  if selection != nil and selection.location.len > 1:
    var prevDirection = S
    for i in 1 ..< replay.maxSteps:
      let
        loc0 = selection.location.at(i - 1)
        loc1 = selection.location.at(i)
        cx0 = loc0.x.int
        cy0 = loc0.y.int
        cx1 = loc1.x.int
        cy1 = loc1.y.int

      if cx0 != cx1 or cy0 != cy1:
        var thisDirection: Orientation =
          if cx1 > cx0:
            E
          elif cx1 < cx0:
            W
          elif cy1 > cy0:
            S
          else:
            N
        let a = 1.0f - abs(i - step).float32 / 200.0f
        if a > 0:
          var
            tint = color(0, 0, 0, a)
            image = ""

          let isAgent = selection.typeName == "agent"
          if i <= step:
            image = "agents/tracks." & prevDirection.char & thisDirection.char
          else:
            #image = "agents/path"
            break

          # Draw centered at the tile with rotation. Use a slightly larger scale on diagonals.
          px.drawSprite(
            image,
            ivec2(cx0.int32, cy0.int32) * TILE_SIZE,
          )
        prevDirection = thisDirection

proc drawAgentDecorations*() =
  # Draw energy bars, shield and frozen status.
  for agent in replay.agents:
    if agent.isFrozen.at:
      px.drawSprite(
        "agents/frozen",
        agent.location.at.xy.ivec2 * TILE_SIZE,
      )

proc drawGrid*() =
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

proc drawPlannedPath*() =
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
      px.drawSprite(
        "agents/path",
        pos0.ivec2 * TILE_SIZE
      )
      currentPos = action.pos

    # Draw final queued objective.
    if agentObjectives.hasKey(agentId):
      let objectives = agentObjectives[agentId]
      if objectives.len > 0:
        let objective = objectives[^1]
        if objective.kind in {Move, Bump}:
          px.drawSprite(
            "objects/selection",
            objective.pos.ivec2 * TILE_SIZE
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
          px.drawSprite(
            "agents/arrow",
            approachPos.ivec2 * TILE_SIZE + offset.ivec2
          )

proc drawSelection*() =
  # Draw selection.
  if selection != nil:
    px.drawSprite(
      "objects/selection",
      selection.location.at.xy.ivec2 * TILE_SIZE,
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

proc drawTerrain*() =
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

proc drawObjectPips*() =
  ## Draw the pips for the objects on the minimap.
  for obj in replay.objects:
    if obj.typeName == "wall":
      continue
    let pipName = "minimap/" & obj.typeName
    if pipName in px:
      let loc = obj.location.at(step).xy
      px.drawSprite(
        pipName,
        loc.ivec2 * TILE_SIZE
      )
    else:
      echo "pipName not found: ", pipName

proc drawWorldMini*() =

  const wallTypeName = "wall"
  const agentTypeName = "agent"

  drawTerrain()

  # Overlays
  if settings.showVisualRange:
    drawVisualRanges()
  elif settings.showFogOfWar:
    drawFogOfWar()

  drawObjectPips()

  px.flush(getProjectionView() * scale(vec3(TS, TS, 1.0f)))

proc centerAt*(zoomInfo: ZoomInfo, entity: Entity) =
  ## Center the map on the given entity.
  if entity.isNil:
    return
  let location = entity.location.at(step).xy
  let rectW = zoomInfo.rect.w.float32
  let rectH = zoomInfo.rect.h.float32
  if rectW <= 0 or rectH <= 0:
    return
  let z = zoomInfo.zoom * zoomInfo.zoom
  zoomInfo.pos.x = rectW / 2.0f - location.x.float32 * z
  zoomInfo.pos.y = rectH / 2.0f - location.y.float32 * z

proc drawWorldMain*() =
  ## Draw the world map.
  drawTerrain()
  drawTrajectory()

  drawObjects()
  drawSelection()

  drawAgentDecorations()
  drawPlannedPath()

  px.flush(getProjectionView() * scale(vec3(TS, TS, 1.0f)))

  if settings.showVisualRange:
    drawVisualRanges()
  if settings.showFogOfWar:
    drawFogOfWar()
  if settings.showGrid:
    drawGrid()


proc fitFullMap*(zoomInfo: ZoomInfo) =
  ## Set zoom and pan so the full map fits in the panel.
  if replay.isNil:
    return
  let rectW = zoomInfo.rect.w.float32
  let rectH = zoomInfo.rect.h.float32
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
  zoomInfo.zoom = clamp(sqrt(zoomScale), zoomInfo.minZoom, zoomInfo.maxZoom)
  let
    cx = (mapMinX + mapMaxX) / 2.0f
    cy = (mapMinY + mapMaxY) / 2.0f
    z = zoomInfo.zoom * zoomInfo.zoom
  zoomInfo.pos.x = rectW / 2.0f - cx * z
  zoomInfo.pos.y = rectH / 2.0f - cy * z

proc fitVisibleMap*(zoomInfo: ZoomInfo) =
  ## Set zoom and pan so the visible area (union of all agent vision ranges) fits in the panel.
  if replay.isNil:
    return

  if replay.agents.len == 0:
    fitFullMap(zoomInfo)
    return

  let rectSize = vec2(zoomInfo.rect.w.float32, zoomInfo.rect.h.float32)

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
    fitFullMap(zoomInfo)
    return

  let
    visibleSize = maxPos - minPos
    zoomScale = min(rectSize.x / visibleSize.x, rectSize.y / visibleSize.y)
    center = (minPos + maxPos) / 2.0f
    zoom = clamp(sqrt(zoomScale), zoomInfo.minZoom, zoomInfo.maxZoom)

  zoomInfo.zoom = zoom
  zoomInfo.pos = rectSize / 2.0f - center * (zoom * zoom)

proc adjustPanelForResize*(zoomInfo: ZoomInfo) =
  ## Adjust pan and zoom when panel resizes to show the same portion of the map.
  let currentSize = vec2(zoomInfo.rect.w.float32, zoomInfo.rect.h.float32)

  # Skip if this is the first time or no change
  if previousPanelSize.x <= 0 or previousPanelSize.y <= 0 or currentSize == previousPanelSize:
    previousPanelSize = currentSize
    return

  # Calculate current center point in world coordinates using previous panel size
  let
    oldRectW = previousPanelSize.x
    oldRectH = previousPanelSize.y
    rectW = zoomInfo.rect.w.float32
    rectH = zoomInfo.rect.h.float32
    z = zoomInfo.zoom * zoomInfo.zoom
    centerX = (oldRectW / 2.0f - zoomInfo.pos.x) / z
    centerY = (oldRectH / 2.0f - zoomInfo.pos.y) / z

  # Adjust zoom with square root of proportional scaling - moderate the zoom increase
  # when panel gets bigger to keep map elements reasonably sized
  let
    oldDiagonal = sqrt(oldRectW * oldRectW + oldRectH * oldRectH)
    newDiagonal = sqrt(rectW * rectW + rectH * rectH)
    zoomFactor = sqrt(newDiagonal / oldDiagonal)

  zoomInfo.zoom = clamp(zoomInfo.zoom * zoomFactor, zoomInfo.minZoom, zoomInfo.maxZoom)

  # Recalculate pan to keep the same center point
  let newZ = zoomInfo.zoom * zoomInfo.zoom
  zoomInfo.pos.x = rectW / 2.0f - centerX * newZ
  zoomInfo.pos.y = rectH / 2.0f - centerY * newZ

  # Update previous size
  previousPanelSize = currentSize

proc drawWorldMap*(zoomInfo: ZoomInfo) =
  ## Draw the world map.

  if replay == nil or replay.mapSize[0] == 0 or replay.mapSize[1] == 0:
    # Replay has not been loaded yet.
    return

  if needsInitialFit:
    fitFullMap(zoomInfo)
    var baseEntity: Entity = nil
    for obj in replay.objects:
      if obj.typeName == "assembler":
        baseEntity = obj
        break
    if baseEntity != nil:
      centerAt(zoomInfo, baseEntity)
    needsInitialFit = false

  ## Draw the world map.
  if settings.lockFocus:
    centerAt(zoomInfo, selection)

  zoomInfo.beginPanAndZoom()

  if zoomInfo.hasMouse:
    useSelections(zoomInfo)

  agentControls()

  if zoomInfo.zoom < 7:
    drawWorldMini()
  else:
    drawWorldMain()

  zoomInfo.endPanAndZoom()
