import
  std/[math, os, strutils, tables, strformat, random],
  boxy, vmath, windy, fidget2/[hybridrender, common],
  common, panels, actions, utils, replays, objectinfo, pathfinding, tilemap

const TS = 1.0 / 64.0 # Tile scale.

var terrainMap*: TileMap

proc weightedRandomInt*(weights: seq[int]): int =
  ## Return a random integer between 0 and 7, with a weighted distribution.
  var r = rand(sum(weights))
  var acc = 0
  for i, w in weights:
    acc += w
    if r <= acc:
      return i
  doAssert false, "should not happen"

proc generateTileMap(): TileMap =
  ## Generate a 1024x1024 texture where each pixel is a byte index into the 16x16 tile map.
  let
    width = ceil(replay.mapSize[0].float32 / 32.0f).int * 32
    height = ceil(replay.mapSize[1].float32 / 32.0f).int * 32

  echo "Real map size: ", replay.mapSize[0], "x", replay.mapSize[1]
  echo "Tile map size: ", width, "x", height, " (multiples of 32)"

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
  let patternToTile = @[
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

proc buildAtlas*() =
  ## Build the atlas.
  for path in walkDirRec(dataDir):
    if path.endsWith(".png") and "fidget" notin path:
      let name = path.replace(dataDir & "/", "").replace(".png", "")
      bxy.addImage(name, readImage(path))

proc useSelections*(panel: Panel) =
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

proc drawObjects*() =
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
      bxy.drawImage(
        agentImage,
        pos.vec2,
        angle = 0,
        scale = TS
      )
    else:
      bxy.drawImage(
        replay.typeImages.getOrDefault(thing.typeName, "objects/unknown"),
        pos.vec2,
        angle = 0,
        scale = TS
      )

proc drawVisualRanges*(alpha = 0.2) =
  ## Draw the visual ranges of the selected agent.
  var visibility = newSeq2D[bool](replay.mapSize[0], replay.mapSize[1])
  let agentTypeName = "agent"
  for obj in replay.objects:
    if obj.typeName == agentTypeName:
      if selection != nil and
        selection.typeName == agentTypeName and
        selection.agentId != obj.agentId:
        continue
      let agent = obj
      for i in 0 ..< agent.visionSize:
        for j in 0 ..< agent.visionSize:
          let
            center = ivec2(
              (agent.visionSize div 2).int32,
              (agent.visionSize div 2).int32
            )
            gridPos = agent.location.at.xy - center + ivec2(i.int32, j.int32)

          if gridPos.x >= 0 and gridPos.x < replay.mapSize[0] and
            gridPos.y >= 0 and gridPos.y < replay.mapSize[1]:
            visibility[gridPos.x][gridPos.y] = true

  for x in 0 ..< replay.mapSize[0]:
    for y in 0 ..< replay.mapSize[1]:
      if not visibility[x][y]:
        bxy.drawRect(
          rect(x.float32 - 0.5, y.float32 - 0.5, 1, 1),
          color(0, 0, 0, alpha)
        )

proc drawFogOfWar*() =
  ## Draw the fog of war.
  drawVisualRanges(alpha = 1.0)

proc drawTrajectory*() =
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

proc drawActions*() =
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

proc drawAgentDecorations*() =
  # Draw energy bars, shield and frozen status.
  for agent in replay.agents:
    if agent.isFrozen.at:
      bxy.drawImage(
        "agents/frozen",
        agent.location.at.xy.vec2,
        angle = 0,
        scale = TS
      )

proc drawClippedStatus*() =
  # Draw the clipped status of the selected agent.
  for obj in replay.objects:
    if obj.isClipped.at:
      bxy.drawImage(
        "agents/frozen",
        obj.location.at.xy.vec2,
        angle = 0,
        scale = TS
      )

proc drawGrid*() =
  # Draw the grid.
  for x in 0 ..< replay.mapSize[0]:
    for y in 0 ..< replay.mapSize[1]:
      bxy.drawImage(
        "view/grid",
        ivec2(x.int32, y.int32).vec2,
        angle = 0,
        scale = TS
      )

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

proc drawSelection*() =
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

proc drawThoughtBubbles*() =
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

proc drawTerrain*() =
  ## Draw the terrain, space and asteroid tiles using the terrainMap tilemap.
  bxy.enterRawOpenGLMode()
  if terrainMap == nil:
    terrainMap = generateTileMap()

  let m = bxy.getTransform()
  let boxyView = mat4(
    m[0, 0], m[0, 1], m[0, 2], 0,
    m[1, 0], m[1, 1], m[1, 2], 0,
    0, 0, 0, 1,
    m[2, 0], m[2, 1], m[2, 2], 1
  )
  let projection = ortho(0.0f, window.size.x.float32, window.size.y.float32, 0.0f, -1.0f, 1.0f)
  let mvp = projection *
    boxyView *
    translate(vec3(-0.5, -0.5, 0)) *
    scale(vec3(
      terrainMap.width.float32,
      terrainMap.height.float32,
      1.0f
    ))

  terrainMap.draw(mvp, 2.0f, 1.5f)
  bxy.exitRawOpenGLMode()

proc drawWorldMini*() =
  const wallTypeName = "wall"
  const agentTypeName = "agent"

  drawTerrain()

  # Agents
  let scale = 3.0
  bxy.saveTransform()
  bxy.scale(vec2(scale, scale))

  for obj in replay.objects:
    if obj.typeName != agentTypeName:
      continue

    let loc = obj.location.at(step).xy
    bxy.drawImage("minimapPip", rect((loc.x.float32) / scale - 0.5, (
        loc.y.float32) / scale - 0.5, 1, 1), color(1, 1, 1, 1))

  bxy.restoreTransform()

  # Overlays
  if settings.showVisualRange:
    drawVisualRanges()
  elif settings.showFogOfWar:
    drawFogOfWar()

proc centerAt*(panel: Panel, entity: Entity) =
  ## Center the map on the given entity.
  ## TODO: Implement this.
  discard

proc drawWorldMain*() =
  ## Draw the world map.
  drawTerrain()
  drawTrajectory()
  drawObjects()
  drawActions()
  drawAgentDecorations()
  drawClippedStatus()
  drawSelection()
  drawPlannedPath()

  if settings.showVisualRange:
    drawVisualRanges()
  if settings.showFogOfWar:
    drawFogOfWar()
  if settings.showGrid:
    drawGrid()

  drawThoughtBubbles()

proc fitFullMap*(panel: Panel) =
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

proc drawWorldMap*(panel: Panel) =
  ## Draw the world map.
  panel.beginPanAndZoom()

  if panel.hasMouse:
    useSelections(panel)

  agentControls()

  if followSelection:
    centerAt(panel, selection)

  if panel.zoom < 3:
    drawWorldMini()
  else:
    drawWorldMain()

  panel.endPanAndZoom()
