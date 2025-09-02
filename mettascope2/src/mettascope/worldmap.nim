import
  std/[strformat],
  boxy, vmath, windy,
  common, panels, actions, utils, replays

proc agentColor*(id: int): Color =
  ## Get the color for an agent.
  let f = id.float32
  color(
    f * Pi mod 1.0,
    f * E mod 1.0,
    f * sqrt(2.0) mod 1.0,
    1.0
  )

proc useSelections*() =
  ## Reads the mouse position and selects the thing under it.
  if window.buttonPressed[MouseLeft]:
    selection = nil
    let
      mousePos = bxy.getTransform().inverse * window.mousePos.vec2
      gridPos = (mousePos + vec2(0.5, 0.5)).ivec2
    if gridPos.x >= 0 and gridPos.x < replay.mapSize[0] and
      gridPos.y >= 0 and gridPos.y < replay.mapSize[1]:
        for obj in replay.objects:
          if obj.location.at(step).xy == gridPos:
            selection = obj
            break

proc drawFloor*() =
  # Draw the floor tiles.
  for x in 0 ..< replay.mapSize[0]:
    for y in 0 ..< replay.mapSize[1]:
      bxy.drawImage("objects/floor", ivec2(x.int32, y.int32).vec2, angle = 0, scale = 1/200)

const wallSprites = @[
  "objects/wall",
  "objects/wall.e",
  "objects/wall.s",
  "objects/wall.se",
  "objects/wall.w",
  "objects/wall.we",
  "objects/wall.ws",
  "objects/wall.wse",
  "objects/wall.n",
  "objects/wall.ne",
  "objects/wall.ns",
  "objects/wall.nse",
  "objects/wall.nw",
  "objects/wall.nwe",
  "objects/wall.nws",
  "objects/wall.nwse",
]

type WallTile = enum
  WallNone = 0,
  WallE = 1,
  WallS = 2,
  WallW = 4,
  WallN = 8,
  WallSE = 2 or 1,
  WallNW = 8 or 4,

proc drawWalls*() =
  ## Draw the walls on the map.
  var grid = newSeq2D[bool](replay.mapSize[0], replay.mapSize[1])
  let wallTypeId = replay.typeNames.find("wall")
  for obj in replay.objects:
    if obj.typeId == wallTypeId:
      let pos = obj.location.at
      grid[pos.x][pos.y] = true

  template hasWall(x: int, y: int): bool =
    x >= 0 and x < replay.mapSize[0] and
    y >= 0 and y < replay.mapSize[1] and
    grid[x][y]

  var wallFills: seq[IVec2]
  for x in 0 ..< replay.mapSize[0]:
    for y in 0 ..< replay.mapSize[1]:
      if grid[x][y]:
        var tile = 0'u16
        if hasWall(x, y + 1): tile = tile or WallS.uint16
        if hasWall(x + 1, y): tile = tile or WallE.uint16
        if hasWall(x, y - 1): tile = tile or WallN.uint16
        if hasWall(x - 1, y): tile = tile or WallW.uint16

        if (tile and WallSE.uint16) == WallSE.uint16 and
            hasWall(x + 1, y + 1):
          wallFills.add(ivec2(x.int32, y.int32))
          if (tile and WallNW.uint16) == WallNW.uint16 and
              hasWall(x - 1, y - 1) and
              hasWall(x - 1, y + 1) and
              hasWall(x + 1, y - 1):
            continue
        bxy.drawImage(wallSprites[tile], vec2(x.float32, y.float32), angle = 0, scale = 1/200)

  for fillPos in wallFills:
    bxy.drawImage("objects/wall.fill", fillPos.vec2 + vec2(0.5, 0.3), angle = 0, scale = 1/200)

proc drawObjects*() =
  ## Draw the objects on the map.
  for thing in replay.objects:
    let typeName = replay.typeNames[thing.typeId]
    let pos = thing.location.at().xy
    case typeName
    of "wall":
      discard
      # bxy.drawImage("objects/wall",  pos.vec2, angle = 0, scale = 1/200)
    of "agent":
      let agent = thing
      var agentImage = case agent.orientation.at:
        of 0: "agents/agent.n"
        of 1: "agents/agent.s"
        of 2: "agents/agent.e"
        of 3: "agents/agent.w"
        else:
          echo "Unknown orientation: ", agent.orientation.at
          "agents/agent.n"
      bxy.drawImage(
        agentImage,
        pos.vec2,
        angle = 0,
        scale = 1/200,
        tint = agentColor(agent.agentId)
      )
    else:
      bxy.drawImage(
        "objects/" & typeName,
        pos.vec2,
        angle = 0,
        scale = 1/200
      )

proc drawVisualRanges*(alpha = 0.2) =
  ## Draw the visual ranges of the selected agent.
  var visibility = newSeq2D[bool](replay.mapSize[0], replay.mapSize[1])
  let agentTypeId = replay.typeNames.find("agent")
  for obj in replay.objects:
    if obj.typeId == agentTypeId:
      if selection != nil and
        selection.typeId == agentTypeId and
        selection.agentId != obj.agentId:
          continue
      let agent = obj
      for i in 0 ..< agent.visionSize:
        for j in 0 ..< agent.visionSize:
          let
            center = ivec2((agent.visionSize div 2).int32, (agent.visionSize div 2).int32)
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

proc drawActions*() =
  ## Draw the actions of the selected agent.
   # # Draw all possible attacks:
  # for agent in env.agents:
  #   for i in 1 .. 9:
  #     let
  #       distance = 1 + (i - 1) div 3
  #       offset = -((i - 1) mod 3 - 1)
  #       targetPos = agent.pos + relativeLocation(agent.orientation, distance, offset)
  #     bxy.drawImage(
  #       "empty",
  #       targetPos.vec2 * 64,
  #       angle = 0,
  #       scale = 2,
  #       tint = color(1, 0, 0, 1)
  #     )

  # Draw attack actions
  # for agentId, action in actionsArray:
  #   if action[0] == 4:
  #     let
  #       distance = 1 + (action[1].int - 1) div 3
  #       offset = -((action[1].int - 1) mod 3 - 1)
  #       agent = env.agents[agentId]
  #       targetPos = agent.pos + relativeLocation(agent.orientation, distance, offset)
  #     if agent.energy > MapObjectAgentAttackCost:
  #       discard
  #       # bxy.drawImage(
  #       #   "fire",
  #       #   targetPos.vec2 * 64,
  #       #   angle = 0
  #       # )
  #       # bxy.drawBubbleLine(
  #       #   agent.pos.vec2 * 64,
  #       #   targetPos.vec2 * 64,
  #       #   color(1, 0, 0, 0.5)
  #       # )

proc drawAgentDecorations*() =
  # Draw energy bars, shield and frozen status.
  for agent in replay.agents:
    if agent.isFrozen.at:
      bxy.drawImage(
        "agents/frozen",
        agent.location.at.xy.vec2,
        angle = 0,
        scale = 1/200
      )

proc drawGrid*() =
  # Draw the grid.
  for x in 0 ..< replay.mapSize[0]:
    for y in 0 ..< replay.mapSize[1]:
      bxy.drawImage(
        "view/grid",
        ivec2(x.int32, y.int32).vec2,
        angle = 0,
        scale = 1/200
      )

proc drawInventory*() =
  # Draw the inventory.
  for obj in replay.objects:
    let inventory = obj.inventory.at
    var numItems = 0
    for itemAmount in inventory:
      numItems += itemAmount.count
    let widthItems = (numItems.float32 * 0.1).clamp(0.0, 1.0)
    var x = -widthItems / 2
    var xAdvance = widthItems / numItems.float32
    for itemAmount in inventory:
      let itemName = replay.itemNames[itemAmount.itemId]
      for i in 0 ..< itemAmount.count:
        bxy.drawImage(
          "resources/" & itemName,
          obj.location.at.xy.vec2 + vec2(x.float32, -0.5),
          angle = 0,
          scale = 1/200 / 4
        )
        x += xAdvance

proc drawSelection*() =
  # Draw selection.
  if selection != nil:
    bxy.drawImage(
      "selection",
      selection.location.at.xy.vec2,
      angle = 0,
      scale = 1/200
    )

proc drawInfoText*() =

  var info = ""

  if selection != nil:
    let typeName = replay.typeNames[selection.typeId]
    case typeName
    of "wall":
      info = &"""
Wall
      """
    of "agent":
      info = &"""
Agent
  agentId: {selection.agentId}
  orientation: {selection.orientation.at}
  inventory: {selection.inventory.at}
  reward: {selection.currentReward.at}
  frozen: {selection.isFrozen.at}
      """
    else:
      info = &"""
{typeName}
  inventory: {selection.inventory.at}
      """
  else:
    info = &"""
World
  size: {replay.mapSize[0]}x{replay.mapSize[1]}
  speed: {1/playSpeed:0.3f}
  step: {step}
    """
  bxy.drawText(
    "info",
    translate(vec2(10, 10)),
    typeface,
    info,
    16,
    color(1, 1, 1, 1)
  )

proc drawWorldMap*(panel: Panel) =

  panel.beginPanAndZoom()
  useSelections()
  agentControls()

  drawFloor()
  drawWalls()
  drawObjects()
  # drawActions()
  # drawAgentDecorations()

  if settings.showGrid:
    drawGrid()
  if settings.showVisualRange:
    drawVisualRanges()

  drawSelection()
  drawInventory()

  if settings.showFogOfWar:
    drawFogOfWar()

  panel.endPanAndZoom()

  drawInfoText()
