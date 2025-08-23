import
  std/[strformat],
  boxy, vmath, windy,
  common, panels, sim, actions, utils

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
    if gridPos.x >= 0 and gridPos.x < MapWidth and
       gridPos.y >= 0 and gridPos.y < MapHeight:
      let thing = env.grid[gridPos.x][gridPos.y]
      if thing != nil:
        selection = thing

proc drawFloor*() =
  # Draw the floor tiles.
  for x in 0 ..< MapWidth:
    for y in 0 ..< MapHeight:
      bxy.drawImage("objects/floor", ivec2(x, y).vec2, angle = 0, scale = 1/200)

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
  template hasWall(x: int, y: int): bool =
    x >= 0 and x < MapWidth and
    y >= 0 and y < MapHeight and
    env.grid[x][y] != nil and
    env.grid[x][y].kind == Wall

  var wallFills: seq[IVec2]
  for x in 0 ..< MapWidth:
    for y in 0 ..< MapHeight:
      let thing = env.grid[x][y]
      if thing != nil and thing.kind == Wall:
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
  for x in 0 ..< MapWidth:
    for y in 0 ..< MapHeight:
      if env.grid[x][y] != nil:
        let thing = env.grid[x][y]
        case thing.kind
        of Wall:
          discard
          # bxy.drawImage("objects/wall",  ivec2(x, y).vec2, angle = 0, scale = 1/200)
        of Agent:
          let agent = thing
          var agentImage = case agent.orientation:
            of N: "agents/agent.n"
            of S: "agents/agent.s"
            of E: "agents/agent.e"
            of W: "agents/agent.w"
          bxy.drawImage(
            agentImage,
            ivec2(x, y).vec2,
            angle = 0,
            scale = 1/200,
            tint = agentColor(agent.agentId)
          )

          # var face = case agent.orientation:
          #   of N: ivec2(0, -1)
          #   of S: ivec2(0, 1)
          #   of E: ivec2(1, 0)
          #   of W: ivec2(-1, 0)
          # bxy.drawImage(
          #   "bubble",
          #   (agent.pos + face).vec2 * 64,
          #   angle = 0,
          #   tint = color(1, 0, 0, 0.5)
          # )

          # var face2 = relativeLocation(agent.orientation, 2, 0)
          # bxy.drawImage(
          #   "bubble",
          #   (agent.pos + face).vec2 * 64,
          #   angle = 0,
          #   tint = color(1, 0, 0, 0.5)
          # )

        of Altar:
          bxy.drawImage(
            "objects/altar",
            ivec2(x, y).vec2,
            angle = 0,
            scale = 1/200
          )
        of Converter:
          bxy.drawImage(
            "objects/converter",
            ivec2(x, y).vec2,
            angle = 0,
            scale = 1/200
          )
        of Generator:
          let
            tint = color(0.5, 0.5, 1, 1)
          bxy.drawImage(
            "objects/generator",
            ivec2(x, y).vec2,
            angle = 0,
            scale = 1/200
          )
      else:
        discard


proc drawVisualRanges*(alpha = 0.2) =
  ## Draw the visual ranges of the selected agent.
  var visibility: array[MapWidth, array[MapHeight, bool]]
  for agent in env.agents:
    for i in 0 ..< ObservationWidth:
      for j in 0 ..< ObservationHeight:
        let
          gridPos = (agent.pos + ivec2(i - ObservationWidth div 2, j - ObservationHeight div 2))

        if gridPos.x >= 0 and gridPos.x < MapWidth and
           gridPos.y >= 0 and gridPos.y < MapHeight:
          visibility[gridPos.x][gridPos.y] = true

  for x in 0 ..< MapWidth:
    for y in 0 ..< MapHeight:
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
  for agentId, action in actionsArray:
    if action[0] == 4:
      let
        distance = 1 + (action[1].int - 1) div 3
        offset = -((action[1].int - 1) mod 3 - 1)
        agent = env.agents[agentId]
        targetPos = agent.pos + relativeLocation(agent.orientation, distance, offset)
      if agent.energy > MapObjectAgentAttackCost:
        discard
        # bxy.drawImage(
        #   "fire",
        #   targetPos.vec2 * 64,
        #   angle = 0
        # )
        # bxy.drawBubbleLine(
        #   agent.pos.vec2 * 64,
        #   targetPos.vec2 * 64,
        #   color(1, 0, 0, 0.5)
        # )

proc drawObservations*() =
  # Draw observations
  if settings.showObservations > -1 and selection != nil and selection.kind == Agent:
    bxy.drawText(
      "observationTitle",
      translate((selection.pos - ivec2(ObservationWidth div 2, ObservationHeight div 2)).vec2 * 64 + vec2(-32, -64)),
      typeface,
      $ObservationName(settings.showObservations),
      20,
      color(1, 1, 1, 1)
    )
    for x in 0 ..< ObservationWidth:
      for y in 0 ..< ObservationHeight:
        let
          gridPos = (selection.pos + ivec2(x - ObservationWidth div 2, y - ObservationHeight div 2))
          value = env.observations[selection.agentId][settings.showObservations][x][y]

        bxy.drawText(
          "observation" & $x & $y,
          translate(gridPos.vec2 * 64 + vec2(-28, -28)),
          typeface,
          $value,
          20,
          color(1, 1, 1, 1)
        )

proc drawAgentDecorations*() =
  # Draw energy bars, shield and frozen status.
  for agent in env.agents:
    # if agent.shield:
    #   bxy.drawImage(
    #     "shield",
    #     agent.pos.vec2 * 64,
    #     angle = 0
    #   )
    if agent.frozen > 0:
      bxy.drawImage(
        "agents/frozen",
        agent.pos.vec2,
        angle = 0,
        scale = 1/200
      )

proc drawGrid*() =
  # Draw the grid.
  for x in 0 ..< MapWidth:
    for y in 0 ..< MapHeight:
      bxy.drawImage(
        "view/grid",
        ivec2(x, y).vec2,
        angle = 0,
        scale = 1/200
      )

proc drawSelection*() =
  # Draw selection.
  if selection != nil:
    bxy.drawImage(
      "selection",
      selection.pos.vec2,
      angle = 0,
      scale = 1/200
    )

proc drawInfoText*() =

  var info = ""

  if selection != nil:
    case selection.kind
    of Wall:
      info = &"""
Wall
hp: {selection.hp}
      """
    of Agent:
      info = &"""
Agent
agentId: {selection.agentId}
energy: {selection.energy}
orientation: {selection.orientation}
inventory: {selection.inventory}
reward: {selection.reward}
frozen: {selection.frozen}
shield: {selection.shield}
hp: {selection.hp}
      """
    of Altar:
      info = &"""
Altar
hp: {selection.hp}
cooldown: {selection.cooldown}
      """
    of Converter:
      info = &"""
Converter
hp: {selection.hp}
cooldown: {selection.cooldown}
      """
    of Generator:
      info = &"""
Generator
hp: {selection.hp}
cooldown: {selection.cooldown}
      """
  else:
    info = &"""
speed: {1/playSpeed:0.3f}
step: {env.currentStep}
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
