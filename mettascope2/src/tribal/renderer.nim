import
  std/[strformat, tables],
  boxy, vmath, windy, chroma,
  common, environment, utils, colors

proc useSelections*() =
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
  # Draw the floor tiles everywhere first as the base layer
  for x in 0 ..< MapWidth:
    for y in 0 ..< MapHeight:
 
      let tileColor = env.tileColors[x][y]
      
      let finalR = min(tileColor.r * tileColor.intensity, 1.5)
      let finalG = min(tileColor.g * tileColor.intensity, 1.5)
      let finalB = min(tileColor.b * tileColor.intensity, 1.5)
      
      if env.terrain[x][y] == Water:
        let waterBlend = 0.7  # How much water color to keep
        let r = finalR * (1.0 - waterBlend) + 0.3 * waterBlend
        let g = finalG * (1.0 - waterBlend) + 0.5 * waterBlend
        let b = finalB * (1.0 - waterBlend) + 0.8 * waterBlend
        bxy.drawImage("objects/floor", ivec2(x, y).vec2, angle = 0, scale = 1/200, tint = color(r, g, b, 1.0))
      else:
        bxy.drawImage("objects/floor", ivec2(x, y).vec2, angle = 0, scale = 1/200, tint = color(finalR, finalG, finalB, 1.0))

proc drawTerrain*() =
  for x in 0 ..< MapWidth:
    for y in 0 ..< MapHeight:
      case env.terrain[x][y]
      of Wheat:
        bxy.drawImage("objects/wheat_field", ivec2(x, y).vec2, angle = 0, scale = 1/200)
      of Tree:
        bxy.drawImage("objects/palm_tree", ivec2(x, y).vec2, angle = 0, scale = 1/200)
      else:
        discard

proc generateWallSprites(): seq[string] =
  result = newSeq[string](16)
  for i in 0 .. 15:
    var suffix = ""
    if (i and 8) != 0: suffix.add("n")
    if (i and 4) != 0: suffix.add("w")  
    if (i and 2) != 0: suffix.add("s")
    if (i and 1) != 0: suffix.add("e")
    
    if suffix.len > 0:
      result[i] = "objects/wall." & suffix
    else:
      result[i] = "objects/wall"

const wallSprites = generateWallSprites()

type WallTile = enum
  WallNone = 0,
  WallE = 1,
  WallS = 2,
  WallW = 4,
  WallN = 8,
  WallSE = 2 or 1,
  WallNW = 8 or 4,

proc drawWalls*() =
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
        
        let brightness = 0.3  # Fixed wall brightness
        let wallTint = color(brightness, brightness, brightness, 1.0)
        
        bxy.drawImage(wallSprites[tile], vec2(x.float32, y.float32), 
                     angle = 0, scale = 1/200, tint = wallTint)

  for fillPos in wallFills:
    let brightness = 0.3  # Fixed wall fill brightness
    let fillTint = color(brightness, brightness, brightness, 1.0)
    bxy.drawImage("objects/wall.fill", fillPos.vec2 + vec2(0.5, 0.3), 
                  angle = 0, scale = 1/200, tint = fillTint)

proc drawObjects*() =
  for x in 0 ..< MapWidth:
    for y in 0 ..< MapHeight:
      if env.grid[x][y] != nil:
        let thing = env.grid[x][y]
        case thing.kind
        of Wall:
          discard
        of Agent:
          let agent = thing
          var agentImage = case agent.orientation:
            of N: "agents/agent.n"
            of S: "agents/agent.s"
            of E: "agents/agent.e"
            of W: "agents/agent.w"
            of NW: "agents/agent.w"  # Use west sprite for NW
            of NE: "agents/agent.e"  # Use east sprite for NE
            of SW: "agents/agent.w"  # Use west sprite for SW
            of SE: "agents/agent.e"  # Use east sprite for SE
          bxy.drawImage(
            agentImage,
            ivec2(x, y).vec2,
            angle = 0,
            scale = 1/200,
            tint = generateEntityColor("agent", agent.agentId)
          )
        of Altar:
          bxy.drawImage(
            "objects/altar",
            ivec2(x, y).vec2,
            angle = 0,
            scale = 1/200,
            tint = getAltarColor(ivec2(x, y))
          )
        of Converter:
          bxy.drawImage(
            "objects/converter",
            ivec2(x, y).vec2,
            angle = 0,
            scale = 1/200
          )
        of Mine, Spawner:
          let imageName = if thing.kind == Mine: "objects/mine" else: "objects/spawner"
          bxy.drawImage(
            imageName,
            ivec2(x, y).vec2,
            angle = 0,
            scale = 1/200
          )
        of Clippy:
          # Map diagonal orientations to cardinal sprites
          let spriteDir = case thing.orientation:
            of N: "n"
            of S: "s"
            of E, NE, SE: "e"
            of W, NW, SW: "w"
          bxy.drawImage(
            "agents/clippy.color." & spriteDir,
            ivec2(x, y).vec2,
            angle = 0,
            scale = 1/200
          )
        of Armory, Forge, ClayOven, WeavingLoom:
          let imageName = case thing.kind:
            of Armory: "objects/armory"
            of Forge: "objects/forge"
            of ClayOven: "objects/clay_oven"
            of WeavingLoom: "objects/weaving_loom"
            else: ""
          bxy.drawImage(
            imageName,
            ivec2(x, y).vec2,
            angle = 0,
            scale = 1/200
          )

proc drawVisualRanges*(alpha = 0.2) =
  var visibility: array[MapWidth, array[MapHeight, bool]]
  for agent in env.agents:
    for i in 0 ..< ObservationWidth:
      for j in 0 ..< ObservationHeight:
        let
          gridPos = (agent.pos + ivec2(i - ObservationWidth div 2, j -
              ObservationHeight div 2))

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
  drawVisualRanges(alpha = 1.0)

proc drawActions*() =
  discard

proc drawObservations*() =
  if settings.showObservations > -1 and selection != nil and selection.kind == Agent:
    bxy.drawText(
      "observationTitle",
      translate((selection.pos - ivec2(ObservationWidth div 2,
          ObservationHeight div 2)).vec2 * 64 + vec2(-32, -64)),
      typeface,
      $ObservationName(settings.showObservations),
      20,
      color(1, 1, 1, 1)
    )
    for x in 0 ..< ObservationWidth:
      for y in 0 ..< ObservationHeight:
        let
          gridPos = (selection.pos + ivec2(x - ObservationWidth div 2, y -
              ObservationHeight div 2))
          value = env.observations[selection.agentId][
              settings.showObservations][x][y]

        bxy.drawText(
          "observation" & $x & $y,
          translate(gridPos.vec2 * 64 + vec2(-28, -28)),
          typeface,
          $value,
          20,
          color(1, 1, 1, 1)
        )

proc drawAgentDecorations*() =
  for agent in env.agents:
    if agent.frozen > 0:
      bxy.drawImage(
        "agents/frozen",
        agent.pos.vec2,
        angle = 0,
        scale = 1/200
      )

proc drawGrid*() =
  for x in 0 ..< MapWidth:
    for y in 0 ..< MapHeight:
      bxy.drawImage(
        "view/grid",
        ivec2(x, y).vec2,
        angle = 0,
        scale = 1/200
      )

proc drawSelection*() =
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
pos: ({selection.pos.x}, {selection.pos.y})
      """
    of Agent:
      info = &"""
Agent
agentId: {selection.agentId}
orientation: {selection.orientation}
ore: {selection.inventoryOre}
batteries: {selection.inventoryBattery}
water: {selection.inventoryWater}
wheat: {selection.inventoryWheat}
wood: {selection.inventoryWood}
reward: {selection.reward}
frozen: {selection.frozen}
      """
    of Altar:
      info = &"""
Altar
hearts: {selection.hearts}
cooldown: {selection.cooldown}
      """
    of Converter:
      info = &"""
Converter
cooldown: {selection.cooldown}
ready: {selection.cooldown == 0}
      """
    of Mine:
      info = &"""
Mine
resources: {selection.resources}
cooldown: {selection.cooldown}
      """
    of Spawner:
      info = &"""
Spawner
cooldown: {selection.cooldown}
spawn ready: {selection.cooldown == 0}
      """
    of Clippy:
      info = &"""
Clippy
home: ({selection.homeSpawner.x}, {selection.homeSpawner.y})
target: ({selection.targetPos.x}, {selection.targetPos.y})
      """
    of Armory:
      info = &"""
Armory
pos: ({selection.pos.x}, {selection.pos.y})
cooldown: {selection.cooldown}
      """
    of Forge:
      info = &"""
Forge
pos: ({selection.pos.x}, {selection.pos.y})
cooldown: {selection.cooldown}
      """
    of ClayOven:
      info = &"""
Clay Oven
pos: ({selection.pos.x}, {selection.pos.y})
cooldown: {selection.cooldown}
      """
    of WeavingLoom:
      info = &"""
Weaving Loom
pos: ({selection.pos.x}, {selection.pos.y})
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
