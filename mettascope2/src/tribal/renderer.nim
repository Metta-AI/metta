import
  std/[strformat, tables],
  boxy, vmath, windy, chroma,
  common, panels, environment, simulation, utils

proc agentColor*(id: int): Color =
  if id >= 0 and id < agentVillageColors.len:
    return agentVillageColors[id]
  # Fallback for agents without village assignment
  let f = id.float32
  color(
    f * Pi mod 1.0,
    f * E mod 1.0,
    f * sqrt(2.0) mod 1.0,
    1.0
  )

proc altarColor*(pos: IVec2): Color =
  if altarColors.hasKey(pos):
    return altarColors[pos]
  # Fallback to white if no color assigned
  return color(1.0, 1.0, 1.0, 1.0)

proc generateVillageColor*(villageId: int): Color =
  let hue = (villageId.float32 * 137.5) mod 360.0 / 360.0  # Golden angle for color spacing
  let saturation = 0.7 + (villageId.float32 * 0.13) mod 0.3
  let lightness = 0.5 + (villageId.float32 * 0.17) mod 0.2
  # Convert HSL to RGB (simplified conversion)
  return color(hue, saturation, lightness, 1.0)

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
      # Get ice intensity for this tile (0.0 to 1.0)
      let iceLevel = env.iceIntensity[x][y]
      
      # First draw floor everywhere (water gets a blue tint, ice adds more blue)
      if env.terrain[x][y] == Water:
        bxy.drawImage("objects/floor", ivec2(x, y).vec2, angle = 0, scale = 1/200, tint = color(0.3, 0.5, 0.8, 1.0))
      elif iceLevel > 0:
        # Ice tinting: progressively more blue as ice intensity increases
        # Start with normal floor color, blend toward icy blue
        let iceTint = min(iceLevel, 1.0)  # Cap at 1.0
        let r = 1.0 - (iceTint * 0.4)  # Reduce red channel
        let g = 1.0 - (iceTint * 0.2)  # Slightly reduce green  
        let b = 1.0 + (iceTint * 0.3)  # Increase blue channel
        bxy.drawImage("objects/floor", ivec2(x, y).vec2, angle = 0, scale = 1/200, tint = color(r, g, min(b, 1.0), 1.0))
      else:
        bxy.drawImage("objects/floor", ivec2(x, y).vec2, angle = 0, scale = 1/200)

proc drawTerrain*() =
  # Draw terrain features on top of the floor
  for x in 0 ..< MapWidth:
    for y in 0 ..< MapHeight:
      case env.terrain[x][y]
      of Wheat:
        # Draw wheat field sprite on top of floor
        bxy.drawImage("objects/wheat_field", ivec2(x, y).vec2, angle = 0, scale = 1/200)
      of Tree:
        # Draw palm tree sprite on top of floor
        bxy.drawImage("objects/palm_tree", ivec2(x, y).vec2, angle = 0, scale = 1/200)
      else:
        discard  # Water and Empty don't need additional sprites

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

proc getAltarRadiance*(pos: IVec2): float32 =
  var maxRadiance = 0.0'f32
  
  # Check all altars in the environment
  for thing in env.things:
    if thing.kind == Altar:
      let altarPos = thing.pos
      let hearts = thing.hearts.float32
      
      # Calculate distance from this position to the altar
      let dx = abs(pos.x - altarPos.x).float32
      let dy = abs(pos.y - altarPos.y).float32
      let distance = max(dx, dy)  # Use Chebyshev distance for square radiance
      
      if distance <= 10:  # Maximum radiance radius
        # Calculate radiance: more hearts = brighter, closer = brighter
        # Hearts scale: 0-10 hearts is typical, can go higher
        let heartIntensity = min(hearts / 5.0, 2.0)  # Cap at 2.0 for very rich altars
        let distanceFalloff = 1.0 - (distance / 10.0)
        let radiance = heartIntensity * distanceFalloff * 0.8  # Max 80% brightness boost
        
        maxRadiance = max(maxRadiance, radiance)
  
  return maxRadiance

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
        
        # Calculate brightness based on nearby altars
        let radiance = getAltarRadiance(ivec2(x.int32, y.int32))
        let brightness = 0.2 + radiance  # Base darkness of 0.2, can go up to 1.0
        let wallTint = color(brightness, brightness, brightness, 1.0)
        
        bxy.drawImage(wallSprites[tile], vec2(x.float32, y.float32), 
                     angle = 0, scale = 1/200, tint = wallTint)

  for fillPos in wallFills:
    # Apply the same radiance to wall fills
    let radiance = getAltarRadiance(fillPos)
    let brightness = 0.2 + radiance
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
            tint = agentColor(agent.agentId)
          )
        of Altar:
          bxy.drawImage(
            "objects/altar",
            ivec2(x, y).vec2,
            angle = 0,
            scale = 1/200,
            tint = altarColor(ivec2(x, y))
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
          # Draw production buildings
          let imageName = case thing.kind:
            of Armory: "objects/armory"
            of Forge: "objects/forge"
            of ClayOven: "objects/clay_oven"
            of WeavingLoom: "objects/weaving_loom"
            else: ""  # Won't happen due to case constraint
          bxy.drawImage(
            imageName,
            ivec2(x, y).vec2,
            angle = 0,
            scale = 1/200
          )
      else:
        discard


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
  # Draw observations
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
  # Draw agent status indicators (frozen, etc.)
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
wander radius: {selection.wanderRadius}
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
