import
  std/[strformat, tables],
  boxy, vmath, windy, chroma, pixie,
  game, terrain, colors

# Module-level variables that get set by the main draw procedure
var
  bxy*: Boxy
  env*: Environment
  selection*: Thing
  window*: Window
  typeface*: Typeface
  settings*: tuple[showGrid: bool, showObservations: int]
  play*: bool
  playSpeed*: float

# Re-export key types
export Environment, Thing

proc drawText*(
  imageKey: string,
  transform: Mat3,
  typeface: Typeface,
  text: string,
  size: float32,
  color: Color
) =
  ## Draw text on the screen.
  var font = newFont(typeface)
  font.size = size
  font.paint = color
  let
    arrangement = typeset(@[newSpan(text, font)], bounds = vec2(400, 600))
    globalBounds = arrangement.computeBounds(transform).snapToPixels()
    textImage = newImage(globalBounds.w.int, globalBounds.h.int)
    imageSpace = translate(-globalBounds.xy) * transform
  textImage.fillText(arrangement, imageSpace)
  
  bxy.addImage(imageKey, textImage)
  bxy.drawImage(imageKey, globalBounds.xy)

proc agentColor*(id: int): Color =
  ## Get the color for an agent based on their village
  # Agents now get colors from their village assignment stored in tribal module
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
  ## Get the color for an altar based on its village association
  if altarColors.hasKey(pos):
    return altarColors[pos]
  # Fallback to white if no color assigned
  return color(1.0, 1.0, 1.0, 1.0)

proc generateVillageColor*(villageId: int): Color =
  ## Generate a distinct color for a village
  # Use HSL to generate distinct colors with good saturation and lightness
  let hue = (villageId.float32 * 137.5) mod 360.0 / 360.0  # Golden angle for color spacing
  let saturation = 0.7 + (villageId.float32 * 0.13) mod 0.3
  let lightness = 0.5 + (villageId.float32 * 0.17) mod 0.2
  # Convert HSL to RGB (simplified conversion)
  return color(hue, saturation, lightness, 1.0)

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
  # Draw the floor tiles everywhere first as the base layer
  for x in 0 ..< MapWidth:
    for y in 0 ..< MapHeight:
      # First draw floor everywhere (water gets a blue tint)
      if env.terrain[x][y] == Water:
        bxy.drawImage("objects/floor", ivec2(x, y).vec2, angle = 0, scale = 1/200, tint = color(0.3, 0.5, 0.8, 1.0))
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
  ## Calculate brightness based on distance to nearby altars and their hearts
  ## Returns a value between 0.0 (dark) and 1.0 (bright)
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

          # var face = case agent.orientation:
          #   of N: ivec2(0, -1)
          #   of S: ivec2(0, 1)
          #   of E: ivec2(1, 0)
          #   of W: ivec2(-1, 0)
          #   of NW: ivec2(-1, -1)
          #   of NE: ivec2(1, -1)
          #   of SW: ivec2(-1, 1)
          #   of SE: ivec2(1, 1)
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
        of Mine:
          let
            tint = color(0.5, 0.5, 1, 1)
          bxy.drawImage(
            "objects/mine",
            ivec2(x, y).vec2,
            angle = 0,
            scale = 1/200
          )
        of Temple:
          bxy.drawImage(
            "objects/temple",
            ivec2(x, y).vec2,
            angle = 0,
            scale = 1/200
          )
        of Clippy:
          bxy.drawImage(
            "objects/clippy",
            ivec2(x, y).vec2,
            angle = 0,
            scale = 1/200
          )
        of Armory:
          # Draw armory building
          bxy.drawImage(
            "objects/armory",
            ivec2(x, y).vec2,
            angle = 0,
            scale = 1/200
          )
        of Forge:
          # Draw forge building
          bxy.drawImage(
            "objects/forge",
            ivec2(x, y).vec2,
            angle = 0,
            scale = 1/200
          )
        of ClayOven:
          # Draw clay oven building
          bxy.drawImage(
            "objects/clay_oven",
            ivec2(x, y).vec2,
            angle = 0,
            scale = 1/200
          )
        of WeavingLoom:
          # Draw weaving loom building
          bxy.drawImage(
            "objects/weaving_loom",
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

  # Attack actions removed - no longer rendering attack effects
  # for agentId, action in actionsArray:
  #   if action[0] == 4:
  #     # Attack effects would be rendered here

proc drawObservations*() =
  # Draw observations
  if settings.showObservations > -1 and selection != nil and selection.kind == Agent:
    drawText(
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

        drawText(
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
    of Temple:
      info = &"""
Temple
cooldown: {selection.cooldown}
spawn ready: {selection.cooldown == 0}
      """
    of Clippy:
      info = &"""
Clippy
home: ({selection.homeTemple.x}, {selection.homeTemple.y})
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
  drawText(
    "info",
    translate(vec2(10, 10)),
    typeface,
    info,
    16,
    color(1, 1, 1, 1)
  )

proc draw*(boxy: Boxy, environment: Environment, selected: Thing,
          winRef: Window = nil, typefaceRef: Typeface = nil,
          settingsRef: tuple[showGrid: bool, showObservations: int] = (false, -1),
          playRef: bool = false, playSpeedRef: float = 0.01) =
  ## Draw the world map
  
  # Update global references
  bxy = boxy
  env = environment
  selection = selected
  if winRef != nil:
    window = winRef
  if typefaceRef != nil:
    typeface = typefaceRef
  settings = settingsRef
  play = playRef
  playSpeed = playSpeedRef
  
  drawFloor()
  drawTerrain()
  drawWalls()
  drawObjects()
  drawAgentDecorations()
  drawSelection()
  
  if settings.showObservations > -1:
    drawObservations()
  
  drawInfoText()
