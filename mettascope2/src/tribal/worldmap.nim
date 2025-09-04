import
  std/[strformat, tables],
  boxy, vmath, windy, chroma,
  tribal

# Import necessary globals from tribalgrid
var
  window*: Window
  bxy*: Boxy
  env*: Environment
  selection*: Thing

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
      bxy.drawRect(
        rect = rect(x.float32 - 0.5, y.float32 - 0.5, 1, 1),
        color = color(0.1, 0.1, 0.1, 1.0)
      )

proc drawGridLines*() =
  # Draw grid lines
  for x in 0 .. MapWidth:
    bxy.drawRect(
      rect = rect(x.float32 - 0.51, -0.51, 0.02, MapHeight.float32 + 0.02),
      color = color(0.2, 0.2, 0.2, 0.5)
    )
  for y in 0 .. MapHeight:
    bxy.drawRect(
      rect = rect(-0.51, y.float32 - 0.51, MapWidth.float32 + 0.02, 0.02),
      color = color(0.2, 0.2, 0.2, 0.5)
    )

proc drawAgent*(agent: Thing, isSelected: bool) =
  let
    pos = vec2(agent.pos.x.float32, agent.pos.y.float32)
    agentCol = agentColor(agent.agentId)
  
  # Draw agent body
  bxy.drawRect(
    rect = rect(pos.x - 0.3, pos.y - 0.3, 0.6, 0.6),
    color = agentCol
  )
  
  # Draw orientation indicator
  let orientationOffset = case agent.orientation
    of N: vec2(0, -0.2)
    of S: vec2(0, 0.2)
    of E: vec2(0.2, 0)
    of W: vec2(-0.2, 0)
    of NE: vec2(0.14, -0.14)
    of NW: vec2(-0.14, -0.14)
    of SE: vec2(0.14, 0.14)
    of SW: vec2(-0.14, 0.14)
  
  bxy.drawRect(
    rect = rect(pos.x + orientationOffset.x - 0.05, pos.y + orientationOffset.y - 0.05, 0.1, 0.1),
    color = color(1, 1, 1, 0.8)
  )
  
  # Draw selection indicator
  if isSelected:
    bxy.drawRect(
      rect = rect(pos.x - 0.4, pos.y - 0.4, 0.8, 0.8),
      color = color(1, 1, 0, 0.5)
    )

proc drawClippy*(clippy: Thing) =
  let pos = vec2(clippy.pos.x.float32, clippy.pos.y.float32)
  
  # Draw clippy body (smaller than agent)
  bxy.drawRect(
    rect = rect(pos.x - 0.2, pos.y - 0.2, 0.4, 0.4),
    color = color(0.8, 0.3, 0.3, 1.0)
  )

proc drawAltar*(altar: Thing) =
  let
    pos = vec2(altar.pos.x.float32, altar.pos.y.float32)
    altarCol = altarColor(altar.pos)
  
  # Draw altar (triangular shape approximated with rect)
  bxy.drawRect(
    rect = rect(pos.x - 0.35, pos.y - 0.35, 0.7, 0.7),
    color = altarCol
  )

proc drawBuilding*(building: Thing) =
  let pos = vec2(building.pos.x.float32, building.pos.y.float32)
  
  # Draw building
  bxy.drawRect(
    rect = rect(pos.x - 0.4, pos.y - 0.4, 0.8, 0.8),
    color = color(0.4, 0.3, 0.2, 1.0)
  )

proc draw*(boxy: Boxy, environment: Environment, selected: Thing) =
  ## Draw the world map
  
  # Update global references
  bxy = boxy
  env = environment
  selection = selected
  
  drawFloor()
  drawGridLines()
  
  # Draw all game objects
  for x in 0 ..< MapWidth:
    for y in 0 ..< MapHeight:
      let thing = env.grid[x][y]
      if thing != nil:
        case thing.kind
        of Agent:
          drawAgent(thing, thing == selection)
        of Wall:
          # Draw wall
          bxy.drawRect(
            rect = rect(thing.pos.x.float32 - 0.5, thing.pos.y.float32 - 0.5, 1.0, 1.0),
            color = color(0.3, 0.3, 0.3, 1.0)
          )
        of Mine:
          # Draw mine (with resource indicator)
          bxy.drawRect(
            rect = rect(thing.pos.x.float32 - 0.4, thing.pos.y.float32 - 0.4, 0.8, 0.8),
            color = color(0.6, 0.4, 0.2, 1.0)
          )
          if thing.resources > 0:
            bxy.drawRect(
              rect = rect(thing.pos.x.float32 - 0.2, thing.pos.y.float32 - 0.2, 0.4, 0.4),
              color = color(0.9, 0.7, 0.3, 1.0)
            )
        of Converter:
          # Draw converter
          bxy.drawRect(
            rect = rect(thing.pos.x.float32 - 0.35, thing.pos.y.float32 - 0.35, 0.7, 0.7),
            color = color(0.3, 0.5, 0.7, 1.0)
          )
        of Clippy:
          drawClippy(thing)
        of Altar:
          drawAltar(thing)
        of Armory, Forge, ClayOven, WeavingLoom:
          drawBuilding(thing)
        of Temple:
          # Draw temple (similar to altar but larger)
          bxy.drawRect(
            rect = rect(thing.pos.x.float32 - 0.45, thing.pos.y.float32 - 0.45, 0.9, 0.9),
            color = color(0.7, 0.6, 0.4, 1.0)
          )