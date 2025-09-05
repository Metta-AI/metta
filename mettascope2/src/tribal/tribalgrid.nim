import std/[random, os, times, strformat, strutils],
  boxy, opengl, windy, chroma, vmath,
  tribal_game, worldmap,
  vmath, bumpy, windy, boxy, chroma, tribal_core 

var
  typeface* = readTypeface("data/fonts/Inter-Regular.ttf")

type Panel* = ref object
  rect*: IRect

  pos*: Vec2
  vel*: Vec2
  zoom*: float32 = 10
  zoomVel*: float32

window = newWindow("MettaScope in Nim", ivec2(1280, 800))
makeContextCurrent(window)

when not defined(emscripten):
  loadExtensions()

bxy = newBoxy()
env = newEnvironment()
worldMapPanel = Panel()

const
  BgColor = parseHtmlColor("#273646")
  FootBgColor = parseHtmlColor("#2D343D")
  HeaderSize = 30

proc drawIconButton*(
  image: string,
  pos: Vec2,
  size = vec2(32, 32),
): bool =
  let box = Rect(
    x: pos.x,
    y: pos.y,
    w: size.x,
    h: size.y
  )

  if window.boxyMouse.vec2.overlaps(box):
    if window.buttonPressed[MouseLeft]:
      result = true
    bxy.drawRect(
      rect = box,
      color = color(1, 1, 1, 0.5)
    )

  bxy.drawImage(
    image,
    pos = pos
  )

proc drawIconToggle*(
  image: string,
  pos: Vec2,
  size = vec2(32, 32),
  value: var bool
): bool =
  let box = Rect(
    x: pos.x,
    y: pos.y,
    w: size.x,
    h: size.y
  )

  if window.boxyMouse.vec2.overlaps(box):
    if window.buttonPressed[MouseLeft]:
      value = not value
      result = true
    bxy.drawRect(
      rect = box,
      color = color(1, 1, 1, 0.5)
    )

  var alpha = 0.4
  if value:
    alpha = 1.0

  bxy.drawImage(
    image,
    pos = pos,
    tint = color(1, 1, 1, alpha)
  )

var
  actionsArray*: array[MapAgents, array[2, uint8]]
  # Controller will use a random seed each time
  agentController* = newController(seed = int(epochTime() * 1000))

proc simStep*() =
  # Use controller for agent actions
  for j, agent in env.agents:
    if selection != agent:
      # Use the controller to decide actions
      actionsArray[j] = agentController.decideAction(env, j)
    # else: selected agent uses manual controls
  
  # Step the environment (this handles mines, clippys, etc.)
  env.step(addr actionsArray)
  
  # Update controller state
  agentController.updateController()

proc agentControls*() =
  ## Controls for the selected agent.
  if selection != nil and selection.kind == Agent:
    let agent = selection

    # Direct movement with auto-rotation
    if window.buttonPressed[KeyW] or window.buttonPressed[KeyUp]:
      # Move North
      actionsArray[agent.agentId] = [1, 0]
      simStep()
    elif window.buttonPressed[KeyS] or window.buttonPressed[KeyDown]:
      # Move South
      actionsArray[agent.agentId] = [1, 1]
      simStep()
    elif window.buttonPressed[KeyD] or window.buttonPressed[KeyRight]:
      # Move East
      actionsArray[agent.agentId] = [1, 2]
      simStep()
    elif window.buttonPressed[KeyA] or window.buttonPressed[KeyLeft]:
      # Move West
      actionsArray[agent.agentId] = [1, 3]
      simStep()

    # Use - face current direction of agent
    if window.buttonPressed[KeyU]:
      # Use in the direction the agent is facing
      let useDir = agent.orientation.uint8
      actionsArray[agent.agentId] = [3, useDir]
      simStep()

    # Swap (still valid - swaps positions with frozen agents)
    if window.buttonPressed[KeyP]:
      actionsArray[agent.agentId] = [8, 0]
      simStep()

var
  actionsArray*: array[MapAgents, array[2, uint8]]
  # Controller will use a random seed each time
  agentController* = newController(seed = int(epochTime() * 1000))

proc drawText*(
  bxy: Boxy,
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
    arrangement = typeset(@[newSpan(text, font)], bounds = vec2(1280, 800))
    globalBounds = arrangement.computeBounds(transform).snapToPixels()
    textImage = newImage(globalBounds.w.int, globalBounds.h.int)
    imageSpace = translate(-globalBounds.xy) * transform
  textImage.fillText(arrangement, imageSpace)

  bxy.addImage(imageKey, textImage)
  bxy.drawImage(imageKey, globalBounds.xy)

proc measureText*(
  text: string,
  size: float32
): Vec2 =
  var font = newFont(typeface)
  font.size = size
  let arrangement = typeset(@[newSpan(text, font)], bounds = vec2(1280, 800))
  let transform = translate(vec2(0, 0))
  let bounds = arrangement.computeBounds(transform).snapToPixels()
  return vec2(bounds.w, bounds.h)


proc drawBubbleLine*(bxy: Boxy, start: Vec2, stop: Vec2, color: Color) =
  ## Draw a line with circles.
  let
    dir = (stop - start).normalize
  for i in 0 ..< int(dist(start, stop) / 5):
    let pos = start + dir * i.float32 * 5
    # bxy.drawImage(
    #   "bubble",
    #   pos,
    #   angle = 0,
    #   scale = 0.25,
    #   tint = color
    # )

proc boxyMouse*(window: Window): Vec2 =
  return inverse(bxy.getTransform()) * window.mousePos.vec2


proc simStep*() =
  # Use controller for agent actions
  for j, agent in env.agents:
    if selection != agent:
      # Use the controller to decide actions
      actionsArray[j] = agentController.decideAction(env, j)
    # else: selected agent uses manual controls
  
  # Step the environment (this handles mines, clippys, etc.)
  env.step(addr actionsArray)
  
  # Update controller state
  agentController.updateController()

proc agentControls*() =
  ## Controls for the selected agent.
  if selection != nil and selection.kind == Agent:
    let agent = selection

    # Direct movement with auto-rotation
    if window.buttonPressed[KeyW] or window.buttonPressed[KeyUp]:
      # Move North
      actionsArray[agent.agentId] = [1, 0]
      simStep()
    elif window.buttonPressed[KeyS] or window.buttonPressed[KeyDown]:
      # Move South
      actionsArray[agent.agentId] = [1, 1]
      simStep()
    elif window.buttonPressed[KeyD] or window.buttonPressed[KeyRight]:
      # Move East
      actionsArray[agent.agentId] = [1, 2]
      simStep()
    elif window.buttonPressed[KeyA] or window.buttonPressed[KeyLeft]:
      # Move West
      actionsArray[agent.agentId] = [1, 3]
      simStep()

    # Use - face current direction of agent
    if window.buttonPressed[KeyU]:
      # Use in the direction the agent is facing
      let useDir = agent.orientation.uint8
      actionsArray[agent.agentId] = [3, useDir]
      simStep()

    # Swap (still valid - swaps positions with frozen agents)
    if window.buttonPressed[KeyP]:
      actionsArray[agent.agentId] = [8, 0]
      simStep()

proc rect*(rect: IRect): Rect =
  Rect(x: rect.x.float32, y: rect.y.float32, w: rect.w.float32, h: rect.h.float32)

proc beginPanAndZoom*(panel: Panel) =
  ## Pan and zoom the map.
  if window.buttonDown[MouseLeft] or window.buttonDown[MouseMiddle]:
    panel.vel = window.mouseDelta.vec2
  else:
    panel.vel *= 0.9

  panel.pos += panel.vel

  if window.scrollDelta.y != 0:
    panel.zoomVel = window.scrollDelta.y * 0.03
  else:
    panel.zoomVel *= 0.9

  bxy.saveTransform()

  let oldMat = translate(vec2(panel.pos.x, panel.pos.y)) * scale(vec2(panel.zoom*panel.zoom, panel.zoom*panel.zoom))
  panel.zoom += panel.zoomVel
  panel.zoom = clamp(panel.zoom, 0.3, 100)
  let newMat = translate(vec2(panel.pos.x, panel.pos.y)) * scale(vec2(panel.zoom*panel.zoom, panel.zoom*panel.zoom))
  let newAt = newMat.inverse() * window.mousePos.vec2
  let oldAt = oldMat.inverse() * window.mousePos.vec2
  panel.pos -= (oldAt - newAt).xy * (panel.zoom*panel.zoom)

  bxy.translate(panel.pos)
  bxy.scale(vec2(panel.zoom*panel.zoom, panel.zoom*panel.zoom))

proc endPanAndZoom*(panel: Panel) =
  bxy.restoreTransform()

proc beginDraw*(panel: Panel) =
  # bxy.pushLayer()
  bxy.saveTransform()

  bxy.translate(vec2(panel.rect.x.float32, panel.rect.y.float32))


proc endDraw*(panel: Panel) =

  bxy.restoreTransform()

  # # Draw the mask.
  # bxy.pushLayer()
  # bxy.drawRect(
  #   rect = panel.rect.rect,
  #   color = color(1, 0, 0, 1.0)
  # )
  # bxy.popLayer(blendMode = MaskBlend)

  # bxy.popLayer()

proc updatePanelsSizes*(area: Area) =
  # Update the sizes of the panels in the area and its subareas and subpanels.
  for num,panel in area.panels:
    if num == area.selectedPanelNum:
      panel.rect.x = area.rect.x
      panel.rect.y = area.rect.y + HeaderSize
      panel.rect.w = area.rect.w
      panel.rect.h = area.rect.h - HeaderSize
    else:
      panel.rect.x = 0
      panel.rect.y = 0
      panel.rect.w = 0
      panel.rect.h = 0

  for subarea in area.areas:
    updatePanelsSizes(subarea)

proc drawHeader*(rect: Rect) =
  bxy.drawRect(
    rect, BgColor)
  
  bxy.drawImage(
    "ui/logo",
    pos = vec2(0, 0),
  )
  bxy.drawImage(
    "ui/header-bg",
    pos = vec2(0, 0),
  )

  # Draw the title.
  bxy.drawText(
    "Mettascope Arena Basic",
    translate(vec2(64+16, 16)),
    typeface,
    "Mettascope Arena Basic",
    24,
    color(1, 1, 1, 1)
  )

  if drawIconButton(
    "ui/share",
    pos = vec2(panel.rect.w.float32 - (16 + 32)*1, 16)
  ):
    echo "Share"

  if drawIconButton(
    "ui/help",
    pos = vec2(panel.rect.w.float32 - (16 + 32)*2, 16)
  ):
    echo "Help"

proc drawFooter*(rect: Rect) =
  bxy.drawRect(
    rect, FootBgColor)

  # Draw the left side buttons.
  var x = 16f
  if drawIconButton(
    "ui/rewindToStart",
    pos = vec2(x, 16)
  ):
    echo "Rewind to start"
  x += 32 + 5

  if drawIconButton(
    "ui/stepBack",
    pos = vec2(x, 16)
  ):
    echo "Step back"
  x += 32 + 5

  if drawIconButton(
    if play: "ui/pause" else: "ui/play",
    pos = vec2(x, 16)
  ):
    play = not play
    echo if play: "Playing" else: "Paused"
  x += 32 + 5

  # if drawIconButton(
  #   "ui/pause",
  #   pos = vec2(x, 16)
  # ):
  #   echo "Pause"
  # x += 32 + 5

  if drawIconButton(
    "ui/stepForward",
    pos = vec2(x, 16)
  ):
    echo "Step forward"
    simStep()  # Step the simulation once
  x += 32 + 5

  if drawIconButton(
    "ui/rewindToEnd",
    pos = vec2(x, 16)
  ):
    echo "Rewind to end"


  # Draw the middle buttons.
  x = panel.rect.w.float32 / 2 - 32

  if drawIconButton(
    "ui/turtle",
    pos = vec2(x, 16)
  ):
    playSpeed = 0.5
    play = true
    echo "Speed: Slow (0.5x)"
  x += 32 + 3

  if drawIconButton(
    "ui/speed",
    pos = vec2(x, 16),
    size = vec2(20, 32)
  ):
    playSpeed = 0.25
    play = true
    echo "Speed: 1x"
  x += 20

  if drawIconButton(
    "ui/speed",
    pos = vec2(x, 16),
    size = vec2(20, 32)
  ):
    playSpeed = 0.125
    play = true
    echo "Speed: 2x"
  x += 20

  if drawIconButton(
    "ui/speed",
    pos = vec2(x, 16),
    size = vec2(20, 32)
  ):
    playSpeed = 0.0625
    play = true
    echo "Speed: 4x"
  x += 20

  if drawIconButton(
    "ui/speed",
    pos = vec2(x, 16),
    size = vec2(20, 32)
  ):
    playSpeed = 0.03125
    play = true
    echo "Speed: 8x"
  x += 20

  if drawIconButton(
    "ui/rabbit",
    pos = vec2(x, 16)
  ):
    playSpeed = 0.015625
    play = true
    echo "Speed: Fast (16x)"


  # Draw the right side buttons.
  x = panel.rect.w.float32 - 16 - 32

  if drawIconToggle(
    "ui/cloud",
    pos = vec2(x, 16),
    value = settings.showFogOfWar
  ):
    echo "Fog of war"
  x -= 32 + 5

  if drawIconToggle(
    "ui/eye",
    pos = vec2(x, 16),
    value = settings.showVisualRange
  ):
    echo "Visual range"
  x -= 32 + 5

  if drawIconToggle(
    "ui/grid",
    pos = vec2(x, 16),
    value = settings.showGrid
  ):
    echo "Grid"
  x -= 32 + 5

  if drawIconToggle(
    "ui/tack",
    pos = vec2(x, 16),
    value = settings.lockFocus
  ):
    echo "Focus"
  x -= 32 + 5

proc display() =
  if window.buttonPressed[KeySpace]:
    play = false
  if window.buttonPressed[KeyMinus]:
    playSpeed *= 0.5
    playSpeed = clamp(playSpeed, 0.00001, 60.0)
    play = true
  if window.buttonPressed[KeyEqual]:
    playSpeed *= 2
    playSpeed = clamp(playSpeed, 0.00001, 60.0)
    play = true

  if window.buttonPressed[KeyN]:
    dec settings.showObservations
    echo "showObservations: ", settings.showObservations
  if window.buttonPressed[KeyM]:
    inc settings.showObservations
    echo "showObservations: ", settings.showObservations
  settings.showObservations = clamp(settings.showObservations, -1, 23)

  let now = epochTime()
  while play and (lastSimTime + playSpeed < now):
    lastSimTime += playSpeed
    simStep()
  if window.buttonPressed[KeySpace]:
    lastSimTime = now
    simStep()

  bxy.beginFrame(window.size)
  const RibbonHeight = 64
  rootArea.rect = IRect(x: 0, y: RibbonHeight, w: window.size.x, h: window.size.y - RibbonHeight*3)
  rootArea.updatePanelsSizes()
  let headerRect = IRect(x: 0, y: 0, w: window.size.x, h: RibbonHeight)
  let footerRect = IRect(x: 0, y: window.size.y - RibbonHeight, w: window.size.x, h: RibbonHeight)






  worldMapPanel.beginDraw()

  worldMapPanel.beginPanAndZoom()
  useSelections()
  agentControls()

  drawFloor()
  drawTerrain()
  drawWalls()
  drawObjects()
  drawActions()
  drawObservations()
  drawAgentDecorations()
  if settings.showVisualRange:
    drawVisualRanges()
  if settings.showGrid:
    drawGrid()
  if settings.showFogOfWar:
    drawFogOfWar()
  drawSelection()

  worldMapPanel.endPanAndZoom()

  drawInfoText()

  worldMapPanel.endDraw()


  drawHeader(headerRect)

  drawFooter(footerRect)


  bxy.endFrame()
  window.swapBuffers()
  inc frame


# Build the atlas.
for path in walkDirRec("data/"):
  if path.endsWith(".png"):
    echo path
    bxy.addImage(path.replace("data/", "").replace(".png", ""), readImage(path))

when defined(emscripten):
  proc main() {.cdecl.} =
    echo "draw frame"
    display()
    pollEvents()
  window.run(main)
else:
  while not window.closeRequested:
    display()
    pollEvents()
