import std/[random, os, times, strformat, strutils, sequtils]
import boxy, opengl, windy, chroma, vmath, pixie
import tribal/[tribal_game, worldmap, controller, ui]

# Global variables
type
  IRect* = object
    x*, y*, w*, h*: int
  
  WorldMapPanel = ref object
    rect*: IRect
    pos*: Vec2
    vel*: Vec2
    zoom*: float32
    zoomVel*: float32

var
  window*: Window
  bxy*: Boxy
  env*: Environment
  selection*: Thing
  worldMapPanel*: WorldMapPanel
  typeface*: Typeface

const
  BgColor = parseHtmlColor("#273646")
  HeaderHeight = 64
  FooterHeight = 64

var
  actionsArray*: array[MapAgents, array[2, uint8]]
  # Controller will use a random seed each time
  agentController* = newController(seed = int(epochTime() * 1000))
  # UI state
  play* = false
  playSpeed* = 0.01
  lastSimTime* = 0.0
  settings* = (showGrid: false, showObservations: -1)

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

proc drawText*(
  text: string,
  pos: Vec2,
  size: float32,
  color: Color
) =
  ## Draw text on the screen.
  var font = newFont(typeface)
  font.size = size
  font.paint = color
  let
    arrangement = typeset(@[newSpan(text, font)], bounds = vec2(1280, 800))
    transform = translate(pos)
    globalBounds = arrangement.computeBounds(transform).snapToPixels()
    textImage = newImage(globalBounds.w.int, globalBounds.h.int)
    imageSpace = translate(-globalBounds.xy) * transform
  textImage.fillText(arrangement, imageSpace)
  
  let imageKey = &"text_{text}_{size}"
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

proc boxyMouse*(): Vec2 =
  return inverse(bxy.getTransform()) * window.mousePos.vec2

proc beginPanAndZoom*() =
  ## Pan and zoom the map.
  if window.buttonDown[MouseLeft] or window.buttonDown[MouseMiddle]:
    worldMapPanel.vel = window.mouseDelta.vec2
  else:
    worldMapPanel.vel *= 0.9

  worldMapPanel.pos += worldMapPanel.vel

  if window.scrollDelta.y != 0:
    worldMapPanel.zoomVel = window.scrollDelta.y * 0.03
  else:
    worldMapPanel.zoomVel *= 0.9

  bxy.saveTransform()

  let oldMat = translate(vec2(worldMapPanel.pos.x, worldMapPanel.pos.y)) * scale(vec2(worldMapPanel.zoom*worldMapPanel.zoom, worldMapPanel.zoom*worldMapPanel.zoom))
  worldMapPanel.zoom += worldMapPanel.zoomVel
  worldMapPanel.zoom = clamp(worldMapPanel.zoom, 0.3, 100)
  let newMat = translate(vec2(worldMapPanel.pos.x, worldMapPanel.pos.y)) * scale(vec2(worldMapPanel.zoom*worldMapPanel.zoom, worldMapPanel.zoom*worldMapPanel.zoom))
  let newAt = newMat.inverse() * window.mousePos.vec2
  let oldAt = oldMat.inverse() * window.mousePos.vec2
  worldMapPanel.pos -= (oldAt - newAt).xy * (worldMapPanel.zoom*worldMapPanel.zoom)

  bxy.translate(worldMapPanel.pos)
  bxy.scale(vec2(worldMapPanel.zoom*worldMapPanel.zoom, worldMapPanel.zoom*worldMapPanel.zoom))

proc endPanAndZoom*() =
  bxy.restoreTransform()

proc drawStats*() =
  ## Draw basic stats in the corner
  let statsText = &"""Step: {env.currentStep}
Agents: {env.agents.len}"""
  
  drawText(statsText, vec2(10, 10), 14, color(1, 1, 1, 0.8))

proc drawHeader*(bxy: Boxy, window: Window, typeface: Typeface, width: float32) =
  ## Draw simple header bar
  bxy.drawRect(
    rect(0, 0, width, HeaderHeight.float32),
    color(0.16, 0.21, 0.27, 1.0)  # Dark header color
  )
  
  # Draw title
  drawText("Tribal Grid", vec2(20, 20), 24, color(1, 1, 1, 0.9))
  
  # Draw grid toggle button
  if drawIconButton(
    if settings.showGrid: "ui/grid" else: "ui/grid",
    pos = vec2(width - 50, 16)
  ):
    settings.showGrid = not settings.showGrid

proc drawFooter*(bxy: Boxy, window: Window, width: float32, simStepProc: proc()) =
  ## Draw simple footer with play controls
  bxy.drawRect(
    rect(0, 0, width, FooterHeight.float32),
    color(0.18, 0.20, 0.24, 1.0)  # Slightly lighter than header
  )
  
  var x = 20.0
  
  # Play/pause button
  if drawIconButton(
    if play: "ui/pause" else: "ui/play",
    pos = vec2(x, 16)
  ):
    play = not play
    lastSimTime = epochTime()
  x += 40
  
  # Step button (when paused)
  if not play:
    if drawIconButton(
      "ui/stepForward",
      pos = vec2(x, 16)
    ):
      simStepProc()
    x += 40
  
  # Speed controls
  let speedText = if play:
    &"Speed: {1.0 / playSpeed:0.1f}x"
  else:
    "Paused"
  drawText(speedText, vec2(x, 24), 16, color(1, 1, 1, 0.8))

proc main() =
  # Initialize window
  window = newWindow("Tribal Grid", ivec2(1280, 800))
  makeContextCurrent(window)
  
  when not defined(emscripten):
    loadExtensions()
  
  # Initialize graphics
  bxy = newBoxy()
  typeface = readTypeface("data/fonts/Inter-Regular.ttf")
  
  # Set up UI module globals
  ui.window = window
  ui.bxy = bxy
  
  # Initialize game
  env = newEnvironment()
  echo "Environment created with ", env.agents.len, " agents"
  worldMapPanel = WorldMapPanel(
    rect: IRect(x: 0, y: 0, w: 1280, h: 800),
    pos: vec2(640, 400),  # Center the view
    vel: vec2(0, 0),
    zoom: 5,  # Start with less zoom
    zoomVel: 0
  )
  
  # Load all sprites
  for path in walkDirRec("data/"):
    if path.endsWith(".png"):
      echo "Loading sprite: ", path
      bxy.addImage(path.replace("data/", "").replace(".png", ""), readImage(path))
  
  # Main loop
  while not window.closeRequested:
    # Poll events
    pollEvents()
    
    # Handle keyboard shortcuts
    if window.buttonPressed[KeySpace]:
      play = not play
    if window.buttonPressed[KeyMinus]:
      playSpeed *= 2.0
      playSpeed = clamp(playSpeed, 0.00001, 1.0)
      play = true
      echo "Speed: ", 1.0 / playSpeed, "x"
    if window.buttonPressed[KeyEqual]:
      playSpeed *= 0.5
      playSpeed = clamp(playSpeed, 0.00001, 1.0) 
      play = true
      echo "Speed: ", 1.0 / playSpeed, "x"
    
    # Handle observation controls
    if window.buttonPressed[KeyN]:
      dec settings.showObservations
      echo "showObservations: ", settings.showObservations
    if window.buttonPressed[KeyM]:
      inc settings.showObservations
      echo "showObservations: ", settings.showObservations
    settings.showObservations = clamp(settings.showObservations, -1, 23)
    
    # Auto-step simulation based on play speed
    let now = epochTime()
    while play and (lastSimTime + playSpeed < now):
      lastSimTime += playSpeed
      simStep()
    if window.buttonPressed[KeySpace]:
      lastSimTime = now
      if not play:  # If paused, step once
        simStep()
    
    # Handle controls
    agentControls()
    
    # Clear screen
    bxy.beginFrame(window.size)
    bxy.drawRect(rect(0, 0, window.size.x.float32, window.size.y.float32), BgColor)
    
    # Calculate main view area (between header and footer)
    let mainAreaY = HeaderHeight.float32
    let mainAreaHeight = window.size.y.float32 - HeaderHeight - FooterHeight
    
    # Save transform and clip to main area
    bxy.saveTransform()
    bxy.translate(vec2(0, mainAreaY))
    
    # Draw world with pan/zoom in the main area
    beginPanAndZoom()
    
    # Handle mouse selection
    useSelections(window, bxy, env, selection)
    
    draw(bxy, env, selection)
    
    # Draw grid overlay if enabled
    if settings.showGrid:
      drawGrid()
    
    endPanAndZoom()
    
    # Draw UI overlay
    drawStats()
    
    # Restore transform for header/footer
    bxy.restoreTransform()
    
    # Draw header at top
    bxy.saveTransform()
    drawHeader(bxy, window, typeface, window.size.x.float32)
    bxy.restoreTransform()
    
    # Draw footer at bottom
    bxy.saveTransform()
    bxy.translate(vec2(0, window.size.y.float32 - FooterHeight))
    drawFooter(bxy, window, window.size.x.float32, simStep)
    bxy.restoreTransform()
    
    # End frame
    bxy.endFrame()
    window.swapBuffers()

when isMainModule:
  main()