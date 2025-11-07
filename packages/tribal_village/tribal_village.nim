import std/[os, strutils],
  boxy, windy, vmath,
  src/environment, src/controls, src/common, src/panels, src/renderer, src/external_actions

when not defined(emscripten):
  import opengl

window = newWindow("Tribal Village", ivec2(1280, 800))
makeContextCurrent(window)

when not defined(emscripten):
  loadExtensions()

bxy = newBoxy()
rootArea = Area(layout: Horizontal)
worldMapPanel = Panel(panelType: WorldMap, name: "World Map")

rootArea.areas.add(Area(layout: Horizontal))
rootArea.panels.add(worldMapPanel)

proc display() =
  # Handle mouse capture release
  if window.buttonReleased[MouseLeft]:
    common.mouseCaptured = false
    common.mouseCapturedPanel = nil
  
  if window.buttonPressed[KeySpace]:
    if play:
      play = false
    else:
      lastSimTime = nowSeconds()
      simStep()
  if window.buttonPressed[KeyMinus] or window.buttonPressed[KeyLeftBracket]:
    playSpeed *= 0.5
    playSpeed = clamp(playSpeed, 0.00001, 60.0)
    play = true
  if window.buttonPressed[KeyEqual] or window.buttonPressed[KeyRightBracket]:
    playSpeed *= 2
    playSpeed = clamp(playSpeed, 0.00001, 60.0)
    play = true

  if window.buttonPressed[KeyN]:
    dec settings.showObservations
  if window.buttonPressed[KeyM]:
    inc settings.showObservations
  settings.showObservations = clamp(settings.showObservations, -1, 23)

  let now = nowSeconds()
  while play and (lastSimTime + playSpeed < now):
    lastSimTime += playSpeed
    simStep()

  bxy.beginFrame(window.size)
    # Use full window minus footer for the world view; remove header/tabs/timeline
  rootArea.rect = IRect(x: 0, y: 0, w: window.size.x, h: window.size.y)
  rootArea.updatePanelsSizes()






  worldMapPanel.beginDraw()

  worldMapPanel.beginPanAndZoom()
  useSelections()
  agentControls()

  drawFloor()
  drawTerrain()
  drawWalls()
  drawObjects()
  drawAgentDecorations()
  if settings.showVisualRange:
    drawVisualRanges()
  if settings.showGrid:
    drawGrid()
  if settings.showFogOfWar:
    drawFogOfWar()
  drawSelection()

  worldMapPanel.endPanAndZoom()

  worldMapPanel.endDraw()



  bxy.endFrame()
  window.swapBuffers()
  inc frame


# Build the atlas with progress feedback and error handling.
echo "🎨 Loading tribal assets..."
var loadedCount = 0
var totalFiles = 0

# Count total PNG files first
for path in walkDirRec("data/"):
  if path.endsWith(".png"):
    inc totalFiles


for path in walkDirRec("data/"):
  if path.endsWith(".png"):
    inc loadedCount

    try:
      bxy.addImage(path.replace("data/", "").replace(".png", ""), readImage(path))
    except Exception as e:
      echo "⚠️  Skipping ", path, ": ", e.msg

# Check for command line arguments to determine controller type
var useExternalController = false
for i in 1..paramCount():
  let param = paramStr(i)
  if param == "--external-controller":
    useExternalController = true
    # Command line: Requested external controller mode

# Check environment variable for Python training control
let pythonControlMode = existsEnv("TRIBAL_PYTHON_CONTROL") or existsEnv("TRIBAL_EXTERNAL_CONTROL")

# Initialize controller - prioritize external control, then existing controller, then default to BuiltinAI
if useExternalController or pythonControlMode:
  initGlobalController(ExternalNN)
  if pythonControlMode:
    # Environment variable: Using external NN controller for Python training
    discard  # Python mode uses external controller
  else:
    # Command line: Using external NN controller
    discard
elif globalController != nil:
  # Keeping existing controller
  discard
else:
  # DEFAULT: Use built-in AI for standalone execution
  initGlobalController(BuiltinAI)

# Check if external controller is active and start playing if so
if isExternalControllerActive():
  play = true

when defined(emscripten):
  proc main() {.cdecl.} =
    display()
    pollEvents()
  window.run(main)
else:
  while not window.closeRequested:
    display()
    pollEvents()
