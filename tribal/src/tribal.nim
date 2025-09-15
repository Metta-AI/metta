import std/[os, times, strutils],
  boxy, opengl, windy, vmath,
  tribal/environment, tribal/controls, tribal/common, tribal/panels, tribal/renderer, tribal/ui, tribal/external_actions

window = newWindow("Tribal", ivec2(1280, 800))
makeContextCurrent(window)

when not defined(emscripten):
  loadExtensions()

bxy = newBoxy()
rootArea = Area(layout: Horizontal)
worldMapPanel = Panel(panelType: WorldMap, name: "World Map")
globalFooterPanel = Panel(panelType: GlobalFooter)

rootArea.areas.add(Area(layout: Horizontal))
rootArea.panels.add(worldMapPanel)

proc display() =
  # Handle mouse capture release
  if window.buttonReleased[MouseLeft]:
    common.mouseCaptured = false
    common.mouseCapturedPanel = nil
  
  if window.buttonPressed[KeySpace]:
    play = false
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
  # Use full window minus footer for the world view; remove header/tabs/timeline
  rootArea.rect = IRect(x: 0, y: 0, w: window.size.x, h: window.size.y - RibbonHeight)
  rootArea.updatePanelsSizes()
  globalFooterPanel.rect = IRect(x: 0, y: window.size.y - RibbonHeight, w: window.size.x, h: RibbonHeight)






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


  globalFooterPanel.beginDraw()
  drawFooter(globalFooterPanel)
  globalFooterPanel.endDraw()


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

echo "📁 Found ", totalFiles, " PNG files to load"

for path in walkDirRec("data/"):
  if path.endsWith(".png"):
    inc loadedCount
    echo "Loading ", loadedCount, "/", totalFiles, ": ", path
    
    try:
      bxy.addImage(path.replace("data/", "").replace(".png", ""), readImage(path))
    except Exception as e:
      echo "⚠️  Skipping ", path, ": ", e.msg
    
    # Show progress every 50 files
    if loadedCount mod 50 == 0:
      echo "✅ Loaded ", loadedCount, "/", totalFiles, " assets..."

echo "🎨 Asset loading complete! Loaded ", loadedCount, "/", totalFiles, " files"

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
  echo "🤖 Standalone mode: Initialized BuiltinAI controller - agents will move autonomously"
  echo "   Use TRIBAL_PYTHON_CONTROL=1 environment variable to disable for Python training"

# Check if external controller is active and start playing if so
if isExternalControllerActive():
  play = true
  echo "🎮 External controller detected - starting automatic play mode"

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
