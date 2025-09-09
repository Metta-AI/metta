import std/[os, times, strutils],
  boxy, opengl, windy, vmath,
  tribal/[environment, controls, common, panels, renderer, ui, external_actions]

window = newWindow("MettaScope in Nim", ivec2(1280, 800))
makeContextCurrent(window)

when not defined(emscripten):
  loadExtensions()

bxy = newBoxy()
rootArea = Area(layout: Horizontal)
worldMapPanel = Panel(panelType: WorldMap, name: "World Map")
minimapPanel = Panel(panelType: Minimap, name: "Minimap")
agentTablePanel = Panel(panelType: AgentTable, name: "Agent Table")
agentTracesPanel = Panel(panelType: AgentTraces, name: "Agent Traces")
globalTimelinePanel = Panel(panelType: GlobalTimeline)
globalFooterPanel = Panel(panelType: GlobalFooter)
globalHeaderPanel = Panel(panelType: GlobalHeader)

rootArea.areas.add(Area(layout: Horizontal))
rootArea.panels.add(worldMapPanel)
rootArea.panels.add(minimapPanel)
rootArea.panels.add(agentTablePanel)
rootArea.panels.add(agentTracesPanel)

proc display() =
  # Handle mouse capture release
  if window.buttonReleased[MouseLeft]:
    mouseCaptured = false
    mouseCapturedPanel = nil
  
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
  globalHeaderPanel.rect = IRect(x: 0, y: 0, w: window.size.x, h: RibbonHeight)
  globalFooterPanel.rect = IRect(x: 0, y: window.size.y - RibbonHeight, w: window.size.x, h: RibbonHeight)
  globalTimelinePanel.rect = IRect(x: 0, y: window.size.y - RibbonHeight*2, w: window.size.x, h: RibbonHeight)






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


  globalHeaderPanel.beginDraw()
  drawHeader(globalHeaderPanel)
  globalHeaderPanel.endDraw()

  globalFooterPanel.beginDraw()
  drawFooter(globalFooterPanel)
  globalFooterPanel.endDraw()

  globalTimelinePanel.beginDraw()
  drawTimeline(globalTimelinePanel)
  globalTimelinePanel.endDraw()

  rootArea.drawFrame()

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

# Initialize controller to use built-in AI by default
initGlobalController(BuiltinAI)
echo "🤖 Initialized with built-in AI controller"

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
