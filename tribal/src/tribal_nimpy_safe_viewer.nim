## Safe Tribal Nimpy Visualization Module
## Provides gradual OpenGL initialization to isolate crash points

import nimpy
import tribal/[environment, renderer, common, panels, controls, ui]
import boxy, opengl, windy, vmath
import std/[os, strutils]

# Global visualization state
var
  window*: Window
  bxy*: Boxy
  initialized = false
  openglReady = false
  
# Panel globals
var
  rootArea*: Area
  worldMapPanel*: Panel
  minimapPanel*: Panel  
  agentTablePanel*: Panel
  agentTracesPanel*: Panel
  globalTimelinePanel*: Panel
  globalFooterPanel*: Panel
  globalHeaderPanel*: Panel

proc initWindowOnly*(): bool {.exportpy.} =
  ## Initialize only the window - no OpenGL context yet
  if initialized:
    return true
    
  try:
    echo "🔧 Step 1: Creating window without OpenGL..."
    window = newWindow("Tribal - Python Controlled", ivec2(1280, 800))
    echo "✅ Window created successfully"
    initialized = true
    return true
    
  except Exception as e:
    echo "❌ Failed to create window: ", e.msg
    return false

proc initOpenGLContext*(): bool {.exportpy.} =
  ## Initialize OpenGL context separately 
  if not initialized:
    echo "❌ Must call initWindowOnly() first"
    return false
    
  if openglReady:
    return true
    
  try:
    echo "🔧 Step 2: Making OpenGL context current..."
    makeContextCurrent(window)
    echo "✅ OpenGL context created"
    
    when not defined(emscripten):
      echo "🔧 Step 3: Loading OpenGL extensions..."
      loadExtensions()
      echo "✅ OpenGL extensions loaded"
    
    openglReady = true
    return true
    
  except Exception as e:
    echo "❌ Failed to initialize OpenGL: ", e.msg
    return false

proc initPanels*(): bool {.exportpy.} =
  ## Initialize panels and UI components
  if not openglReady:
    echo "❌ Must initialize OpenGL context first"
    return false
    
  try:
    echo "🔧 Step 4: Initializing Boxy and panels..."
    
    # Initialize boxy and panels
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
    
    echo "✅ Panels initialized"
    return true
    
  except Exception as e:
    echo "❌ Failed to initialize panels: ", e.msg
    return false

proc initVisualizationSafe*(): bool {.exportpy.} =
  ## Complete safe initialization - combines all steps
  echo "🎨 Starting safe tribal visualization initialization..."
  
  if not initWindowOnly():
    return false
    
  if not initOpenGLContext():
    return false
    
  if not initPanels():
    return false
    
  echo "✅ Safe tribal visualization fully initialized"
  return true

proc loadAssetsSafe*(): bool {.exportpy.} =
  ## Load all tribal assets with error handling
  if not openglReady:
    echo "❌ Must initialize OpenGL first"
    return false
    
  try:
    echo "🎨 Loading tribal assets safely..."
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
        if loadedCount mod 20 == 0:
          echo "Loading ", loadedCount, "/", totalFiles, ": ", path
        
        try:
          bxy.addImage(path.replace("data/", "").replace(".png", ""), readImage(path))
        except Exception as e:
          echo "⚠️  Skipping ", path, ": ", e.msg
    
    echo "🎨 Asset loading complete! Loaded ", loadedCount, "/", totalFiles, " files"
    return true
    
  except Exception as e:
    echo "❌ Failed to load assets: ", e.msg
    return false

proc renderFrameMinimal*(): bool {.exportpy.} =
  ## Render frame with minimal OpenGL operations
  if not openglReady:
    echo "❌ OpenGL not ready for rendering"
    return false
    
  if window.closeRequested:
    return false
    
  try:
    echo "🔧 DEBUG: Minimal render frame starting..."
    
    # Try the absolute minimum OpenGL operations
    bxy.beginFrame(window.size)
    echo "🔧 DEBUG: beginFrame() succeeded"
    
    # Skip all drawing operations for now
    echo "🔧 DEBUG: Skipping drawing operations"
    
    bxy.endFrame()
    echo "🔧 DEBUG: endFrame() succeeded"
    
    window.swapBuffers()
    echo "🔧 DEBUG: swapBuffers() succeeded"
    
    echo "✅ Minimal render completed successfully"
    return true
    
  except Exception as e:
    echo "❌ Minimal render failed: ", e.msg
    return false

proc renderFrameBasic*(): bool {.exportpy.} =
  ## Render with basic drawing operations
  if not openglReady:
    return false
    
  if window.closeRequested:
    return false
    
  try:
    # Handle mouse capture release
    if window.buttonReleased[MouseLeft]:
      mouseCaptured = false
      mouseCapturedPanel = nil
    
    # Begin frame
    bxy.beginFrame(window.size)
    const RibbonHeight = 64
    rootArea.rect = IRect(x: 0, y: RibbonHeight, w: window.size.x, h: window.size.y - RibbonHeight*3)
    rootArea.updatePanelsSizes()
    globalHeaderPanel.rect = IRect(x: 0, y: 0, w: window.size.x, h: RibbonHeight)
    globalFooterPanel.rect = IRect(x: 0, y: window.size.y - RibbonHeight, w: window.size.x, h: RibbonHeight)
    globalTimelinePanel.rect = IRect(x: 0, y: window.size.y - RibbonHeight*2, w: window.size.x, h: RibbonHeight)
    
    # Draw world map
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
    
    # Draw UI panels
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
    
    # End frame
    bxy.endFrame()
    window.swapBuffers()
    
    return true
    
  except Exception as e:
    echo "❌ Basic render failed: ", e.msg
    return false

proc closeVisualization*() {.exportpy.} =
  ## Clean up visualization resources
  if initialized:
    echo "🧹 Closing tribal visualization"
    initialized = false
    openglReady = false

proc isWindowOpen*(): bool {.exportpy.} =
  ## Check if window is still open
  return initialized and not window.closeRequested

proc getInitializationStatus*(): string {.exportpy.} =
  ## Get current initialization status for debugging
  if not initialized:
    return "not_initialized"
  elif not openglReady:
    return "window_only"
  else:
    return "fully_ready"