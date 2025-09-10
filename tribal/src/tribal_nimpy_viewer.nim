## Tribal Nimpy Visualization Module
## Provides direct Python callable functions for rendering the tribal environment

import nimpy
import tribal/[environment, renderer, common, panels, controls, ui]
import boxy, opengl, windy, vmath
import std/[os, strutils]

# Global visualization state
var
  window*: Window
  bxy*: Boxy
  initialized = false
  
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

proc initVisualization*(): bool {.exportpy.} =
  ## Initialize the visualization system - call once from Python
  if initialized:
    return true
    
  try:
    echo "🎨 Initializing tribal visualization..."
    
    # Create window
    window = newWindow("Tribal - Python Controlled", ivec2(1280, 800))
    makeContextCurrent(window)
    
    when not defined(emscripten):
      loadExtensions()
    
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
    
    echo "✅ Tribal visualization initialized"
    initialized = true
    return true
    
  except Exception as e:
    echo "❌ Failed to initialize visualization: ", e.msg
    return false

proc loadAssets*(): bool {.exportpy.} =
  ## Load all tribal assets - call once after initialization
  if not initialized:
    echo "❌ Must call initVisualization() first"
    return false
    
  try:
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
        if loadedCount mod 50 == 0:
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

proc renderFrame*(): bool {.exportpy.} =
  ## Render one frame - call this continuously from Python
  if not initialized:
    echo "❌ Must call initVisualization() first"
    return false
    
  if window.closeRequested:
    return false
    
  try:
    # Ultra-basic rendering without any OpenGL calls
    echo "🔧 DEBUG: renderFrame() entry point reached"
    
    # Skip OpenGL entirely for now to isolate the crash
    echo "🔧 DEBUG: Skipping all OpenGL calls to isolate crash point"
    echo "🔧 DEBUG: renderFrame() completed without OpenGL operations"
    return true
    
  except Exception as e:
    echo "❌ DEBUG: Even basic renderFrame() failed: ", e.msg
    return false
    
    # TODO: Add back full rendering once minimal version works
    # # Handle mouse capture release
    # if window.buttonReleased[MouseLeft]:
    #   mouseCaptured = false
    #   mouseCapturedPanel = nil
    # 
    # # Begin frame
    # bxy.beginFrame(window.size)
    # const RibbonHeight = 64
    # rootArea.rect = IRect(x: 0, y: RibbonHeight, w: window.size.x, h: window.size.y - RibbonHeight*3)
    # rootArea.updatePanelsSizes()
    # globalHeaderPanel.rect = IRect(x: 0, y: 0, w: window.size.x, h: RibbonHeight)
    # globalFooterPanel.rect = IRect(x: 0, y: window.size.y - RibbonHeight, w: window.size.x, h: RibbonHeight)
    # globalTimelinePanel.rect = IRect(x: 0, y: window.size.y - RibbonHeight*2, w: window.size.x, h: RibbonHeight)
    # 
    # # Draw world map
    # worldMapPanel.beginDraw()
    # worldMapPanel.beginPanAndZoom()
    # useSelections()
    # agentControls()
    # 
    # drawFloor()
    # drawTerrain()
    # drawWalls()
    # drawObjects()
    # drawActions()
    # drawObservations()
    # drawAgentDecorations()
    # if settings.showVisualRange:
    #   drawVisualRanges()
    # if settings.showGrid:
    #   drawGrid()
    # if settings.showFogOfWar:
    #   drawFogOfWar()
    # drawSelection()
    # 
    # worldMapPanel.endPanAndZoom()
    # drawInfoText()
    # worldMapPanel.endDraw()
    # 
    # # Draw UI panels
    # globalHeaderPanel.beginDraw()
    # drawHeader(globalHeaderPanel)
    # globalHeaderPanel.endDraw()
    # 
    # globalFooterPanel.beginDraw()
    # drawFooter(globalFooterPanel)
    # globalFooterPanel.endDraw()
    # 
    # globalTimelinePanel.beginDraw()
    # drawTimeline(globalTimelinePanel)
    # globalTimelinePanel.endDraw()
    # 
    # rootArea.drawFrame()
    # 
    # # End frame
    # bxy.endFrame()
    # window.swapBuffers()
    # 
    # return true
    
  except Exception as e:
    echo "❌ Error rendering frame: ", e.msg
    return false

proc closeVisualization*() {.exportpy.} =
  ## Clean up visualization resources
  if initialized:
    echo "🧹 Closing tribal visualization"
    initialized = false

proc isWindowOpen*(): bool {.exportpy.} =
  ## Check if window is still open
  return initialized and not window.closeRequested