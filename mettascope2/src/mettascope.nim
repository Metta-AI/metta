import std/[random, os, times, strformat, strutils],
  boxy, opengl, windy, chroma, vmath,
  mettascope/[actions, replays, common, panels, utils, minimap, worldmap,
      header, footer, timeline]

window = newWindow("MettaScope in Nim", ivec2(1280, 800))
makeContextCurrent(window)

when not defined(emscripten):
  loadExtensions()

bxy = newBoxy(quadsPerBatch = 10921)

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
  playControls()
  if worldMapPanel.zoom < 3:
    drawMiniMap()
  else:
    drawWorldMap()
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


# Build the atlas.
for path in walkDirRec("data/"):
  if path.endsWith(".png"):
    echo path
    bxy.addImage(path.replace("data/", "").replace(".png", ""), readImage(path))

when defined(emscripten):
  common.replay = loadReplay("replays/pens.json.z")
  proc main() {.cdecl.} =
    display()
    pollEvents()
  window.run(main)

else:
  import cligen, puppy
  proc cmd(replay: string = "") =
    if replay != "":
      if replay.startsWith("http"):
        let data = puppy.fetch(replay)
        common.replay = loadReplay(data, replay)
      else:
        common.replay = loadReplay(replay)
    else:
      common.replay = loadReplay("replays/pens.json.z")

    while not window.closeRequested:
      display()
      pollEvents()

  dispatch(cmd)
