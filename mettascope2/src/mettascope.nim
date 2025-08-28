import std/[random, os, times, strformat, strutils],
  boxy, opengl, windy, windy/http, chroma, vmath,
  mettascope/[replays, common, panels, utils, header, footer, timeline,
  worldmap, minimap, agenttable, agenttraces, envconfig]

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
envConfigPanel = Panel(panelType: EnvConfig, name: "Env Config")
globalTimelinePanel = Panel(panelType: GlobalTimeline)
globalFooterPanel = Panel(panelType: GlobalFooter)
globalHeaderPanel = Panel(panelType: GlobalHeader)

rootArea.areas.add(Area(layout: Horizontal))
let topArea = Area(layout: Horizontal)
rootArea.areas.add(topArea)
let bottomArea = Area(layout: Horizontal)
rootArea.areas.add(bottomArea)

topArea.panels.add(worldMapPanel)
topArea.panels.add(minimapPanel)
topArea.panels.add(agentTablePanel)
bottomArea.panels.add(agentTracesPanel)
bottomArea.panels.add(envConfigPanel)

proc display() =
  let now = epochTime()

  bxy.beginFrame(window.size)
  const RibbonHeight = 64
  rootArea.rect = IRect(x: 0, y: RibbonHeight, w: window.size.x, h: window.size.y - RibbonHeight*3)
  topArea.rect = IRect(x: 0, y: rootArea.rect.y, w: rootArea.rect.w, h: rootArea.rect.h div 2)
  bottomArea.rect = IRect(x: 0, y: rootArea.rect.y + rootArea.rect.h div 2, w: rootArea.rect.w, h: rootArea.rect.h div 2)
  rootArea.updatePanelsSizes()

  globalHeaderPanel.rect = IRect(x: 0, y: 0, w: window.size.x, h: RibbonHeight)
  globalFooterPanel.rect = IRect(x: 0, y: window.size.y - RibbonHeight, w: window.size.x, h: RibbonHeight)
  globalTimelinePanel.rect = IRect(x: 0, y: window.size.y - RibbonHeight*2, w: window.size.x, h: RibbonHeight)




  globalHeaderPanel.beginDraw()
  drawHeader(globalHeaderPanel)
  globalHeaderPanel.endDraw()

  globalFooterPanel.beginDraw()
  drawFooter(globalFooterPanel)
  globalFooterPanel.endDraw()

  globalTimelinePanel.beginDraw()
  drawTimeline(globalTimelinePanel)
  globalTimelinePanel.endDraw()

  worldMapPanel.beginDraw()
  drawWorldMap(worldMapPanel)
  worldMapPanel.endDraw()

  minimapPanel.beginDraw()
  drawMinimap(minimapPanel)
  minimapPanel.endDraw()

  agentTablePanel.beginDraw()
  drawAgentTable(agentTablePanel)
  agentTablePanel.endDraw()

  agentTracesPanel.beginDraw()
  drawAgentTraces(agentTracesPanel)
  agentTracesPanel.endDraw()

  envConfigPanel.beginDraw()
  drawEnvConfig(envConfigPanel)
  envConfigPanel.endDraw()


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
        let req = startHttpRequest(replay)
        req.onError = proc(msg: string) =
          echo "onError: " & msg
        req.onResponse = proc(response: HttpResponse) =
          echo "onResponse: code=", $response.code, ", len=", response.body.len
          common.replay = loadReplay(response.body, replay)
      else:
        common.replay = loadReplay(replay)
    else:
      common.replay = loadReplay("replays/pens.json.z")

    while not window.closeRequested:
      display()
      pollEvents()

  dispatch(cmd)
