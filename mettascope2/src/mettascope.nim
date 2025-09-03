import std/[random, os, times, strformat, strutils],
  boxy, opengl, windy, windy/httpchroma, vmath, fidget2, puppy, fidget2/hybridrender,
  mettascope/[actions, replays, common, panels, utils, worldmap, header, footer, timeline]

var replay = ""

find "/UI/Main":
  onShow:

    # Build the atlas.
    for path in walkDirRec("data/"):
      if path.endsWith(".png"):
        echo path
        bxy.addImage(path.replace("data/", "").replace(".png", ""), readImage(path))

    echo "onShow"

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


    let worldMap = find "**/WorldMap"
    worldMap.onRenderCallback = proc(thisNode: Node) =
      echo "onRender WorldMap"
      bxy.drawImage("meta_grid_icon", vec2(100, 100))

      worldMapPanel.beginPanAndZoom()

      useSelections()
      agentControls()
      playControls()

      drawFloor()
      drawWalls()
      drawObjects()
      # drawActions()
      # drawAgentDecorations()

      if settings.showGrid:
        drawGrid()
      if settings.showVisualRange:
        drawVisualRanges()

      drawSelection()
      drawInventory()

      if settings.showFogOfWar:
        drawFogOfWar()

      worldMapPanel.endPanAndZoom()


startFidget(
  figmaUrl = "https://www.figma.com/design/hHmLTy7slXTOej6opPqWpz/MetaScope-V2-Rig",
  windowTitle = "MetaScope V2",
  entryFrame = "UI/Main",
  windowStyle = DecoratedResizable
)
