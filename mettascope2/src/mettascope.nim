import std/[random, os, times, strformat, strutils],
  boxy, opengl, windy, windy/http, chroma, vmath, fidget2, puppy, fidget2/hybridrender,
  mettascope/[replays, common, panels, utils, header, footer, timeline,
  worldmap, minimap, agenttable, agenttraces, envconfig]

var replay = ""

# TODO: Remove with dynamic panels.
var topArea: Area
var bottomArea: Area

var loaded = false

find "/UI/Main":

  find "**/PanelHeader":
    onClick:
      let title = thisNode.find("**/title")
      echo "Selected panel: ", title.text

  find "AreaHeader":
    onClick:
      echo "Clicked: AreaHeader: ", thisNode.name


  find "WorldMapPanel":
    onClick:
      echo "Clicked: WorldMapPanel: ", thisNode.name

  find "AgentTracesPanel":
    onClick:
      echo "Clicked: AgentTracesPanel: ", thisNode.name

  # onClick:
  #   echo "Clicked: ", thisNode.name

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

    echo "Creating panels"
    rootArea = Area(layout: Horizontal, node: find("**/AreaHeader"))

    worldMapPanel = Panel(panelType: WorldMap, name: "World Map", node: find("**/WorldMapPanel"))
    minimapPanel = Panel(panelType: Minimap, name: "Minimap", node: find("**/MinimapPanel"))
    # agentTablePanel = Panel(panelType: AgentTable, name: "Agent Table", node: find("**/AgentTablePanel"))
    agentTracesPanel = Panel(panelType: AgentTraces, name: "Agent Traces", node: find("**/AgentTracesPanel"))
    # envConfigPanel = Panel(panelType: EnvConfig, name: "Env Config", node: find("**/EnvConfigPanel"))
    # globalTimelinePanel = Panel(panelType: GlobalTimeline, node: find("GlobalTimeline"))
    # globalFooterPanel = Panel(panelType: GlobalFooter, node: find("GlobalFooter"))
    # globalHeaderPanel = Panel(panelType: GlobalHeader, node: find("GlobalHeader"))

    topArea = Area(layout: Horizontal, node: rootArea.node.copy())
    thisNode.addChild(topArea.node)
    # rootArea.areas.add(topArea)
    bottomArea = Area(layout: Horizontal, node: rootArea.node.copy())
    thisNode.addChild(bottomArea.node)
    # rootArea.areas.add(bottomArea)

    # topArea.panels.add(worldMapPanel)
    # topArea.panels.add(minimapPanel)
    # topArea.panels.add(agentTablePanel)
    # bottomArea.panels.add(agentTracesPanel)
    # bottomArea.panels.add(envConfigPanel)

    worldMapPanel.node.onRenderCallback = proc(thisNode: Node) =
      bxy.saveTransform()
      bxy.translate(thisNode.position)
      drawWorldMap(worldMapPanel)
      bxy.restoreTransform()

    minimapPanel.node.onRenderCallback = proc(thisNode: Node) =
      bxy.saveTransform()
      bxy.translate(thisNode.position)
      drawMinimap(minimapPanel)
      bxy.restoreTransform()

    agentTracesPanel.node.onRenderCallback = proc(thisNode: Node) =
      bxy.saveTransform()
      bxy.translate(thisNode.position)
      drawAgentTraces(agentTracesPanel)
      bxy.restoreTransform()

    loaded = true

  onFrame:
    if not loaded:
      return

    if window.buttonReleased[MouseLeft]:
      mouseCaptured = false
      mouseCapturedPanel = nil

    const RibbonHeight = 64
    const HeaderHeight = 28
    # rootArea.rect = IRect(x: 0, y: RibbonHeight, w: window.size.x, h: window.size.y - RibbonHeight*3)
    # topArea.rect = IRect(x: 0, y: rootArea.rect.y, w: rootArea.rect.w, h: (rootArea.rect.h.float32 * 0.75).int)
    # bottomArea.rect = IRect(x: 0, y: rootArea.rect.y + (rootArea.rect.h.float32 * 0.75).int, w: rootArea.rect.w, h: (rootArea.rect.h.float32 * 0.25).int)
    # rootArea.updatePanelsSizes()

    # globalHeaderPanel.rect = IRect(x: 0, y: 0, w: window.size.x, h: RibbonHeight)
    # globalFooterPanel.rect = IRect(x: 0, y: window.size.y - RibbonHeight, w: window.size.x, h: RibbonHeight)
    # globalTimelinePanel.rect = IRect(x: 0, y: window.size.y - RibbonHeight*2, w: window.size.x, h: RibbonHeight)

    worldMapPanel.rect = irect(
      0,
      RibbonHeight + HeaderHeight,
      window.size.x,
      ((window.size.y.float32 - RibbonHeight*3) * 0.75).int - HeaderHeight
    )
    worldMapPanel.node.position = worldMapPanel.rect.xy.vec2
    worldMapPanel.node.size = worldMapPanel.rect.wh.vec2

    agentTracesPanel.rect = irect(
      0,
      RibbonHeight + ((window.size.y.float32 - RibbonHeight*3) * 0.75).int + HeaderHeight,
      window.size.x,
      ((window.size.y.float32 - RibbonHeight*3) * 0.25).int - HeaderHeight
    )
    agentTracesPanel.node.position = agentTracesPanel.rect.xy.vec2
    agentTracesPanel.node.size = agentTracesPanel.rect.wh.vec2

    minimapPanel.rect = irect(
      0,
      0,
      0,
      0
    )
    minimapPanel.node.position = minimapPanel.rect.xy.vec2
    minimapPanel.node.size = minimapPanel.rect.wh.vec2


    rootArea.node.position = vec2(0, RibbonHeight)
    rootArea.node.visible = false

    topArea.node.position = vec2(0, RibbonHeight)
    topArea.node.visible = true

    bottomArea.node.position = vec2(0, RibbonHeight + ((window.size.y.float32 - RibbonHeight*3) * 0.75))
    bottomArea.node.visible = true


startFidget(
  figmaUrl = "https://www.figma.com/design/hHmLTy7slXTOej6opPqWpz/MetaScope-V2-Rig",
  windowTitle = "MetaScope V2",
  entryFrame = "UI/Main",
  windowStyle = DecoratedResizable
)
