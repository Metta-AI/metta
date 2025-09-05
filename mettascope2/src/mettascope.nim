import std/[random, os, times, strformat, strutils],
  boxy, opengl, windy, windy/http, chroma, vmath, fidget2, fidget2/hybridrender,
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
      rootArea.select(title.text)
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

    worldMapPanel = Panel(panelType: WorldMap, name: "World Map", node: find("**/WorldMap"))
    minimapPanel = Panel(panelType: Minimap, name: "Minimap", node: find("**/Minimap"))
    agentTablePanel = Panel(panelType: AgentTable, name: "Agent Table", node: find("**/AgentTable"))
    agentTracesPanel = Panel(panelType: AgentTraces, name: "Agent Traces", node: find("**/AgentTraces"))
    envConfigPanel = Panel(panelType: EnvConfig, name: "Env Config", node: find("**/EnvConfig"))

    globalTimelinePanel = Panel(panelType: GlobalTimeline, node: find("GlobalTimeline"))
    globalFooterPanel = Panel(panelType: GlobalFooter, node: find("GlobalFooter"))
    globalHeaderPanel = Panel(panelType: GlobalHeader, node: find("GlobalHeader"))

    topArea = Area(layout: Horizontal, node: rootArea.node.copy())
    thisNode.addChild(topArea.node)
    rootArea.add(topArea)
    bottomArea = Area(layout: Horizontal, node: rootArea.node.copy())
    # Update the names of the headers.
    bottomArea.node.children[0].find("title").text = "Agent Traces"
    bottomArea.node.children[1].find("title").text = "Env Config"
    bottomArea.node.children[2].remove()


    thisNode.addChild(bottomArea.node)
    rootArea.add(bottomArea)

    topArea.add(worldMapPanel)
    topArea.add(minimapPanel)
    topArea.add(agentTablePanel)
    bottomArea.add(agentTracesPanel)
    bottomArea.add(envConfigPanel)

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

    playControls()

    if window.buttonReleased[MouseLeft]:
      mouseCaptured = false
      mouseCapturedPanel = nil

    const RibbonHeight = 64
    const HeaderHeight = 28

    rootArea.rect = irect(0, RibbonHeight, window.size.x, window.size.y - RibbonHeight*3)
    topArea.rect = irect(0, rootArea.rect.y, rootArea.rect.w, (rootArea.rect.h.float32 * 0.75).int32)
    bottomArea.rect = irect(0, rootArea.rect.y + (rootArea.rect.h.float32 * 0.75).int, rootArea.rect.w, (rootArea.rect.h.float32 * 0.25).int)
    rootArea.updatePanelsSizes()

startFidget(
  figmaUrl = "https://www.figma.com/design/hHmLTy7slXTOej6opPqWpz/MetaScope-V2-Rig",
  windowTitle = "MetaScope V2",
  entryFrame = "UI/Main",
  windowStyle = DecoratedResizable
)
