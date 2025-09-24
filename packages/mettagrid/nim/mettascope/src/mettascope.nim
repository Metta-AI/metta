import std/[random, os, times, strformat, strutils, parseopt, json],
  boxy, opengl, windy, windy/http, chroma, vmath, fidget2, fidget2/hybridrender,
  mettascope/[replays, common, panels, utils, footer, timeline,
  worldmap, minimap, agenttable, agenttraces, envconfig]

var replay = ""

# TODO: Remove with dynamic panels.
var topArea: Area
var bottomArea: Area

proc updateReplayHeader(replayPath: string) =
  ## Set the global header's display name for the current replay.
  if common.replay.isNil:
    return
  var display = ""
  if common.replay.mgConfig != nil and common.replay.mgConfig.contains("label"):
    let node = common.replay.mgConfig["label"]
    if node.kind == JString:
      display = node.getStr
  if display.len == 0 and common.replay.fileName.len > 0:
    display = common.replay.fileName
  if display.len == 0 and replayPath.len > 0:
    display = extractFilename(replayPath)
  if display.len == 0:
    display = "unknown"

  let titleNode = find("**/GlobalTitle")
  titleNode.text = display



proc parseArgs() =
  ## Parse command line arguments.
  var p = initOptParser(commandLineParams())
  while true:
    p.next()
    case p.kind
    of cmdEnd:
      break
    of cmdLongOption, cmdShortOption:
      case p.key
      of "replay", "r":
        replay = p.val
        echo "Replay: ", replay
      else:
        discard
    of cmdArgument:
      discard



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

  onLoad:
    echo "onLoad"

    # Build the atlas.
    for path in walkDirRec(dataDir):
      if path.endsWith(".png") and "fidget" notin path:
        echo path
        bxy.addImage(path.replace(dataDir & "/", "").replace(".png", ""), readImage(path))

    utils.typeface = readTypeface(dataDir / "fonts" / "Inter-Regular.ttf")

    if replay != "":
      if replay.startsWith("http"):
        echo "Loading replay from URL: ", replay
        let req = startHttpRequest(replay)
        req.onError = proc(msg: string) =
          echo "onError: " & msg
        req.onResponse = proc(response: HttpResponse) =
          echo "onResponse: code=", $response.code, ", len=", response.body.len
          common.replay = loadReplay(response.body, replay)
          updateReplayHeader(replay)
      else:
        common.replay = loadReplay(replay)
        updateReplayHeader(replay)
    elif common.replay == nil:
      common.replay = loadReplay( dataDir / "replays" / "pens.json.z")
      updateReplayHeader(dataDir / "replays" / "pens.json.z")

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

    globalTimelinePanel.node.onRenderCallback = proc(thisNode: Node) =
      bxy.saveTransform()
      timeline.drawTimeline(globalTimelinePanel)
      bxy.restoreTransform()

    echo "Loaded!"

  onFrame:

    playControls()

    if window.buttonReleased[MouseLeft]:
      mouseCaptured = false
      mouseCapturedPanel = nil

    const RibbonHeight = 64
    const HeaderHeight = 28

    let size = (window.size.vec2 / window.contentScale).ivec2
    rootArea.rect = irect(0, RibbonHeight, size.x, size.y - RibbonHeight*3)
    topArea.rect = irect(0, rootArea.rect.y, rootArea.rect.w, (rootArea.rect.h.float32 * 0.75).int32)
    bottomArea.rect = irect(0, rootArea.rect.y + (rootArea.rect.h.float32 * 0.75).int, rootArea.rect.w, (rootArea.rect.h.float32 * 0.25).int)
    rootArea.updatePanelsSizes()

    if not common.replay.isNil and worldMapPanel.pos == vec2(0, 0):
      fitFullMap(worldMapPanel)

when isMainModule:

  parseArgs()

  initFidget(
    figmaUrl = "https://www.figma.com/design/hHmLTy7slXTOej6opPqWpz/MetaScope-V2-Rig",
    windowTitle = "MetaScope V2",
    entryFrame = "UI/Main",
    windowStyle = DecoratedResizable,
    dataDir = "packages/mettagrid/nim/mettascope/data"
  )

  when defined(emscripten):
    # Emscripten can't block so it will call this callback instead.
    window.run(mainLoop)
  else:
    # When running native code we can block in an infinite loop.
    while not window.closeRequested:
      mainLoop()
    # Destroy the window.
    window.close()
