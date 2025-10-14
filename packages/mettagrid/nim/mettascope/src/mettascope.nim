import std/[os, strutils, parseopt, json],
  boxy, windy, vmath, fidget2, fidget2/hybridrender, webby,
  mettascope/[replays, common, panels, utils, timeline,
  worldmap, minimap, agenttraces, footer, objectinfo, envconfig]

proc updateReplayHeader() =
  ## Set the global header's display name for the current session.
  var display = "Mettascope"

  if not common.replay.isNil:
    if common.replay.mgConfig != nil and common.replay.mgConfig.contains("label"):
      let node = common.replay.mgConfig["label"]
      if node.kind == JString:
        display = node.getStr
    if display == "Mettascope" and common.replay.fileName.len > 0:
      display = common.replay.fileName
  let titleNode = find("**/GlobalTitle")
  titleNode.text = display

proc onReplayLoaded() =
  updateReplayHeader()

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
        commandLineReplay = p.val
      else:
        quit("Unknown option: " & p.key)
    of cmdArgument:
      quit("Unknown option: " & p.key)

proc parseUrlParams() =
  ## Parse URL parameters.
  let url = parseUrl(window.url)
  commandLineReplay = url.query["replay"]
  echo "Command line replay: ", commandLineReplay

find "/UI/Main":

  onLoad:
    # We need to build the atlas before loading the replay.
    buildAtlas()

    utils.typeface = readTypeface(dataDir / "fonts" / "Inter-Regular.ttf")

    case common.playMode
    of Historical:
      if commandLineReplay != "":
        if commandLineReplay.startsWith("http"):
          echo "Loading built-in replay while web is loading"
          common.replay = loadReplay(dataDir / "replays" / "pens.json.z")
          onReplayLoaded()
          echo "Loading replay from URL: ", commandLineReplay
          let req = startHttpRequest(commandLineReplay)
          req.onError = proc(msg: string) =
            echo "onError: " & msg
          req.onResponse = proc(response: HttpResponse) =
            echo "onResponse: code=", $response.code, ", len=", response.body.len
            common.replay = loadReplay(response.body, commandLineReplay)
            onReplayLoaded()
        else:
          echo "Loading replay from file: ", commandLineReplay
          common.replay = loadReplay(commandLineReplay)
          onReplayLoaded()
      elif common.replay == nil:
        echo "Loading built-in replay"
        common.replay = loadReplay( dataDir / "replays" / "pens.json.z")
        onReplayLoaded()
    of Realtime:
      echo "Realtime mode detected"
      onReplayLoaded()

    rootArea.split(Vertical)
    rootArea.split = 0.20

    rootArea.areas[0].split(Horizontal)
    rootArea.areas[0].split = 0.8

    objectInfoPanel = rootArea.areas[0].areas[0].addPanel(ObjectInfo, "Object")
    environmentInfoPanel = rootArea.areas[0].areas[0].addPanel(EnvironmentInfo, "Environment")

    worldMapPanel = rootArea.areas[1].addPanel(WorldMap, "Map")
    minimapPanel = rootArea.areas[0].areas[1].addPanel(Minimap, "Minimap")

    agentTracesPanel = rootArea.areas[1].addPanel(AgentTraces, "Agent Traces")
    # agentTablePanel = rootArea.areas[1].areas[1].addPanel(AgentTable, "Agent Table")

    rootArea.refresh()

    globalTimelinePanel = Panel(panelType: GlobalTimeline, node: find("GlobalTimeline"))
    globalFooterPanel = Panel(panelType: GlobalFooter, node: find("GlobalFooter"))
    globalHeaderPanel = Panel(panelType: GlobalHeader, node: find("GlobalHeader"))

    worldMapPanel.node.onRenderCallback = proc(thisNode: Node) =
      bxy.saveTransform()
      worldMapPanel.rect = irect(
        thisNode.absolutePosition.x,
        thisNode.absolutePosition.y,
        thisNode.size.x,
        thisNode.size.y
      )
      if not common.replay.isNil and worldMapPanel.pos == vec2(0, 0):
        fitFullMap(worldMapPanel)
      bxy.translate(worldMapPanel.rect.xy.vec2 * window.contentScale)
      drawWorldMap(worldMapPanel)
      bxy.restoreTransform()

    minimapPanel.node.onRenderCallback = proc(thisNode: Node) =
      bxy.saveTransform()
      minimapPanel.rect = irect(
        thisNode.absolutePosition.x,
        thisNode.absolutePosition.y,
        thisNode.size.x,
        thisNode.size.y
      )
      bxy.translate(minimapPanel.rect.xy.vec2 * window.contentScale)
      drawMinimap(minimapPanel)
      bxy.restoreTransform()

    agentTracesPanel.node.onRenderCallback = proc(thisNode: Node) =
      bxy.saveTransform()
      agentTracesPanel.rect = irect(
        thisNode.absolutePosition.x,
        thisNode.absolutePosition.y,
        thisNode.size.x,
        thisNode.size.y
      )
      bxy.translate(agentTracesPanel.rect.xy.vec2 * window.contentScale)
      drawAgentTraces(agentTracesPanel)
      bxy.restoreTransform()

    globalTimelinePanel.node.onRenderCallback = proc(thisNode: Node) =
      bxy.saveTransform()
      globalTimelinePanel.rect = irect(
        thisNode.position.x,
        thisNode.position.y,
        thisNode.size.x,
        thisNode.size.y
      )
      timeline.drawTimeline(globalTimelinePanel)
      bxy.restoreTransform()

    onStepChanged()
    updateEnvConfig()

  onFrame:

    playControls()

    # super+w or super+q closes window on Mac.
    when defined(macosx):
      let superDown = window.buttonDown[KeyLeftSuper] or window.buttonDown[KeyRightSuper]
      if superDown and (window.buttonPressed[KeyW] or window.buttonPressed[KeyQ]):
        window.closeRequested = true

    if window.buttonReleased[MouseLeft]:
      mouseCaptured = false
      mouseCapturedPanel = nil

when isMainModule:

  # Check if the data directory exists.
  let dataDir = "packages/mettagrid/nim/mettascope/data"
  if not dirExists(dataDir):
    echo "Data directory does not exist: ", dataDir
    echo "Please run it from the root of the project."
    quit(1)

  when defined(emscripten):
    parseUrlParams()
  else:
    parseArgs()

  startFidget(
    figmaUrl = "https://www.figma.com/design/hHmLTy7slXTOej6opPqWpz/MetaScope-V2-Rig",
    windowTitle = "MetaScope V2",
    entryFrame = "UI/Main",
    windowStyle = DecoratedResizable,
    dataDir = dataDir
  )

  while isRunning():
    tickFidget()
  closeFidget()
