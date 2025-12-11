import std/[os, strutils, parseopt, json, tables],
  boxy, windy, vmath, fidget2, fidget2/hybridrender, webby,
  mettascope/[replays, common, panels, utils, timeline,
  worldmap, minimap, agenttraces, footer, envconfig, vibes]

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
  ## Called when a replay is loaded.
  # Clear cached maps that depend on the old replay
  terrainMap = nil
  visibilityMap = nil

  # Reset global state for the new replay
  step = 0
  stepFloat = 0.0
  previousStep = -1
  selection = nil
  play = false
  requestPython = false
  agentPaths = initTable[int, seq[PathAction]]()
  agentObjectives = initTable[int, seq[Objective]]()

  replay.loadImages()
  updateReplayHeader()
  worldMapPanel.pos = vec2(0, 0)
  onStepChanged()
  updateEnvConfig()
  updateVibePanel()

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

when defined(emscripten):
  # include for EMSCRIPTEN_KEEPALIVE
  {.emit: """
  #include <emscripten.h>
  """.}

  # PostMessage handler for receiving replay data from Jupyter notebooks
  {.emit: """
  EM_JS(void, setup_postmessage_replay_handler_internal, (void* userData), {
    // Validate origin helper function
    function isValidOrigin(origin) {
      // Allow Google Colab
      if (origin === 'https://colab.research.google.com') {
        return true;
      }
      // Allow localhost variants (http and https, any port)
      if (origin.startsWith('http://localhost:') || origin.startsWith('https://localhost:')) {
        return true;
      }
      // Allow 127.0.0.1 variants
      if (origin.startsWith('http://127.0.0.1:') || origin.startsWith('https://127.0.0.1:')) {
        return true;
      }
      return false;
    }

    window.addEventListener('message', function(event) {
      // Validate origin for security
      console.log("Received postMessage from origin:", event.origin);
      if (!isValidOrigin(event.origin)) {
        console.log('Ignoring postMessage from invalid origin:', event.origin);
        return;
      }

      // Only process messages with the expected type
      if (!event.data || event.data.type !== 'replayData') {
        return;
      }

      // Get base64 string from message
      const base64Data = event.data.base64;
      if (!base64Data || typeof base64Data !== 'string') {
        console.error('Invalid replayData: base64 field missing or not a string');
        return;
      }

      try {
        // Decode base64 to binary string
        const binaryString = atob(base64Data);
        const binaryLength = binaryString.length;

        // Allocate memory for the binary data
        const binaryPtr = _malloc(binaryLength);
        if (!binaryPtr) {
          console.error('Failed to allocate memory for replay data');
          return;
        }

        // Copy binary string to heap
        for (let i = 0; i < binaryLength; i++) {
          HEAPU8[binaryPtr + i] = binaryString.charCodeAt(i);
        }

        // Get filename from message data or use default
        const fileName = event.data.fileName || 'replay_from_notebook.json.z';
        const fileNameLen = lengthBytesUTF8(fileName) + 1;
        const fileNamePtr = _malloc(fileNameLen);
        stringToUTF8(fileName, fileNamePtr, fileNameLen);

        // Call the Nim callback
        Module._mettascope_postmessage_replay_callback(userData, fileNamePtr, binaryPtr, binaryLength);

        // Free allocated memory
        _free(fileNamePtr);
        _free(binaryPtr);
      } catch (error) {
        console.error('Error processing postMessage replay data:', error);
      }
    });
  });
  """.}

  proc setup_postmessage_replay_handler_internal*(userData: pointer) {.importc.}

  proc mettascope_postmessage_replay_callback(userData: pointer, fileNamePtr: cstring, binaryPtr: pointer, binaryLen: cint) {.exportc, cdecl, codegenDecl: "EMSCRIPTEN_KEEPALIVE $# $#$#".} =
    ## Callback to handle postMessage replay data.
    ## EMSCRIPTEN_KEEPALIVE is required to avoid dead code elimination.
    
    # Convert the JS data into Nim data
    let fileName = $fileNamePtr
    var fileData = newString(binaryLen)
    if binaryLen > 0:
      copyMem(fileData[0].addr, binaryPtr, binaryLen)
    
    # Process the replay data
    echo "Received replay via postMessage: ", fileName, " (", fileData.len, " bytes)"
    if fileName.endsWith(".json.z"):
      try:
        common.replay = loadReplay(fileData, fileName)
        onReplayLoaded()
        echo "Successfully loaded replay from postMessage: ", fileName
      except:
        echo "Error loading replay from postMessage: ", getCurrentExceptionMsg()
    else:
      echo "Ignoring postMessage data (not .json.z): ", fileName

find "/UI/Main":

  onLoad:
    # We need to build the atlas before loading the replay.
    buildAtlas()
    
    window.onFileDrop = proc(fileName: string, fileData: string) =
      echo "File dropped: ", fileName, " (", fileData.len, " bytes)"
      if fileName.endsWith(".json.z"):
        try:
          common.replay = loadReplay(fileData, fileName)
          onReplayLoaded()
          echo "Successfully loaded replay: ", fileName
        except:
          echo "Error loading replay file: ", getCurrentExceptionMsg()
      else:
        echo "Ignoring dropped file (not .json.z): ", fileName

    when defined(emscripten):
      setup_postmessage_replay_handler_internal(cast[pointer](window))

    utils.typeface = readTypeface(dataDir / "fonts" / "Inter-Regular.ttf")

    rootArea.split(Vertical)
    rootArea.split = 0.30

    rootArea.areas[0].split(Horizontal)
    rootArea.areas[0].split = 0.7

    rootArea.areas[1].split(Vertical)
    rootArea.areas[1].split = 0.6

    objectInfoPanel = rootArea.areas[0].areas[0].addPanel(ObjectInfo, "Object")
    environmentInfoPanel = rootArea.areas[0].areas[0].addPanel(EnvironmentInfo, "Environment")

    worldMapPanel = rootArea.areas[1].areas[0].addPanel(WorldMap, "Map")
    minimapPanel = rootArea.areas[0].areas[1].addPanel(Minimap, "Minimap")

    #agentTracesPanel = rootArea.areas[1].areas[0].addPanel(AgentTraces, "Agent Traces")
    # agentTablePanel = rootArea.areas[1].areas[1].addPanel(AgentTable, "Agent Table")

    vibePanel = rootArea.areas[1].areas[1].addPanel(VibePanel, "Vibe Selector")

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
        fitVisibleMap(worldMapPanel)
      adjustPanelForResize(worldMapPanel)
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

    # agentTracesPanel.node.onRenderCallback = proc(thisNode: Node) =
    #   bxy.saveTransform()
    #   agentTracesPanel.rect = irect(
    #     thisNode.absolutePosition.x,
    #     thisNode.absolutePosition.y,
    #     thisNode.size.x,
    #     thisNode.size.y
    #   )
    #   bxy.translate(agentTracesPanel.rect.xy.vec2 * window.contentScale)
    #   drawAgentTraces(agentTracesPanel)
    #   bxy.restoreTransform()

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

    case common.playMode
    of Historical:
      if commandLineReplay != "":
        if commandLineReplay.startsWith("http"):
          common.replay = EmptyReplay
          echo "fetching replay from URL: ", commandLineReplay
          let req = startHttpRequest(commandLineReplay)
          req.onError = proc(msg: string) =
            # TODO: Show error to user.
            echo "onError: " & msg
            echo getCurrentException().getStackTrace()
          req.onResponse = proc(response: HttpResponse) =
            if response.code != 200:
              # TODO: Show error to user.
              echo "Error loading replay: HTTP ", response.code, " ", response.body
              return
            echo "replay fetched, loading..."
            common.replay = loadReplay(response.body, commandLineReplay)
            onReplayLoaded()
        else:
          echo "Loading replay from file: ", commandLineReplay
          common.replay = loadReplay(commandLineReplay)
          onReplayLoaded()
      elif common.replay == nil:
        let defaultReplay = dataDir / "replays" / "pens.json.z"
        echo "Loading replay from default file: ", defaultReplay
        common.replay = loadReplay(defaultReplay)
        onReplayLoaded()
    of Realtime:
      echo "Realtime mode"
      onReplayLoaded()

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

    if window.buttonPressed[KeyF8]:
      fitFullMap(worldMapPanel)

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
    windowTitle = "MettaScope",
    entryFrame = "UI/Main",
    windowStyle = DecoratedResizable,
    dataDir = dataDir
  )

  while isRunning():
    tickFidget()
    when not defined(emscripten):
      pollHttp()
  closeFidget()
