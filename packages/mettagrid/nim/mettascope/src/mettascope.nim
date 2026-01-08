import
  std/[strutils, strformat, os, parseopt],
  opengl, windy, bumpy, vmath, chroma, silky, boxy, webby,
  mettascope/[replays, common, worldmap, panels, objectinfo, envconfig, vibes,
  footer, timeline, minimap, header, replayloader]

when isMainModule:
  # Build the atlas.
  var builder = newAtlasBuilder(1024, 4)
  builder.addDir(rootDir / "data/theme/", rootDir / "data/theme/")
  builder.addDir(rootDir / "data/ui/", rootDir / "data/")
  builder.addDir(rootDir / "data/vibe/", rootDir / "data/")
  builder.addDir(rootDir / "data/resources/", rootDir / "data/")
  # builder.addDir(rootDir / "data/agents/", rootDir / "data/")
  builder.addFont(rootDir / "data/fonts/Inter-Regular.ttf", "H1", 32.0)
  builder.addFont(rootDir / "data/fonts/Inter-Regular.ttf", "Default", 18.0)
  builder.write(rootDir / "data/silky.atlas.png", rootDir / "data/silky.atlas.json")

  window = newWindow(
    "MettaScope",
    ivec2(1200, 800),
    vsync = true
  )
  makeContextCurrent(window)
  loadExtensions()

const
  BackgroundColor = parseHtmlColor("#000000").rgbx
  RibbonColor = parseHtmlColor("#273646").rgbx
  m = 12f # Default margin

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

proc replaySwitch(replay: string) =
  ## Load the replay.
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
      let defaultReplay = dataDir / "replays" / "dinky7.json.z"
      echo "Loading replay from default file: ", defaultReplay
      common.replay = loadReplay(defaultReplay)
      onReplayLoaded()
  of Realtime:
    echo "Realtime mode"
    onReplayLoaded()

proc genericPanelDraw(panel: Panel, frameId: string, contentPos: Vec2, contentSize: Vec2) =
  frame(frameId, contentPos, contentSize):
    # Start content a bit inset.
    sk.at += vec2(8, 8)
    h1text(panel.name)
    text("This is the content of " & panel.name)
    for i in 0 ..< 20:
      text(&"Scrollable line {i} for " & panel.name)

proc drawWorldMap(panel: Panel, frameId: string, contentPos: Vec2, contentSize: Vec2) =
  ## Draw the world map.
  sk.draw9Patch("panel.body.empty.9patch", 3, contentPos, contentSize)

  worldMapZoomInfo.rect = irect(contentPos.x, contentPos.y, contentSize.x, contentSize.y)
  worldMapZoomInfo.hasMouse = mouseInsideClip(rect(contentPos, contentSize))

  glEnable(GL_SCISSOR_TEST)
  glScissor(contentPos.x.int32, window.size.y.int32 - contentPos.y.int32 - contentSize.y.int32, contentSize.x.int32, contentSize.y.int32)
  glClearColor(1.0f, 0.0f, 0.0f, 1.0f)

  bxy.saveTransform()
  bxy.translate(contentPos)
  drawWorldMap(worldMapZoomInfo)
  bxy.restoreTransform()

  glDisable(GL_SCISSOR_TEST)

proc drawMinimap(panel: Panel, frameId: string, contentPos: Vec2, contentSize: Vec2) =
  ## Draw the minimap.
  sk.draw9Patch("panel.body.empty.9patch", 3, contentPos, contentSize)

  glEnable(GL_SCISSOR_TEST)
  glScissor(contentPos.x.int32, window.size.y.int32 - contentPos.y.int32 - contentSize.y.int32, contentSize.x.int32, contentSize.y.int32)

  let minimapZoomInfo = ZoomInfo()
  minimapZoomInfo.rect = irect(contentPos.x, contentPos.y, contentSize.x, contentSize.y)
  # Adjust zoom info and draw the minimap.
  minimapZoomInfo.hasMouse = false

  bxy.saveTransform()
  bxy.translate(contentPos)
  drawMinimap(minimapZoomInfo)
  bxy.restoreTransform()

  glDisable(GL_SCISSOR_TEST)

proc initPanels() =

  rootArea = Area()
  rootArea.split(Vertical)
  rootArea.split = 0.22

  rootArea.areas[0].split(Horizontal)
  rootArea.areas[0].split = 0.7

  rootArea.areas[1].split(Vertical)
  rootArea.areas[1].split = 0.85

  rootArea.areas[0].areas[0].addPanel("Object", drawObjectInfo)
  rootArea.areas[0].areas[0].addPanel("Environment", drawEnvironmentInfo)

  rootArea.areas[1].areas[0].addPanel("Map", drawWorldMap)
  rootArea.areas[0].areas[1].addPanel("Minimap", drawMinimap)

  rootArea.areas[1].areas[1].addPanel("Vibes", drawVibes)


proc onFrame() =

  playControls()

  sk.beginUI(window, window.size)

  glClearColor(0.0f, 0.0f, 0.0f, 1.0f)
  glClear(GL_COLOR_BUFFER_BIT)

  # Header
  drawHeader()

  # Scrubber
  drawTimeline(vec2(0, sk.size.y - 64 - 22), vec2(sk.size.x, 32))

  # Footer
  drawFooter(vec2(0, sk.size.y - 64), vec2(sk.size.x, 64))

  drawPanels()

  when defined(profile):
    let ms = sk.avgFrameTime * 1000
    sk.at = sk.pos + vec2(sk.size.x - 250, 20)
    text(&"frame time: {ms:>7.3f}ms\nquads: {sk.instanceCount}")

  sk.endUi()
  window.swapBuffers()

  if window.cursor.kind != sk.cursor.kind:
    window.cursor = sk.cursor

proc initMettascope*() =

  window.onFrame = onFrame

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

  initPanels()

  sk = newSilky(rootDir / "data/silky.atlas.png", rootDir / "data/silky.atlas.json")
  bxy = newBoxy()

  ## Initialize the world map zoom info.
  worldMapZoomInfo = ZoomInfo()
  worldMapZoomInfo.rect = IRect(x: 0, y: 0, w: 500, h: 500)
  worldMapZoomInfo.pos = vec2(0, 0)
  worldMapZoomInfo.zoom = 10
  worldMapZoomInfo.minZoom = 0.5
  worldMapZoomInfo.maxZoom = 50
  worldMapZoomInfo.scrollArea = Rect(x: 0, y: 0, w: 500, h: 500)
  worldMapZoomInfo.hasMouse = false

  if playMode == Historical:
    when defined(emscripten):
      parseUrlParams()
    else:
      parseArgs()
    replaySwitch(commandLineReplay)

proc tickMettascope*() =
  pollEvents()

proc main() =
  ## Main entry point.
  initMettascope()

  while not window.closeRequested:
    tickMettascope()

when isMainModule:
  main()
