import
  std/[strutils, strformat, os, parseopt, json],
  opengl, windy, bumpy, vmath, chroma, silky, boxy, webby,
  mettascope/[replays, common, worldmap]



var builder = newAtlasBuilder(1024, 4)

builder.addDir(rootDir / "data/theme/", rootDir / "data/theme/")
builder.addDir(rootDir / "data/ui/", rootDir / "data/")
builder.addDir(rootDir / "data/vibe/", rootDir / "data/")

builder.addFont(rootDir / "data/fonts/Inter-Regular.ttf", "H1", 32.0)
builder.addFont(rootDir / "data/fonts/Inter-Regular.ttf", "Default", 18.0)

builder.write(rootDir / "dist/atlas.png", rootDir / "dist/atlas.json")


const
  BackgroundColor = parseHtmlColor("#000000").rgbx
  RibbonColor = parseHtmlColor("#273646").rgbx
  ScrubberColor = parseHtmlColor("#1D1D1D").rgbx
  m = 12f # Default margin

window = newWindow(
  "MettaScope",
  ivec2(1200, 800),
  vsync = false
)

window.centerWindow()
makeContextCurrent(window)
loadExtensions()

sk = newSilky(rootDir / "dist/atlas.png", rootDir / "dist/atlas.json")
bxy = newBoxy()

var scrubValue: float32 = 0


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

proc onReplayLoaded() =
  ## Called when a replay is loaded.
  #replay.loadImages()
  #updateReplayHeader()
  #worldMapPanel.pos = vec2(0, 0)
  #onStepChanged()
  #updateEnvConfig()
  #updateVibePanel()
  echo "Replay loaded: ", replay.fileName

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


when defined(emscripten):
  parseUrlParams()
else:
  parseArgs()
replaySwitch(commandLineReplay)

var vibes = @[
  "vibe/alembic",
  "vibe/angry",
  "vibe/anxious",
  "vibe/assembler",
  "vibe/asterisk",
  "vibe/backpack",
  "vibe/beaming",
  "vibe/black-circle",
  "vibe/black-heart",
  "vibe/blue-circle",
  "vibe/blue-diamond",
  "vibe/blue-heart",
  "vibe/bow",
  "vibe/broken-heart",
  "vibe/brown-circle",
  "vibe/brown-heart",
  "vibe/brown-square",
  "vibe/carbon",
  "vibe/carbon_a",
  "vibe/carbon_b",
  "vibe/carrot",
  "vibe/charger",
  "vibe/chart-down",
  "vibe/chart-up",
  "vibe/chest",
  "vibe/clown",
  "vibe/coin",
  "vibe/compass",
  "vibe/confused",
  "vibe/corn",
  "vibe/crying-cat",
  "vibe/crying",
  "vibe/dagger",
  "vibe/default",
  "vibe/diamond",
  "vibe/divide",
  "vibe/down-left",
  "vibe/down-right",
  "vibe/down",
  "vibe/drooling",
  "vibe/eight",
  "vibe/factory",
  "vibe/fearful",
  "vibe/fire",
  "vibe/five",
  "vibe/four",
  "vibe/fuel",
  "vibe/gear",
  "vibe/germanium",
  "vibe/germanium_a",
  "vibe/germanium_b",
  "vibe/ghost",
  "vibe/green-circle",
  "vibe/green-heart",
  "vibe/grinning-big-eyes",
  "vibe/grinning-smiling-eyes",
  "vibe/grinning",
  "vibe/growing-heart",
  "vibe/halo",
  "vibe/hammer",
  "vibe/hash",
  "vibe/heart-arrow",
  "vibe/heart-decoration",
  "vibe/heart-exclamation",
  "vibe/heart-eyes",
  "vibe/heart-ribbon",
  "vibe/heart",
  "vibe/heart_a",
  "vibe/heart_b",
  "vibe/hundred",
  "vibe/kiss",
  "vibe/left",
  "vibe/light-shade",
  "vibe/lightning",
  "vibe/love-letter",
  "vibe/medium-shade",
  "vibe/minus",
  "vibe/moai",
  "vibe/money",
  "vibe/monocle",
  "vibe/mountain",
  "vibe/multiply",
  "vibe/nine",
  "vibe/numbers",
  "vibe/oil",
  "vibe/one",
  "vibe/orange-circle",
  "vibe/orange-heart",
  "vibe/orange-square",
  "vibe/oxygen",
  "vibe/oxygen_a",
  "vibe/oxygen_b",
  "vibe/package",
  "vibe/paperclip",
  "vibe/pin",
  "vibe/plug",
  "vibe/plus",
  "vibe/pouting",
  "vibe/purple-circle",
  "vibe/purple-heart",
  "vibe/purple-square",
  "vibe/pushpin",
  "vibe/red-circle",
  "vibe/red-heart",
  "vibe/red-triangle",
  "vibe/revolving-hearts",
  "vibe/right",
  "vibe/rock",
  "vibe/rocket",
  "vibe/rofl",
  "vibe/rolling-eyes",
  "vibe/rotate-clockwise",
  "vibe/rotate",
  "vibe/savoring",
  "vibe/seahorse",
  "vibe/seven",
  "vibe/shield",
  "vibe/silicon",
  "vibe/silicon_a",
  "vibe/silicon_b",
  "vibe/six",
  "vibe/skull-crossbones",
  "vibe/sleepy",
  "vibe/small-blue-diamond",
  "vibe/smiling",
  "vibe/smirking",
  "vibe/sobbing",
  "vibe/sparkle",
  "vibe/sparkling-heart",
  "vibe/squinting",
  "vibe/star-struck",
  "vibe/swearing",
  "vibe/swords",
  "vibe/target",
  "vibe/tears-of-joy",
  "vibe/ten",
  "vibe/test-tube",
  "vibe/three",
  "vibe/tree",
  "vibe/two-hearts",
  "vibe/two",
  "vibe/up-left",
  "vibe/up-right",
  "vibe/up",
  "vibe/wall",
  "vibe/water",
  "vibe/wave",
  "vibe/wheat",
  "vibe/white-circle",
  "vibe/white-heart",
  "vibe/white-square",
  "vibe/wood",
  "vibe/wrench",
  "vibe/yawning",
  "vibe/yellow-circle",
  "vibe/yellow-heart",
  "vibe/yellow-square",
  "vibe/zero",
]


var worldMapPanel = Panel(panelType: WorldMap, name: "World Map")
worldMapPanel.rect = IRect(x: 0, y: 0, w: 500, h: 500)
worldMapPanel.pos = vec2(0, 0)
worldMapPanel.zoom = 10
worldMapPanel.minZoom = 0.5
worldMapPanel.maxZoom = 50
worldMapPanel.scrollArea = Rect(x: 0, y: 0, w: 500, h: 500)
worldMapPanel.hasMouse = false

window.onFrame = proc() =

  sk.beginUI(window, window.size)

  # # Draw map background
  # for x in 0 ..< 16:
  #   for y in 0 ..< 10:
  #     sk.at = vec2(x.float32 * 256, y.float32 * 256)
  #     image("testTexture", rgbx(30, 30, 30, 255))



  drawWorldMap(worldMapPanel)

  # Header
  ribbon(sk.pos, vec2(sk.size.x, 64), RibbonColor):
    image("ui/logo")
    h1text("Hello, World!")

    # button("press me"):
    #   echo "pressed"

    sk.at = sk.pos + vec2(sk.size.x - 100, 16)
    iconButton("ui/heart"):
      echo "heart"
    iconButton("ui/cloud"):
      echo "cloud"

  # Scrubber
  ribbon(vec2(0, sk.size.y - 64*2), vec2(sk.size.x, 66), ScrubberColor):
    scrubber("timeline", sk.pos + vec2(16, 32), vec2(sk.size.x - 32, 32), scrubValue, 0, 1000)

  # Footer
  ribbon(vec2(0, sk.size.y - 64), vec2(sk.size.x, 64), RibbonColor):

    group(vec2(16, 16)):
      iconButton("ui/rewindToStart"):
        echo "rewindToStart"
      iconButton("ui/stepBack"):
        echo "stepBack"
      iconButton("ui/play"):
        echo "play"
      iconButton("ui/stepForward"):
        echo "stepForward"
      iconButton("ui/rewindToEnd"):
        echo "rewindToEnd"

    group(vec2(sk.size.x - 240, 16)):
      iconButton("ui/heart"):
        echo "heart"
      iconButton("ui/cloud"):
        echo "cloud"
      iconButton("ui/grid"):
        echo "grid"
      iconButton("ui/eye"):
        echo "eye"
      iconButton("ui/tack"):
        echo "tack"

  # frame(vec2(sk.size.x - (11 * (32 + m)), 100) - vec2(14, 14), vec2(500, 800) + vec2(14, 14)):
  #   sk.at = sk.pos + vec2(m, m) * 2
  #   for i, vibe in vibes:
  #     if i > 0 and i mod 10 == 0:
  #       sk.at.x = sk.pos.x + m * 2
  #       sk.at.y += 32 + m
  #     iconButton(vibe):
  #       echo vibe

  group vec2(10, 200):
    text("Step: 1 of 10\nscore: 100\nlevel: 1\nwidth: 100\nheight: 100\nnum agents: 10")

  let ms = sk.avgFrameTime * 1000
  sk.at = sk.pos + vec2(sk.size.x - 250, 20)
  text(&"frame time: {ms:>7.3f}ms")

  sk.endUi()
  window.swapBuffers()

while not window.closeRequested:
  pollEvents()
