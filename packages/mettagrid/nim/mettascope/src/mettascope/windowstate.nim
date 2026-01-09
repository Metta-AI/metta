import
  std/[os, json],
  vmath, windy

const
  ConfigFileName = "mettascope_state.json"

type
  WindowState* = object
    x*: int32
    y*: int32
    width*: int32
    height*: int32

  ZoomState* = object
    zoom*: float32
    centerX*: float32  ## Center of view in world coordinates
    centerY*: float32  ## Center of view in world coordinates

  AreaState* = object
    layout*: int  # 0 = Horizontal, 1 = Vertical
    split*: float32
    selectedPanelNum*: int
    panelNames*: seq[string]
    areas*: seq[AreaState]

  SettingsState* = object
    showFogOfWar*: bool
    showVisualRange*: bool
    showGrid*: bool
    showResources*: bool
    showObservations*: int
    lockFocus*: bool
    aoeEnabledCollectives*: seq[int]  ## List of collective IDs with AOE enabled

  MettascopeState* = object
    window*: WindowState
    zoom*: ZoomState
    panels*: AreaState
    settings*: SettingsState

proc getConfigPath(): string =
  ## Get the path to the config file in the user's home directory.
  result = getHomeDir() / ".config" / "mettascope"
  discard existsOrCreateDir(result)
  result = result / ConfigFileName

proc saveMettascopeState*(state: MettascopeState) =
  ## Save the full mettascope state to a config file.
  let configPath = getConfigPath()
  try:
    writeFile(configPath, $(%state))
  except:
    echo "Failed to save mettascope state: ", getCurrentExceptionMsg()

proc loadMettascopeState*(): MettascopeState =
  ## Load the mettascope state from a config file.
  let configPath = getConfigPath()
  result = MettascopeState(
    window: WindowState(x: -1, y: -1, width: 1200, height: 800),
    zoom: ZoomState(zoom: 0, centerX: -1, centerY: -1),
    panels: AreaState(),
    settings: SettingsState(
      showFogOfWar: false,
      showVisualRange: true,
      showGrid: true,
      showResources: true,
      showObservations: -1,
      lockFocus: false,
      aoeEnabledCollectives: @[]
    )
  )
  if fileExists(configPath):
    try:
      let content = readFile(configPath)
      let json = parseJson(content)
      if json.hasKey("window"):
        let w = json["window"]
        result.window.x = w["x"].getInt().int32
        result.window.y = w["y"].getInt().int32
        result.window.width = w["width"].getInt().int32
        result.window.height = w["height"].getInt().int32
      if json.hasKey("zoom"):
        let z = json["zoom"]
        result.zoom.zoom = z["zoom"].getFloat().float32
        if z.hasKey("centerX"):
          result.zoom.centerX = z["centerX"].getFloat().float32
        if z.hasKey("centerY"):
          result.zoom.centerY = z["centerY"].getFloat().float32
      if json.hasKey("panels"):
        proc parseAreaState(node: JsonNode): AreaState =
          result.layout = node["layout"].getInt()
          result.split = node["split"].getFloat().float32
          result.selectedPanelNum = node["selectedPanelNum"].getInt()
          if node.hasKey("panelNames"):
            for name in node["panelNames"]:
              result.panelNames.add(name.getStr())
          if node.hasKey("areas"):
            for areaNode in node["areas"]:
              result.areas.add(parseAreaState(areaNode))
        result.panels = parseAreaState(json["panels"])
      if json.hasKey("settings"):
        let s = json["settings"]
        if s.hasKey("showFogOfWar"):
          result.settings.showFogOfWar = s["showFogOfWar"].getBool()
        if s.hasKey("showVisualRange"):
          result.settings.showVisualRange = s["showVisualRange"].getBool()
        if s.hasKey("showGrid"):
          result.settings.showGrid = s["showGrid"].getBool()
        if s.hasKey("showResources"):
          result.settings.showResources = s["showResources"].getBool()
        if s.hasKey("showObservations"):
          result.settings.showObservations = s["showObservations"].getInt()
        if s.hasKey("lockFocus"):
          result.settings.lockFocus = s["lockFocus"].getBool()
        if s.hasKey("aoeEnabledCollectives"):
          for id in s["aoeEnabledCollectives"]:
            result.settings.aoeEnabledCollectives.add(id.getInt())
    except:
      echo "Failed to load mettascope state: ", getCurrentExceptionMsg()

proc applyWindowState*(window: Window, state: WindowState) =
  ## Apply the saved window state to the window.
  if state.width > 0 and state.height > 0:
    window.size = ivec2(state.width, state.height)
  if state.x >= 0 and state.y >= 0:
    window.pos = ivec2(state.x, state.y)
