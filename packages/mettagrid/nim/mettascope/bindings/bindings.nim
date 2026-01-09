import
  std/times,
  os, genny, openGL, jsony, vmath, windy, silky,
  ../src/mettascope,
  ../src/mettascope/[replays, common, worldmap, timeline, envconfig, vibes, windowstate, panels]

from std/sets import incl, excl, clear, items

type
  ActionRequest* = object
    agentId*: int
    actionName*: cstring

  RenderResponse* = ref object
    shouldClose*: bool
    actions*: seq[ActionRequest]

var
  savedMettascopeState: MettascopeState
  lastStateSaveTime: float64 = 0.0
  stateDirty: bool = false

proc captureSettingsState(): SettingsState =
  ## Capture the current settings.
  result.showFogOfWar = settings.showFogOfWar
  result.showVisualRange = settings.showVisualRange
  result.showGrid = settings.showGrid
  result.showResources = settings.showResources
  result.showObservations = settings.showObservations
  result.lockFocus = settings.lockFocus
  for id in settings.aoeEnabledCollectives:
    result.aoeEnabledCollectives.add(id)

proc applySettingsState(state: SettingsState) =
  ## Apply saved settings.
  settings.showFogOfWar = state.showFogOfWar
  settings.showVisualRange = state.showVisualRange
  settings.showGrid = state.showGrid
  settings.showResources = state.showResources
  settings.showObservations = state.showObservations
  settings.lockFocus = state.lockFocus
  settings.aoeEnabledCollectives.clear()
  for id in state.aoeEnabledCollectives:
    settings.aoeEnabledCollectives.incl(id)

proc saveFullState() =
  ## Save the full mettascope state.
  try:
    var state = MettascopeState()
    state.window = WindowState(
      x: window.pos.x,
      y: window.pos.y,
      width: window.size.x,
      height: window.size.y
    )
    state.zoom = captureZoomState()
    state.panels = capturePanelState()
    state.settings = captureSettingsState()
    saveMettascopeState(state)
  except:
    echo "Error saving state: ", getCurrentExceptionMsg()

proc maybeSaveState() =
  ## Save state if dirty and enough time has passed (debounce).
  # Check if view (pan/zoom) changed
  if viewStateChanged:
    viewStateChanged = false
    stateDirty = true
  let now = epochTime()
  if stateDirty and now - lastStateSaveTime > 1.0:
    saveFullState()
    lastStateSaveTime = now
    stateDirty = false

proc markStateDirty() =
  ## Mark state as needing to be saved.
  stateDirty = true

proc ctrlCHandler() {.noconv.} =
  ## Handle ctrl-c signal to exit cleanly.
  echo "\nNim DLL caught ctrl-c, exiting..."
  saveFullState()
  if not window.isNil:
    window.close()
  quit(0)

proc init(dataDir: string, replay: string, autostart: bool): RenderResponse =
  try:
    echo "Initializing Mettascope..."
    if os.getEnv("METTASCOPE_DISABLE_CTRL_C", "") == "":
      setControlCHook(ctrlCHandler)
    result = RenderResponse(shouldClose: false, actions: @[])
    playMode = Realtime
    setDataDir(dataDir)
    play = autostart
    common.replay = loadReplayString(replay, "MettaScope")
    savedMettascopeState = loadMettascopeState()
    window = newWindow(
      "MettaScope",
      ivec2(savedMettascopeState.window.width, savedMettascopeState.window.height),
      vsync = true
    )
    applyWindowState(window, savedMettascopeState.window)
    # Set up window tracking callbacks
    window.onMove = proc() =
      markStateDirty()
    window.onResize = proc() =
      markStateDirty()
    makeContextCurrent(window)
    loadExtensions()
    # Check if we have saved panel state to restore
    let hasSavedPanels = savedMettascopeState.panels.areas.len > 0 or savedMettascopeState.panels.panelNames.len > 0
    initMettascope(useDefaultPanels = not hasSavedPanels)
    # Restore saved panel layout
    if hasSavedPanels:
      applyPanelState(savedMettascopeState.panels)
    # Restore saved settings (AOE checkboxes, etc.)
    applySettingsState(savedMettascopeState.settings)
    # Set saved zoom state to be applied on first draw (when panel rect is available)
    if savedMettascopeState.zoom.zoom > 0:
      setSavedViewState(
        savedMettascopeState.zoom.zoom,
        savedMettascopeState.zoom.centerX,
        savedMettascopeState.zoom.centerY
      )
    return
  except Exception:
    echo "############ Error initializing Mettascope #################"
    echo getCurrentException().getStackTrace()
    echo getCurrentExceptionMsg()
    echo "############################################################"

    result.shouldClose = true
    return

proc render(currentStep: int, replayStep: string): RenderResponse =
  try:
    common.replay.apply(replayStep)
    step = currentStep
    stepFloat = currentStep.float32
    previousStep = currentStep
    requestPython = false
    result = RenderResponse(shouldClose: false, actions: @[])
    while true:
      if window.closeRequested:
        saveFullState()
        window.close()
        result.shouldClose = true
        return
      tickMettascope()
      maybeSaveState()
      if requestPython:
        onRequestPython()
        for action in requestActions:
          result.actions.add(ActionRequest(
            agentId: action.agentId,
            actionName: action.actionName
          ))
        requestActions.setLen(0)
        markStateDirty()
        return
  except Exception:
    echo "############## Error rendering Mettascope ##################"
    echo getCurrentException().getStackTrace()
    echo getCurrentExceptionMsg()
    echo "############################################################"
    result.shouldClose = true
    return

exportObject ActionRequest:
  discard

exportRefObject RenderResponse:
  fields:
    shouldClose
    actions

exportProcs:
  init
  render

writeFiles("bindings/generated", "Mettascope")

include generated/internal
