import
  genny, openGL, jsony, vmath, windy, silky,
  ../src/mettascope,
  ../src/mettascope/[replays, common, worldmap, timeline, envconfig, vibes]

type
  ActionRequest* = object
    agentId*: int
    actionName*: cstring

  RenderResponse* = ref object
    shouldClose*: bool
    actions*: seq[ActionRequest]

proc ctrlCHandler() {.noconv.} =
  ## Handle ctrl-c signal to exit cleanly.
  echo "\nNim DLL caught ctrl-c, exiting..."
  if not window.isNil:
    window.close()
  quit(0)

proc init(dataDir: string, replay: string): RenderResponse =
  try:
    echo "Initializing Mettascope..."
    setControlCHook(ctrlCHandler)
    result = RenderResponse(shouldClose: false, actions: @[])
    playMode = Realtime
    setDataDir(dataDir)
    common.replay = loadReplayString(replay, "MettaScope")
    window = newWindow(
      "MettaScope",
      ivec2(1200, 800),
      vsync = true
    )
    makeContextCurrent(window)
    loadExtensions()
    initMettascope()
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
        window.close()
        result.shouldClose = true
        return
      tickMettascope()
      if requestPython:
        onRequestPython()
        for action in requestActions:
          result.actions.add(ActionRequest(
            agentId: action.agentId,
            actionName: action.actionName
          ))
        requestActions.setLen(0)
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
