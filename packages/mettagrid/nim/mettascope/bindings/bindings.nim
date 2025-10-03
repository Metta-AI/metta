import
  genny, fidget2, openGL, jsony, vmath,
  ../src/mettascope, ../src/mettascope/[replays, common, worldmap, timeline,
  envconfig]

type
  ActionRequest* = object
    agentId*: int
    actionId*: int
    argument*: int

  RenderResponse* = ref object
    shouldClose*: bool
    actions*: seq[ActionRequest]

proc init(dataDir: string, replay: string): RenderResponse =
  try:
    result = RenderResponse(shouldClose: false, actions: @[])
    #echo "Replay from python: ", replay
    echo "Data dir: ", dataDir
    playMode = Realtime
    initFidget(
      figmaUrl = "https://www.figma.com/design/hHmLTy7slXTOej6opPqWpz/MetaScope-V2-Rig",
      windowTitle = "MetaScope V2",
      entryFrame = "UI/Main",
      windowStyle = DecoratedResizable,
      dataDir = dataDir
    )
    common.replay = loadReplayString(replay, "MettaScope")
    updateEnvConfig()
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
    onStepChanged()
    requestPython = false
    result = RenderResponse(shouldClose: false, actions: @[])
    while true:
      if window.closeRequested:
        window.close()
        result.shouldClose = true
        return
      mainLoop()
      if requestPython:
        onRequestPython()
        for action in requestActions:
          result.actions.add(ActionRequest(
            agentId: action.agentId,
            actionId: action.actionId,
            argument: action.argument
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

writeFiles("bindings/generated", "Mettascope2")

include generated/internal
