import
  genny, fidget2, openGL, jsony, vmath,
  ../src/mettascope, ../src/mettascope/[replays, common, worldmap]

type
  RenderResponse* = object
    shouldClose*: bool
    action*: bool
    actionAgentId*: int
    actionActionId*: int
    actionArgument*: int

proc init(dataDir: string, replay: string): RenderResponse =
  try:
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
    requestPython = false
    while true:
      if window.closeRequested:
        window.close()
        result.shouldClose = true
        return
      mainLoop()
      if requestPython:
        if requestAction:
          result.action = true
          result.actionAgentId = requestActionAgentId
          result.actionActionId = requestActionActionId
          result.actionArgument = requestActionArgument
          requestAction = false
          requestActionAgentId = 0
          requestActionActionId = 0
          requestActionArgument = 0
          return
        else:
          return
  except Exception:
    echo "############## Error rendering Mettascope ##################"
    echo getCurrentException().getStackTrace()
    echo getCurrentExceptionMsg()
    echo "############################################################"
    result.shouldClose = true
    return

exportObject RenderResponse:
  discard

exportProcs:
  init
  render

writeFiles("bindings/generated", "Mettascope2")

include generated/internal
