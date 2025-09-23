import
  genny, fidget2, openGL, jsony, vmath,
  ../src/mettascope, ../src/mettascope/[replays, common]

proc init(dataDir: string, replay: string): bool =
  try:
    #echo "Replay from python: ", replay
    echo "Data dir: ", dataDir
    common.replay = loadReplayString(replay, "PlayTool")
    initFidget(
      figmaUrl = "https://www.figma.com/design/hHmLTy7slXTOej6opPqWpz/MetaScope-V2-Rig",
      windowTitle = "MetaScope V2",
      entryFrame = "UI/Main",
      windowStyle = DecoratedResizable,
      dataDir = dataDir
    )
    return false
  except Exception:
    echo "Error initializing Mettascope2: ", getCurrentExceptionMsg()
    return true

proc render(currentStep: int, replayStep: string): bool =
  try:
    echo "Current step from python: ", currentStep
    common.replay.apply(replayStep)
    step = currentStep
    stepFloat = currentStep.float32
    requestPython = false
    while true:
      if window.closeRequested:
        window.close()
        return true
      mainLoop()
      if requestPython:
        echo "Requesting Python, breaking loop"
        return false
  except Exception:
    echo "Error rendering Mettascope2: ", getCurrentExceptionMsg()
    return true

exportProcs:
  init
  render

writeFiles("bindings/generated", "Mettascope2")

include generated/internal
