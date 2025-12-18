import
  std/[tables],
  replays, worldmap, common, panels


proc onReplayLoaded*() =
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

  # Position camera on visible part of the map when loading a replay to replace the current one.
  # this does not affect the first replay that gets loaded while worldMapZoomInfo is nil.
  if worldMapZoomInfo != nil:
    fitVisibleMap(worldMapZoomInfo)

  echo "Replay loaded: ", replay.fileName

