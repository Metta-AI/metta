import
  std/[tables],
  replays, worldmap, common, panels, heatmap, heatmapshader


proc onReplayLoaded*() =
  ## Called when a replay is loaded.
  # Clear cached maps that depend on the old replay
  terrainMap = nil
  visibilityMap = nil
  worldHeatmap = nil

  # Reset global state for the new replay
  step = 0
  stepFloat = 0.0
  previousStep = -1
  selection = nil
  play = false
  requestPython = false
  agentPaths = initTable[int, seq[PathAction]]()
  agentObjectives = initTable[int, seq[Objective]]()

  # Initialize heatmap for the new replay
  worldHeatmap = newHeatmap(replay)
  worldHeatmap.initialize(replay)
  initHeatmapShader()
  echo "Heatmap initialized: ", worldHeatmap.width, "x", worldHeatmap.height, " (replay: ", replay.mapSize[0], "x", replay.mapSize[1], ")"

  needsInitialFit = true

  echo "Replay loaded: ", replay.fileName
