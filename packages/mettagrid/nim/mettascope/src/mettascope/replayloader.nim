import
  std/[tables, json],
  fidget2,
  vmath,
  replays, worldmap, common, timeline, envconfig, vibes

proc updateReplayHeader() =
  ## Set the global header's display name for the current session.
  var display = "Mettascope"

  if not common.replay.isNil:
    if common.replay.mgConfig != nil and common.replay.mgConfig.contains("label"):
      let node = common.replay.mgConfig["label"]
      if node.kind == JString:
        display = node.getStr
    if display == "Mettascope" and common.replay.fileName.len > 0:
      display = common.replay.fileName
  let titleNode = find("**/GlobalTitle")
  titleNode.text = display

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

  replay.loadImages()
  updateReplayHeader()
  worldMapPanel.pos = vec2(0, 0)
  onStepChanged()
  updateEnvConfig()
  updateVibePanel()

