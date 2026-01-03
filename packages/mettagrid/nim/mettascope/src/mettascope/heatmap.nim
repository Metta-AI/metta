import
  std/[math],
  vmath,
  replays, common

proc newHeatmap*(replay: Replay): Heatmap =
  ## Create a new heatmap for the given replay.
  result = Heatmap()
  result.width = replay.mapSize[0]
  result.height = replay.mapSize[1]
  result.maxSteps = replay.maxSteps
  result.maxHeat = newSeq[int](replay.maxSteps)
  result.data = newSeq[seq[int]](replay.maxSteps)
  result.currentTextureStep = -1
  for step in 0 ..< replay.maxSteps:
    result.data[step] = newSeq[int](replay.mapSize[0] * replay.mapSize[1])

proc initialize*(heatmap: Heatmap, replay: Replay) =
  ## Initialize the heatmap for every step in the replay.
  ## For each step, start with previous step's cumulative values, then add agent positions.

  for step in 0 ..< replay.maxSteps:
    # Start with previous step's cumulative values (if not the first step).
    if step > 0:
      for i in 0 ..< heatmap.data[step].len:
        heatmap.data[step][i] = heatmap.data[step - 1][i]

    # Add agent positions for this step.
    for agent in replay.agents:
      let location = agent.location.at(step)
      let x = location.x.int
      let y = location.y.int

      if x >= 0 and x < heatmap.width and y >= 0 and y < heatmap.height:
        heatmap.data[step][y * heatmap.width + x] += 1

    # Calculate max heat for this step.
    var maxHeat = 0
    for heat in heatmap.data[step]:
      if heat > maxHeat:
        maxHeat = heat
    heatmap.maxHeat[step] = maxHeat

proc update*(heatmap: Heatmap, step: int, replay: Replay) =
  ## Update the heatmap for a new step in realtime mode.
  ## Only expand and add heat when step >= maxSteps (new step).

  if step < heatmap.maxSteps:
    # Step already processed, nothing to do.
    return

  # Expand the data structure for the new step.
  let oldMaxSteps = heatmap.maxSteps
  heatmap.maxSteps = step + 1

  # Add new step data arrays.
  for i in oldMaxSteps ..< heatmap.maxSteps:
    heatmap.data.add(newSeq[int](heatmap.width * heatmap.height))

  # Expand maxHeat array.
  heatmap.maxHeat.setLen(heatmap.maxSteps)

  # Copy previous step's data to new steps.
  for i in oldMaxSteps ..< heatmap.maxSteps:
    if i > 0:
      for j in 0 ..< heatmap.data[i].len:
        heatmap.data[i][j] = heatmap.data[i - 1][j]

  # Update the cumulative heatmap for every agent at this step.
  for agent in replay.agents:
    let location = agent.location.at(step)
    let x = location.x.int
    let y = location.y.int

    if x >= 0 and x < heatmap.width and y >= 0 and y < heatmap.height:
      heatmap.data[step][y * heatmap.width + x] += 1

  # Calculate max heat for this step.
  var maxHeat = 0
  for heat in heatmap.data[step]:
    if heat > maxHeat:
      maxHeat = heat
  heatmap.maxHeat[step] = maxHeat

proc getHeat*(heatmap: Heatmap, step: int, x: int, y: int): int =
  ## Get the heat value at the given position and step.
  if step < 0 or step >= heatmap.maxSteps or x < 0 or x >= heatmap.width or y < 0 or y >= heatmap.height:
    return 0
  heatmap.data[step][y * heatmap.width + x]

proc getMaxHeat*(heatmap: Heatmap, step: int): int =
  ## Get the maximum heat value for the given step.
  if step < 0 or step >= heatmap.maxSteps:
    return 0
  heatmap.maxHeat[step]
