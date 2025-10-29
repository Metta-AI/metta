import
  std/[tables],
  windy, fidget2, vmath,
  common, replays, pathfinding

type
  Orientation* = enum
    N = 0
    S = 1
    W = 2
    E = 3

proc sendAction*(agentId, actionId: int) =
  ## Send an action to the Python from the user.
  requestActions.add(ActionRequest(
    agentId: agentId,
    actionId: actionId
  ))
  requestPython = true

proc getOrientationFromDelta(dx, dy: int): Orientation =
  ## Get the orientation from a movement delta.
  if dx == 0 and dy == -1:
    return N
  elif dx == 0 and dy == 1:
    return S
  elif dx == -1 and dy == 0:
    return W
  elif dx == 1 and dy == 0:
    return E
  else:
    return N

proc getMoveActionId(orientation: Orientation): int =
  ## Get the action ID for a move action in the given orientation.
  case orientation
  of N:
    return replay.moveNorthActionId
  of S:
    return replay.moveSouthActionId
  of W:
    return replay.moveWestActionId
  of E:
    return replay.moveEastActionId

proc agentHasEnergy(agent: Entity): bool =
  let energyId = replay.itemNames.find("energy")
  if energyId == -1:
    echo "Energy item not found in replay"
    return true
  let inv = agent.inventory.at(step)
  for item in inv:
    if item.itemId == energyId and item.count > 1:
      return true
  return false

proc processActions*() =
  ## Process path actions and send actions for the current step while in play mode.
  if not (play or requestPython):
    return
  var agentIds: seq[int] = @[]
  for agentId in agentPaths.keys:
    agentIds.add(agentId)

  for agentId in agentIds:
    if not agentPaths.hasKey(agentId):
      continue

    let agent = getAgentById(agentId)
    let currentPos = agent.location.at(step).xy
    let pathActions = agentPaths[agentId]

    if pathActions.len == 0:
      agentPaths.del(agentId)
      continue

    let nextAction = pathActions[0]

    # If the agent has no energy, wait and do not issue new path actions this step.
    if not agentHasEnergy(agent):
      continue

    case nextAction.kind
    of Move:
      # Execute movement action.
      let dx = nextAction.pos.x - currentPos.x
      let dy = nextAction.pos.y - currentPos.y
      let orientation = getOrientationFromDelta(dx.int, dy.int)
      sendAction(agentId, getMoveActionId(orientation))
      # Remove this action from the queue.
      agentPaths[agentId].delete(0)
      # Check if we completed an objective.
      let objective = agentObjectives[agentId][0]
      if objective.kind == Move and nextAction.pos == objective.pos:
        # Completed this Move objective.
        agentObjectives[agentId].delete(0)
        if objective.repeat:
          # Re-queue this objective at the end.
          agentObjectives[agentId].add(objective)
          recomputePath(agentId, nextAction.pos)
        elif agentObjectives[agentId].len == 0:
          # No more objectives, clear the path.
          agentPaths.del(agentId)
    of Bump:
      # Execute bump action.
      let targetOrientation = getOrientationFromDelta(nextAction.bumpDir.x.int, nextAction.bumpDir.y.int)
      sendAction(agentId, getMoveActionId(targetOrientation))
      # Remove this action from the queue.
      agentPaths[agentId].delete(0)
      # Remove the corresponding objective.
      let objective = agentObjectives[agentId][0]
      agentObjectives[agentId].delete(0)
      if objective.repeat:
        # Re-queue this objective at the end.
        agentObjectives[agentId].add(objective)
        recomputePath(agentId, currentPos)
      elif agentObjectives[agentId].len == 0:
        # No more objectives, clear the path.
        agentPaths.del(agentId)
    of Vibe:
      # Execute vibe.
      sendAction(agentId, nextAction.vibeActionId)
      # Remove this action from the queue.
      agentPaths[agentId].delete(0)
      # Remove the corresponding objective.
      let objective = agentObjectives[agentId][0]
      if objective.kind == Vibe:
        agentObjectives[agentId].delete(0)
        if objective.repeat:
          # Re-queue this objective at the end.
          agentObjectives[agentId].add(objective)
          recomputePath(agentId, currentPos)
        elif agentObjectives[agentId].len == 0:
          # No more objectives, clear the path.
          agentPaths.del(agentId)

proc agentControls*() =
  ## Manual controls with WASD for selected agent.
  if selection != nil and selection.isAgent:
    let agent = selection

    # Move
    if window.buttonPressed[KeyW] or window.buttonPressed[KeyUp]:
      sendAction(agent.agentId, getMoveActionId(N))
      clearPath(agent.agentId)

    elif window.buttonPressed[KeyS] or window.buttonPressed[KeyDown]:
      sendAction(agent.agentId, getMoveActionId(S))
      clearPath(agent.agentId)

    elif window.buttonPressed[KeyD] or window.buttonPressed[KeyRight]:
      sendAction(agent.agentId, getMoveActionId(E))
      clearPath(agent.agentId)

    elif window.buttonPressed[KeyA] or window.buttonPressed[KeyLeft]:
      sendAction(agent.agentId, getMoveActionId(W))
      clearPath(agent.agentId)

    # Put items
    elif window.buttonPressed[KeyQ]:
      sendAction(agent.agentId, replay.putItemsActionId)

    # Get items
    elif window.buttonPressed[KeyE]:
      sendAction(agent.agentId, replay.getItemsActionId)

    # Attack
    elif window.buttonPressed[KeyZ]:
      # TODO: Get implementation attack selection ui.
      sendAction(agent.agentId, replay.attackActionId)

    # Noop
    elif window.buttonPressed[KeyX]:
      sendAction(agent.agentId, replay.noopActionId)
