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

proc sendAction*(agentId, actionId, argument: int) =
  ## Send an action to the Python from the user.
  requestActions.add(ActionRequest(
    agentId: agentId,
    actionId: actionId,
    argument: argument
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

proc processActions*() =
  ## Process pathfinding and send actions for the current step while in play mode.
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

    case nextAction.actionType
    of PathMove:
      # Execute movement action.
      let dx = nextAction.pos.x - currentPos.x
      let dy = nextAction.pos.y - currentPos.y
      let orientation = getOrientationFromDelta(dx.int, dy.int)
      sendAction(agentId, replay.moveActionId, orientation.int)
      # Remove this action from the queue.
      agentPaths[agentId].delete(0)
      # Check if we completed a destination.
      if agentDestinations.hasKey(agentId) and agentDestinations[agentId].len > 0:
        let dest = agentDestinations[agentId][0]
        if dest.destinationType == Move and nextAction.pos == dest.pos:
          # Completed this Move destination.
          agentDestinations[agentId].delete(0)
          if dest.repeat:
            # Re-queue this destination at the end.
            agentDestinations[agentId].add(dest)
            recomputePath(agentId, nextAction.pos)
    of PathBump:
      # Execute bump action.
      let targetOrientation = getOrientationFromDelta(nextAction.bumpDir.x.int, nextAction.bumpDir.y.int)
      sendAction(agentId, replay.moveActionId, targetOrientation.int)
      # Remove this action from the queue.
      agentPaths[agentId].delete(0)
      # Remove the corresponding destination.
      if agentDestinations.hasKey(agentId) and agentDestinations[agentId].len > 0:
        let dest = agentDestinations[agentId][0]
        agentDestinations[agentId].delete(0)
        if dest.repeat:
          # Re-queue this destination at the end.
          agentDestinations[agentId].add(dest)
          recomputePath(agentId, currentPos)

proc agentControls*() =
  ## Manual controls with WASD for selected agent.
  if selection != nil and selection.isAgent:
    let agent = selection

    # Move
    if window.buttonPressed[KeyW] or window.buttonPressed[KeyUp]:
      sendAction(agent.agentId, replay.moveActionId, N.int)
      clearPath(agent.agentId)

    elif window.buttonPressed[KeyS] or window.buttonPressed[KeyDown]:
      sendAction(agent.agentId, replay.moveActionId, S.int)
      clearPath(agent.agentId)

    elif window.buttonPressed[KeyD] or window.buttonPressed[KeyRight]:
      sendAction(agent.agentId, replay.moveActionId, E.int)
      clearPath(agent.agentId)

    elif window.buttonPressed[KeyA] or window.buttonPressed[KeyLeft]:
      sendAction(agent.agentId, replay.moveActionId, W.int)
      clearPath(agent.agentId)

    # Put items
    elif window.buttonPressed[KeyQ]:
      sendAction(agent.agentId, replay.putItemsActionId, 0)

    # Get items
    elif window.buttonPressed[KeyE]:
      sendAction(agent.agentId, replay.getItemsActionId, 0)

    # Attack
    elif window.buttonPressed[KeyZ]:
      # TODO: Get implementation attack selection ui.
      sendAction(agent.agentId, replay.attackActionId, 0)

    # Noop
    elif window.buttonPressed[KeyX]:
      sendAction(agent.agentId, replay.noopActionId, 0)
