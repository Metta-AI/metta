import
  std/[tables],
  windy, fidget2,
  common, replays

type
  Orientation* = enum
    N = 0
    S = 1
    W = 2
    E = 3
  
  QueuedAction = object
    agentId: int
    actionId: int
    argument: int

var
  ## Action queue for each agent. Only one action per step.
  actionQueue = initTable[int, QueuedAction]()
  ## Track the step we're currently processing.
  currentProcessedStep = -1
  ## Track which step we last queued a path action at.
  lastPathQueuedStep = -1

proc queueAction(agentId, actionId, argument: int) =
  ## Queue an action for the agent. Will be sent on next step.
  actionQueue[agentId] = QueuedAction(
    agentId: agentId,
    actionId: actionId,
    argument: argument
  )

proc sendAction*(agentId, actionId, argument: int) =
  ## Send an action to the Python from the user.
  requestAction = true
  requestActionAgentId = agentId
  requestActionActionId = actionId
  requestActionArgument = argument
  requestPython = true

proc processActionQueue*() =
  ## Send queued actions only when Python is naturally requested by timeline.
  # Only process when timeline naturally requests Python (out of replay steps)
  if not requestPython:
    return
  
  # Don't process if we already handled this
  if step == currentProcessedStep:
    return
  
  if selection != nil and selection.isAgent:
    let agentId = selection.agentId
    
    if actionQueue.hasKey(agentId):
      let action = actionQueue[agentId]
      # Timeline already set requestPython=true, now add our action
      requestAction = true
      requestActionAgentId = action.agentId
      requestActionActionId = action.actionId
      requestActionArgument = action.argument
      actionQueue.del(agentId)
      currentProcessedStep = step

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

proc agentControls*() =
  ## Controls for the selected agent. Auto-follows path when playing.

  if selection != nil and selection.isAgent:
    let agent = selection
    
    # Auto-follow path when playing - only queue once per step
    if play and agentPaths.hasKey(agent.agentId) and agentPaths[agent.agentId].len > 1:
      # Only queue if this is a new step we haven't processed yet
      if step != lastPathQueuedStep:
        let currentPos = agent.location.at(step).xy
        let path = agentPaths[agent.agentId]
        
        if path[0] == currentPos and path.len > 1:
          let nextPos = path[1]
          let dx = nextPos.x - currentPos.x
          let dy = nextPos.y - currentPos.y
          let orientation = getOrientationFromDelta(dx.int, dy.int)
          queueAction(agent.agentId, replay.moveActionId, orientation.int)
          agentPaths[agent.agentId].delete(0)
          lastPathQueuedStep = step
      return
    
    # Manual controls with WASD - immediate response.
    if window.buttonPressed[KeyW] or window.buttonPressed[KeyUp]:
      sendAction(agent.agentId, replay.moveActionId, N.int)
      agentPaths.del(agent.agentId)
      lastPathQueuedStep = -1

    elif window.buttonPressed[KeyS] or window.buttonPressed[KeyDown]:
      sendAction(agent.agentId, replay.moveActionId, S.int)
      agentPaths.del(agent.agentId)
      lastPathQueuedStep = -1

    elif window.buttonPressed[KeyD] or window.buttonPressed[KeyRight]:
      sendAction(agent.agentId, replay.moveActionId, E.int)
      agentPaths.del(agent.agentId)
      lastPathQueuedStep = -1

    elif window.buttonPressed[KeyA] or window.buttonPressed[KeyLeft]:
      sendAction(agent.agentId, replay.moveActionId, W.int)
      agentPaths.del(agent.agentId)
      lastPathQueuedStep = -1

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
