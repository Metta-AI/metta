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
  ## Send queued actions only when Python is requested and no action is pending.
  # Only process when Python is being requested (out of replay steps, etc)
  if not requestPython:
    return
  
  # Don't process if there's already an action pending (e.g. from WASD)
  if requestAction:
    return
  
  # Don't process if we already handled this step
  if step == currentProcessedStep:
    return
  
  # Process any agent with a queued action (not just selected)
  if actionQueue.len > 0:
    # Get the first queued action
    for agentId, action in actionQueue:
      # Python is being requested, add our action
      requestAction = true
      requestActionAgentId = action.agentId
      requestActionActionId = action.actionId
      requestActionArgument = action.argument
      actionQueue.del(agentId)
      currentProcessedStep = step
      break  # Only process one action per step

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
  ## Controls for agents. Auto-follows paths when playing or single stepping.

  # Auto-follow paths for all agents when playing or when Python is requested
  if (play or requestPython) and step != lastPathQueuedStep:
    for agentId, path in agentPaths:
      if path.len > 1:
        # Find the agent entity
        var agent: Entity = nil
        for obj in replay.objects:
          if obj.isAgent and obj.agentId == agentId:
            agent = obj
            break
        
        if agent != nil:
          let currentPos = agent.location.at(step).xy
          
          # Check if the agent is still on the expected path
          if path[0] == currentPos:
            if path.len > 1:
              let nextPos = path[1]
              let dx = nextPos.x - currentPos.x
              let dy = nextPos.y - currentPos.y
              let orientation = getOrientationFromDelta(dx.int, dy.int)
              queueAction(agentId, replay.moveActionId, orientation.int)
              agentPaths[agentId].delete(0)
            else:
              recomputePath(agentId, currentPos)
          else:
            recomputePath(agentId, currentPos)
    
    if agentPaths.len > 0:
      lastPathQueuedStep = step
  
  # Manual controls with WASD for selected agent - immediate response.
  if selection != nil and selection.isAgent:
    let agent = selection
    
    # Move
    if window.buttonPressed[KeyW] or window.buttonPressed[KeyUp]:
      sendAction(agent.agentId, replay.moveActionId, N.int)
      agentPaths.del(agent.agentId)
      agentDestinations.del(agent.agentId)
      lastPathQueuedStep = -1

    elif window.buttonPressed[KeyS] or window.buttonPressed[KeyDown]:
      sendAction(agent.agentId, replay.moveActionId, S.int)
      agentPaths.del(agent.agentId)
      agentDestinations.del(agent.agentId)
      lastPathQueuedStep = -1

    elif window.buttonPressed[KeyD] or window.buttonPressed[KeyRight]:
      sendAction(agent.agentId, replay.moveActionId, E.int)
      agentPaths.del(agent.agentId)
      agentDestinations.del(agent.agentId)
      lastPathQueuedStep = -1

    elif window.buttonPressed[KeyA] or window.buttonPressed[KeyLeft]:
      sendAction(agent.agentId, replay.moveActionId, W.int)
      agentPaths.del(agent.agentId)
      agentDestinations.del(agent.agentId)
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
