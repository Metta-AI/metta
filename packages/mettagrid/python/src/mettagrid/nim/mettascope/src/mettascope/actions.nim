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

proc queueAction(agentId, actionId, argument: int) =
  ## Queue an action for the agent. Will be sent on next step.
  actionQueue[agentId] = QueuedAction(
    agentId: agentId,
    actionId: actionId,
    argument: argument
  )

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
  ## Process pathfinding and action queue. Called on step change.
  if not (play or requestPython):
    return
  
  for agentId, path in agentPaths:
    if path.len > 1:
      var agent: Entity = nil
      for obj in replay.objects:
        if obj.isAgent and obj.agentId == agentId:
          agent = obj
          break
      
      if agent != nil:
        let currentPos = agent.location.at(step).xy
        
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
  
  if actionQueue.len > 0:
    for agentId, action in actionQueue:
      requestActions.add(ActionRequest(
        agentId: action.agentId,
        actionId: action.actionId,
        argument: action.argument
      ))
    actionQueue.clear()

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
