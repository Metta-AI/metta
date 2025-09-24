import std/[random],
  windy, fidget2,
  common, replays

type
  Orientation* = enum
    N = 0
    S = 1
    W = 2
    E = 3

proc sendAction*(agentId, actionId, argument: int) =
  ## Send an action to the Python from the user.
  requestAction = true
  requestActionAgentId = agentId
  requestActionActionId = actionId
  requestActionArgument = argument
  requestPython = true

proc agentControls*() =
  ## Controls for the selected agent.

  let noopId = replay.actionNames.find("noop")
  let moveId = replay.actionNames.find("move")
  let putItemsId = replay.actionNames.find("put_items")
  let getItemsId = replay.actionNames.find("get_items")
  let attackId = replay.actionNames.find("attack")

  if selection != nil and selection.isAgent:
    let agent = selection

    # Move
    if window.buttonPressed[KeyW] or window.buttonPressed[KeyUp]:
      sendAction(agent.agentId, moveId, N.int)

    elif window.buttonPressed[KeyS] or window.buttonPressed[KeyDown]:
      sendAction(agent.agentId, moveId, S.int)

    elif window.buttonPressed[KeyD] or window.buttonPressed[KeyRight]:
      sendAction(agent.agentId, moveId, E.int)

    elif window.buttonPressed[KeyA] or window.buttonPressed[KeyLeft]:
      sendAction(agent.agentId, moveId, W.int)

    # Put items
    if window.buttonPressed[KeyQ]:
      sendAction(agent.agentId, putItemsId, 0)

    # Get items
    if window.buttonPressed[KeyE]:
      sendAction(agent.agentId, getItemsId, 0)

    # Attack
    if window.buttonPressed[KeyZ]:
      # TODO: Get implementation attack selection ui.
      sendAction(agent.agentId, attackId, 0)

    # Noop
    if window.buttonPressed[KeyX]:
      sendAction(agent.agentId, noopId, 0)
