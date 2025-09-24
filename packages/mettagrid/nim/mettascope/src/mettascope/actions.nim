import std/[random],
  windy, fidget2,
  common, replays

# var
#   actionsArray*: array[MapAgents, array[2, uint8]]
type
  Orientation* = enum
    N = 0
    S = 1
    W = 2
    E = 3

proc simStep*() =
  # Random actions
  discard
  # for j, agent in env.agents:
  #   if selection != agent:
  #     var action = rand(0 .. 9)
  #     var argument = 0
  #     if action == 1: # mov
  #       argument = rand(0 .. 1)
  #     elif action == 2: # rotate
  #       argument = rand(0 .. 3)
  #     elif action == 7: # attack
  #       argument = rand(0 .. 9)
  #       #argument = 2
  #     actionsArray[j] = [action.uint8, argument.uint8]
  # env.nextStep(addr actionsArray)

proc sendAction*(agentId, actionId, argument: int) =
  echo replay.actionNames
  echo "Sending action: ", agentId, " ", actionId, " ", argument
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

  #   # Move
  #   if window.buttonPressed[KeyE]:
  #     actionsArray[agent.agentId] = [1, 0]
  #     simStep()
  #   elif window.buttonPressed[KeyQ]:
  #     actionsArray[agent.agentId] = [1, 1]
  #     simStep()

  #   # Use
  #   if window.buttonPressed[KeyU]:
  #     actionsArray[agent.agentId] = [3, 0]
  #     simStep()

  #   # Shield
  #   if window.buttonPressed[KeyO]:
  #     actionsArray[agent.agentId] = [6, 0]
  #     simStep()

  #   # Swap
  #   if window.buttonPressed[KeyP]:
  #     actionsArray[agent.agentId] = [8, 0]
  #     simStep()

  #   # Attack
  #   if window.buttonPressed[Key1]:
  #     actionsArray[agent.agentId] = [4, 1]
  #     simStep()
  #   if window.buttonPressed[Key2]:
  #     actionsArray[agent.agentId] = [4, 2]
  #     simStep()
  #   if window.buttonPressed[Key3]:
  #     actionsArray[agent.agentId] = [4, 3]
  #     simStep()
  #   if window.buttonPressed[Key4]:
  #     actionsArray[agent.agentId] = [4, 4]
  #     simStep()
  #   if window.buttonPressed[Key5]:
  #     actionsArray[agent.agentId] = [4, 5]
  #     simStep()
  #   if window.buttonPressed[Key6]:
  #     actionsArray[agent.agentId] = [4, 6]
  #     simStep()
  #   if window.buttonPressed[Key7]:
  #     actionsArray[agent.agentId] = [4, 7]
  #     simStep()
  #   if window.buttonPressed[Key8]:
  #     actionsArray[agent.agentId] = [4, 8]
  #     simStep()
  #   if window.buttonPressed[Key9]:
  #     actionsArray[agent.agentId] = [4, 9]
  #     simStep()
