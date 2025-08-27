import std/[random],
  common, windy

# var
#   actionsArray*: array[MapAgents, array[2, uint8]]

proc simStep*() =
  # Random actions
  discard
  # for j, agent in env.agents:
  #   if selection != agent:
  #     var action = rand(0 .. 9)
  #     var argument = 0
  #     if action == 1: # move
  #       argument = rand(0 .. 1)
  #     elif action == 2: # rotate
  #       argument = rand(0 .. 3)
  #     elif action == 7: # attack
  #       argument = rand(0 .. 9)
  #       #argument = 2
  #     actionsArray[j] = [action.uint8, argument.uint8]
  # env.nextStep(addr actionsArray)

proc agentControls*() =
  ## Controls for the selected agent.
  discard
  # if selection != nil and selection.kind == Agent:
  #   let agent = selection

  #   # Rotate
  #   if window.buttonPressed[KeyW] or window.buttonPressed[KeyUp]:
  #     if agent.orientation == N:
  #       actionsArray[agent.agentId] = [1, 1]
  #     else:
  #       actionsArray[agent.agentId] = [2, 0]
  #     simStep()
  #   elif window.buttonPressed[KeyS] or window.buttonPressed[KeyDown]:
  #     if agent.orientation == S:
  #       actionsArray[agent.agentId] = [1, 1]
  #     else:
  #       actionsArray[agent.agentId] = [2, 1]
  #     simStep()
  #   elif window.buttonPressed[KeyD] or window.buttonPressed[KeyRight]:
  #     if agent.orientation == Orientation.E:
  #       actionsArray[agent.agentId] = [1, 1]
  #     else:
  #       actionsArray[agent.agentId] = [2, 3]
  #     simStep()
  #   elif window.buttonPressed[KeyA] or window.buttonPressed[KeyLeft]:
  #     if agent.orientation == W:
  #       actionsArray[agent.agentId] = [1, 1]
  #     else:
  #       actionsArray[agent.agentId] = [2, 2]
  #     simStep()

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
