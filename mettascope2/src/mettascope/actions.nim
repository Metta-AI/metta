import std/[random],
  common, tribal, windy, controller

var
  actionsArray*: array[MapAgents, array[2, uint8]]
  agentController* = newController(seed = 2024)

proc simStep*() =
  # Use controller for agent actions
  for j, agent in env.agents:
    if selection != agent:
      # Use the controller to decide actions
      actionsArray[j] = agentController.decideAction(env, j)
    # else: selected agent uses manual controls
  
  # Step the environment (this handles mines, clippys, etc.)
  env.step(addr actionsArray)
  
  # Update controller state
  agentController.updateController()

proc agentControls*() =
  ## Controls for the selected agent.
  if selection != nil and selection.kind == Agent:
    let agent = selection

    # Direct movement with auto-rotation
    if window.buttonPressed[KeyW] or window.buttonPressed[KeyUp]:
      # Move North
      actionsArray[agent.agentId] = [1, 0]
      simStep()
    elif window.buttonPressed[KeyS] or window.buttonPressed[KeyDown]:
      # Move South
      actionsArray[agent.agentId] = [1, 1]
      simStep()
    elif window.buttonPressed[KeyD] or window.buttonPressed[KeyRight]:
      # Move East
      actionsArray[agent.agentId] = [1, 2]
      simStep()
    elif window.buttonPressed[KeyA] or window.buttonPressed[KeyLeft]:
      # Move West
      actionsArray[agent.agentId] = [1, 3]
      simStep()

    # Use - face current direction of agent
    if window.buttonPressed[KeyU]:
      # Use in the direction the agent is facing
      let useDir = agent.orientation.uint8
      actionsArray[agent.agentId] = [3, useDir]
      simStep()

    # Shield
    if window.buttonPressed[KeyO]:
      actionsArray[agent.agentId] = [6, 0]
      simStep()

    # Swap
    if window.buttonPressed[KeyP]:
      actionsArray[agent.agentId] = [8, 0]
      simStep()

    # Attack
    if window.buttonPressed[Key1]:
      actionsArray[agent.agentId] = [4, 1]
      simStep()
    if window.buttonPressed[Key2]:
      actionsArray[agent.agentId] = [4, 2]
      simStep()
    if window.buttonPressed[Key3]:
      actionsArray[agent.agentId] = [4, 3]
      simStep()
    if window.buttonPressed[Key4]:
      actionsArray[agent.agentId] = [4, 4]
      simStep()
    if window.buttonPressed[Key5]:
      actionsArray[agent.agentId] = [4, 5]
      simStep()
    if window.buttonPressed[Key6]:
      actionsArray[agent.agentId] = [4, 6]
      simStep()
    if window.buttonPressed[Key7]:
      actionsArray[agent.agentId] = [4, 7]
      simStep()
    if window.buttonPressed[Key8]:
      actionsArray[agent.agentId] = [4, 8]
      simStep()
    if window.buttonPressed[Key9]:
      actionsArray[agent.agentId] = [4, 9]
      simStep()
