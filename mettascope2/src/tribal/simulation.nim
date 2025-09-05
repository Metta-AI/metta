import std/times,
  common, environment, windy, ai

var
  actionsArray*: array[MapAgents, array[2, uint8]]
  # Controller will use a random seed each time
  agentController* = newController(seed = int(epochTime() * 1000))

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

    # Direct movement with 8-way support
    if window.buttonPressed[KeyW] or window.buttonPressed[KeyUp]:
      # Move North
      actionsArray[agent.agentId] = [1, ord(N).uint8]
      simStep()
    elif window.buttonPressed[KeyS] or window.buttonPressed[KeyDown]:
      # Move South
      actionsArray[agent.agentId] = [1, ord(S).uint8]
      simStep()
    elif window.buttonPressed[KeyD] or window.buttonPressed[KeyRight]:
      # Move East
      actionsArray[agent.agentId] = [1, ord(E).uint8]
      simStep()
    elif window.buttonPressed[KeyA] or window.buttonPressed[KeyLeft]:
      # Move West
      actionsArray[agent.agentId] = [1, ord(W).uint8]
      simStep()
    # Add diagonal movement support
    elif window.buttonPressed[KeyQ]:
      # Move Northwest
      actionsArray[agent.agentId] = [1, ord(NW).uint8]
      simStep()
    elif window.buttonPressed[KeyE]:
      # Move Northeast  
      actionsArray[agent.agentId] = [1, ord(NE).uint8]
      simStep()
    elif window.buttonPressed[KeyZ]:
      # Move Southwest
      actionsArray[agent.agentId] = [1, ord(SW).uint8]
      simStep()
    elif window.buttonPressed[KeyC]:
      # Move Southeast
      actionsArray[agent.agentId] = [1, ord(SE).uint8]
      simStep()

    # Use - face current direction of agent
    if window.buttonPressed[KeyU]:
      # Use in the direction the agent is facing
      let useDir = agent.orientation.uint8
      actionsArray[agent.agentId] = [3, useDir]
      simStep()

    # Swap (still valid - swaps positions with frozen agents)
    if window.buttonPressed[KeyP]:
      actionsArray[agent.agentId] = [8, 0]
      simStep()
