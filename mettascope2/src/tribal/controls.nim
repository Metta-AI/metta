
import std/times,
  common, environment, windy, ai

var
  actionsArray*: array[MapAgents, array[2, uint8]]
  agentController* = newController(seed = int(epochTime() * 1000))

proc simStep*() =
  for j, agent in env.agents:
    if selection != agent:
      # Use the controller to decide actions
      actionsArray[j] = agentController.decideAction(env, j)
  
  env.step(addr actionsArray)
  
  agentController.updateController()

proc agentControls*() =
  if selection != nil and selection.kind == Agent:
    let agent = selection

    if window.buttonPressed[KeyW] or window.buttonPressed[KeyUp]:
      actionsArray[agent.agentId] = [1, ord(N).uint8]
      simStep()
    elif window.buttonPressed[KeyS] or window.buttonPressed[KeyDown]:
      actionsArray[agent.agentId] = [1, ord(S).uint8]
      simStep()
    elif window.buttonPressed[KeyD] or window.buttonPressed[KeyRight]:
      actionsArray[agent.agentId] = [1, ord(E).uint8]
      simStep()
    elif window.buttonPressed[KeyA] or window.buttonPressed[KeyLeft]:
      actionsArray[agent.agentId] = [1, ord(W).uint8]
      simStep()
    elif window.buttonPressed[KeyQ]:
      actionsArray[agent.agentId] = [1, ord(NW).uint8]
      simStep()
    elif window.buttonPressed[KeyE]:
  
      actionsArray[agent.agentId] = [1, ord(NE).uint8]
      simStep()
    elif window.buttonPressed[KeyZ]:
      actionsArray[agent.agentId] = [1, ord(SW).uint8]
      simStep()
    elif window.buttonPressed[KeyC]:
      actionsArray[agent.agentId] = [1, ord(SE).uint8]
      simStep()

    if window.buttonPressed[KeyU]:
      let useDir = agent.orientation.uint8
      actionsArray[agent.agentId] = [3, useDir]
      simStep()

    if window.buttonPressed[KeyP]:
      actionsArray[agent.agentId] = [8, 0]
      simStep()
