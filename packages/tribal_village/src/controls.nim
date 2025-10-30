
import common, environment, windy, external_actions

var
  actionsArray*: array[MapAgents, array[2, uint8]]

proc simStep*() =
  # Get actions from the unified controller system
  actionsArray = getActions(env)
  env.step(addr actionsArray)

proc playerControlledStep*(playerAgentId: int, action: array[2, uint8]) =
  # Get actions for all agents first
  actionsArray = getActions(env)

  # Override the specific agent's action with player input
  actionsArray[playerAgentId] = action

  env.step(addr actionsArray)

proc agentControls*() =
  if selection != nil and selection.kind == Agent:
    let agent = selection

    if window.buttonPressed[KeyW] or window.buttonPressed[KeyUp]:
      playerControlledStep(agent.agentId, [1'u8, N.uint8])
    elif window.buttonPressed[KeyS] or window.buttonPressed[KeyDown]:
      playerControlledStep(agent.agentId, [1'u8, S.uint8])
    elif window.buttonPressed[KeyD] or window.buttonPressed[KeyRight]:
      playerControlledStep(agent.agentId, [1'u8, E.uint8])
    elif window.buttonPressed[KeyA] or window.buttonPressed[KeyLeft]:
      playerControlledStep(agent.agentId, [1'u8, W.uint8])
    elif window.buttonPressed[KeyQ]:
      playerControlledStep(agent.agentId, [1'u8, NW.uint8])
    elif window.buttonPressed[KeyE]:
      playerControlledStep(agent.agentId, [1'u8, NE.uint8])
    elif window.buttonPressed[KeyZ]:
      playerControlledStep(agent.agentId, [1'u8, SW.uint8])
    elif window.buttonPressed[KeyC]:
      playerControlledStep(agent.agentId, [1'u8, SE.uint8])

    if window.buttonPressed[KeyU]:
      let useDir = agent.orientation.uint8
      playerControlledStep(agent.agentId, [3'u8, useDir])

    if window.buttonPressed[KeyP]:
      playerControlledStep(agent.agentId, [8'u8, 0'u8])
