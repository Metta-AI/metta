import std/[random, times], vmath, windy
import common, game, controller

# Action system for managing agent actions
type
  ActionSystem* = object
    actionsArray*: array[MapAgents, array[2, uint8]]
    agentController*: Controller

proc newActionSystem*(seed: int = int(epochTime() * 1000)): ActionSystem =
  ## Create a new action system with a controller
  result.agentController = newController(seed = seed)

proc simStep*(actionSys: var ActionSystem, env: var Environment, selection: Thing) =
  ## Perform one simulation step with controller decisions
  for j, agent in env.agents:
    if selection != agent:
      # Use the controller to decide actions
      actionSys.actionsArray[j] = actionSys.agentController.decideAction(env, j)
    # else: selected agent uses manual controls
  
  # Step the environment (this handles mines, clippys, etc.)
  env.step(addr actionSys.actionsArray)
  
  # Update controller state
  actionSys.agentController.updateController()

proc handleAgentControls*(actionSys: var ActionSystem, env: var Environment, 
                         selection: Thing, window: Window, simStepCallback: proc()) =
  ## Handle manual controls for the selected agent
  if selection != nil and selection.kind == Agent:
    let agent = selection

    # Direct movement with auto-rotation
    if window.buttonPressed[KeyW] or window.buttonPressed[KeyUp]:
      # Move North
      actionSys.actionsArray[agent.agentId] = [1, 0]
      simStepCallback()
    elif window.buttonPressed[KeyS] or window.buttonPressed[KeyDown]:
      # Move South
      actionSys.actionsArray[agent.agentId] = [1, 1]
      simStepCallback()
    elif window.buttonPressed[KeyD] or window.buttonPressed[KeyRight]:
      # Move East
      actionSys.actionsArray[agent.agentId] = [1, 2]
      simStepCallback()
    elif window.buttonPressed[KeyA] or window.buttonPressed[KeyLeft]:
      # Move West
      actionSys.actionsArray[agent.agentId] = [1, 3]
      simStepCallback()

    # Use - face current direction of agent
    if window.buttonPressed[KeyU]:
      # Use in the direction the agent is facing
      let useDir = agent.orientation.uint8
      actionSys.actionsArray[agent.agentId] = [3, useDir]
      simStepCallback()

    # Swap (still valid - swaps positions with frozen agents)
    if window.buttonPressed[KeyP]:
      actionSys.actionsArray[agent.agentId] = [8, 0]
      simStepCallback()
