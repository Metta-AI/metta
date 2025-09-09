## Unified Action Interface for Agent Control
## Supports both external neural network control and built-in AI control
## Controller type is specified when creating the environment

import std/times
import environment, ai, common

type
  ControllerType* = enum
    BuiltinAI,      # Use built-in Nim AI controller
    ExternalNN      # Use external neural network (Python)

  AgentController* = ref object
    controllerType*: ControllerType
    # Built-in AI controller (when using BuiltinAI)
    aiController*: Controller
    # External action callback (when using ExternalNN)
    externalActionCallback*: proc(): array[MapAgents, array[2, uint8]]

# Global agent controller instance
var globalController*: AgentController

proc newBuiltinAIController*(seed: int = int(epochTime() * 1000)): AgentController =
  ## Create a new controller using built-in AI
  AgentController(
    controllerType: BuiltinAI,
    aiController: newController(seed),
    externalActionCallback: nil
  )

proc newExternalNNController*(actionCallback: proc(): array[MapAgents, array[2, uint8]]): AgentController =
  ## Create a new controller using external neural network
  AgentController(
    controllerType: ExternalNN,
    aiController: nil,
    externalActionCallback: actionCallback
  )

proc initGlobalController*(controllerType: ControllerType, seed: int = int(epochTime() * 1000)) =
  ## Initialize the global controller with specified type
  case controllerType:
  of BuiltinAI:
    globalController = newBuiltinAIController(seed)
    echo "🤖 Debug: Initialized BuiltinAI controller"
  of ExternalNN:
    # External callback will be set later via setExternalActionCallback
    globalController = AgentController(
      controllerType: ExternalNN,
      aiController: nil,
      externalActionCallback: nil
    )
    echo "🔗 Debug: Initialized ExternalNN controller"
    # Start automatic play mode for external controller
    play = true
    echo "🎮 Debug: Enabled automatic play mode for external controller"

proc setExternalActionCallback*(callback: proc(): array[MapAgents, array[2, uint8]]) =
  ## Set the external action callback for neural network control
  if globalController != nil and globalController.controllerType == ExternalNN:
    globalController.externalActionCallback = callback

proc getActions*(env: Environment): array[MapAgents, array[2, uint8]] =
  ## Get actions for all agents using the configured controller
  if globalController == nil:
    # Default to built-in AI if not initialized
    initGlobalController(BuiltinAI)
    echo "🔧 Debug: Initialized default BuiltinAI controller"
  
  case globalController.controllerType:
  of BuiltinAI:
    # Use built-in AI controller
    var actions: array[MapAgents, array[2, uint8]]
    for i, agent in env.agents:
      actions[i] = globalController.aiController.decideAction(env, i)
    globalController.aiController.updateController()
    return actions
    
  of ExternalNN:
    # Use external neural network callback
    echo "🎯 Debug: Using ExternalNN controller"
    if globalController.externalActionCallback != nil:
      echo "📡 Debug: Calling external callback"
      let actions = globalController.externalActionCallback()
      # Log sample actions
      echo "🎮 Debug: Got external actions - Agent 0: [", actions[0][0], ",", actions[0][1], "], Agent 1: [", actions[1][0], ",", actions[1][1], "]"
      return actions
    else:
      echo "⚠️ Debug: No external callback available, using NOOP"
      # Try to read actions from file (for Python neural network control)
      # Note: This requires the readActionsFromFile function to be available
      # For now, fallback to noop actions
      var noopActions: array[MapAgents, array[2, uint8]]
      for i in 0..<MapAgents:
        noopActions[i] = [0'u8, 0'u8]  # noop
      return noopActions

proc getControllerType*(): ControllerType =
  ## Get the current controller type
  if globalController != nil:
    return globalController.controllerType
  else:
    return BuiltinAI  # Default

proc isExternalControllerActive*(): bool =
  ## Check if external neural network controller is active
  return globalController != nil and 
         globalController.controllerType == ExternalNN and
         globalController.externalActionCallback != nil