## Unified Action Interface for Agent Control
## Supports both external neural network control and built-in AI control
## Controller type is specified when creating the environment

import std/os, std/strutils
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

proc newBuiltinAIController*(seed: int = int(nowSeconds() * 1000)): AgentController =
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

proc initGlobalController*(controllerType: ControllerType, seed: int = int(nowSeconds() * 1000)) =
  ## Initialize the global controller with specified type
  case controllerType:
  of BuiltinAI:
    globalController = newBuiltinAIController(seed)
  of ExternalNN:
    # External callback will be set later via setExternalActionCallback
    globalController = AgentController(
      controllerType: ExternalNN,
      aiController: nil,
      externalActionCallback: nil
    )
    # Start automatic play mode for external controller
    play = true

proc setExternalActionCallback*(callback: proc(): array[MapAgents, array[2, uint8]]) =
  ## Set the external action callback for neural network control
  if globalController != nil and globalController.controllerType == ExternalNN:
    globalController.externalActionCallback = callback

proc getActions*(env: Environment): array[MapAgents, array[2, uint8]] =
  ## Get actions for all agents using the configured controller
  if globalController == nil:
    # NO CONTROLLER - return NOOP actions (agents won't move, proving Python control is required)
    var noopActions: array[MapAgents, array[2, uint8]]
    for i in 0..<MapAgents:
      noopActions[i] = [0'u8, 0'u8]  # NOOP action
    return noopActions
  
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
    if globalController.externalActionCallback != nil:
      let actions = globalController.externalActionCallback()
      return actions
    else:
      # Try to read actions from file (for Python neural network control across processes)
      let actionsFile = "actions.tmp"
      if fileExists(actionsFile):
        try:
          let content = readFile(actionsFile)
          let lines = content.replace("\r", "").replace("\n\n", "\n").split('\n')
          if lines.len >= MapAgents:
            var fileActions: array[MapAgents, array[2, uint8]]
            for i in 0..<MapAgents:
              if i < lines.len:
                let parts = lines[i].split(',')
                if parts.len >= 2:
                  fileActions[i][0] = parseInt(parts[0]).uint8
                  fileActions[i][1] = parseInt(parts[1]).uint8
                  
            # Delete the file after reading to avoid stale actions
            try:
              removeFile(actionsFile)
            except:
              discard  # Could not remove actions file
              
            return fileActions
        except Exception:
          discard  # Error reading actions file
      
      # FAIL HARD: ExternalNN controller configured but no actions available!
      echo "❌ FATAL ERROR: ExternalNN controller configured but no callback or actions file found!"
      echo "Python environment must call setExternalActionsFromPython() to provide actions!"
      raise newException(ValueError, "ExternalNN controller has no actions - Python communication failed!")

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