## Tribal Headless Server
## Standalone Nim server that runs the environment without graphics
## Communicates with Python through file-based IPC

import std/[os, times, json]
import tribal/[environment, external_actions]

# Communication files
const
  ActionsFile = "tribal_actions.json"
  StateFile = "tribal_state.json"
  ControlFile = "tribal_control.json"

# Global state
var
  env: Environment
  communicationActive = false

proc initEnvironment(): bool =
  ## Initialize the tribal environment
  try:
    echo "🌍 Initializing tribal environment (headless)"
    
    # Create default environment
    env = newEnvironment()
    
    # Set up for external control (no built-in AI)
    globalController = nil  # No controller - Python will provide actions
    
    echo "✅ Environment initialized with external control"
    return true
    
  except Exception as e:
    echo "❌ Failed to initialize environment: ", e.msg
    return false

proc writeStateToFile() =
  ## Write current environment state to file for Python
  try:
    var state = newJObject()
    
    # Get observations
    var obsArray = newJArray()
    for i in 0..<MapAgents:
      var agentObs = newJArray()
      for layer in 0..<ObservationLayers:
        var layerData = newJArray()
        for y in 0..<ObservationHeight:
          var rowData = newJArray()
          for x in 0..<ObservationWidth:
            rowData.add(newJInt(env.observations[i][layer][x][y].int))
          layerData.add(rowData)
        agentObs.add(layerData)
      obsArray.add(agentObs)
    
    # Get rewards  
    var rewardsArray = newJArray()
    for i in 0..<MapAgents:
      rewardsArray.add(newJFloat(env.agents[i].reward))
    
    # Get terminals/truncations
    var terminalsArray = newJArray()
    var truncationsArray = newJArray()
    for i in 0..<MapAgents:
      terminalsArray.add(newJBool(env.agents[i].frozen >= 999999))  # Agent is dead if frozen is very high
      truncationsArray.add(newJBool(env.currentStep >= env.config.maxSteps))
    
    # Environment info
    state["observations"] = obsArray
    state["rewards"] = rewardsArray
    state["terminals"] = terminalsArray
    state["truncations"] = truncationsArray
    state["current_step"] = newJInt(env.currentStep)
    state["max_steps"] = newJInt(env.config.maxSteps)
    state["episode_done"] = newJBool(env.currentStep >= env.config.maxSteps)
    state["timestamp"] = newJFloat(epochTime())
    
    writeFile(StateFile, $state)
    
  except Exception as e:
    echo "⚠️  Failed to write state file: ", e.msg

proc readActionsFromFile(): array[MapAgents, array[2, uint8]] =
  ## Read actions from file written by Python
  var actions: array[MapAgents, array[2, uint8]]
  
  # Default to NOOP actions
  for i in 0..<MapAgents:
    actions[i] = [0'u8, 0'u8]
  
  if not fileExists(ActionsFile):
    return actions
    
  try:
    let content = readFile(ActionsFile)
    let jsonData = parseJson(content)
    
    if jsonData.hasKey("actions"):
      let actionArray = jsonData["actions"]
      for i in 0..<min(actionArray.len, MapAgents):
        let agentAction = actionArray[i]
        if agentAction.len >= 2:
          actions[i][0] = agentAction[0].getInt().uint8
          actions[i][1] = agentAction[1].getInt().uint8
    
    # Remove file after reading to avoid stale actions
    removeFile(ActionsFile)
    
  except Exception as e:
    echo "⚠️  Failed to read actions file: ", e.msg
  
  return actions

proc checkCommunication(): bool =
  ## Check if Python is still communicating
  if not fileExists(ControlFile):
    return false
    
  try:
    let content = readFile(ControlFile)
    let control = parseJson(content)
    
    if control.hasKey("active") and control["active"].getBool():
      communicationActive = true
      return true
    elif control.hasKey("shutdown") and control["shutdown"].getBool():
      echo "🛑 Received shutdown signal from Python"
      return false
      
  except Exception as e:
    echo "⚠️  Failed to read control file: ", e.msg
  
  return communicationActive

proc runMainLoop() =
  ## Main game loop with file-based communication
  echo "🚀 Starting headless server main loop"
  echo "   Python writes actions to: ", ActionsFile
  echo "   Server writes state to: ", StateFile
  echo "   Control via: ", ControlFile
  
  # Reset environment
  env.reset()
  writeStateToFile()
  
  # Main loop for headless training
  while true:
    # Check for communication from Python
    if not checkCommunication():
      sleep(100)  # Wait 100ms before checking again
      continue
    
    # Read actions from Python (if available)
    let actions = readActionsFromFile()
    
    # Step environment
    env.step(actions.addr)
    
    # Write current state back to Python
    writeStateToFile()
    
    # Brief pause to prevent excessive CPU usage
    sleep(10)  # 10ms delay for ~100 FPS simulation rate

# Main entry point
proc main() =
  ## Main entry point for the headless server
  echo "🎮 Tribal Headless Server Starting"
  
  if not initEnvironment():
    echo "❌ Failed to initialize environment" 
    quit(1)
  
  # Create initial control file to signal readiness
  let controlData = %*{"ready": true, "active": false, "pid": getCurrentProcessId()}
  writeFile(ControlFile, $controlData)
  
  echo "🎯 Headless server ready - waiting for Python communication"
  echo "   Python should write to: ", ActionsFile
  echo "   Python should set active=true in: ", ControlFile
  
  try:
    runMainLoop()
  except Exception as e:
    echo "❌ Error in main loop: ", e.msg
  finally:
    echo "🧹 Cleaning up server"
    if fileExists(ControlFile):
      removeFile(ControlFile)
    if fileExists(StateFile):
      removeFile(StateFile)
    if fileExists(ActionsFile):
      removeFile(ActionsFile)

when isMainModule:
  main()