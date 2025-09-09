## Example showing how to use the tribal replay system
## This demonstrates recording, saving, and loading replays similar to mettascope2

import std/[os, strformat]
import ../src/tribal/[environment, replays, common]

proc recordEpisode() =
  echo "Recording a tribal episode..."
  
  # Create environment and replay recorder
  let env = newEnvironment()
  let recorder = newTribalReplayRecorder(env, "example_episode.tribal")
  
  # Start recording
  recorder.startRecording()
  echo fmt"Started recording with {recorder.replay.objects.len} objects and {recorder.replay.agents.len} agents"
  
  # Simulate an episode (100 steps)
  let numSteps = 100
  var actions: array[MapAgents, array[2, uint8]]
  
  for step in 0 ..< numSteps:
    # Agents perform random actions
    for agentId in 0 ..< MapAgents:
      actions[agentId][0] = uint8(step mod 6)  # Cycle through actions
      actions[agentId][1] = uint8(step mod 8)  # Cycle through directions/parameters
    
    # Record this step
    recorder.recordStep(actions.addr)
    
    # Step the environment
    env.step(actions.addr)
    
    if step mod 20 == 0:
      echo fmt"Recorded step {step + 1}/{numSteps}"
  
  # Save the replay
  let fileName = "recorded_episode.tribal"
  recorder.saveReplay(fileName)
  echo fmt"Episode saved to {fileName}"

proc playbackReplay(fileName: string) =
  echo fmt"Playing back replay: {fileName}"
  
  if not fileExists(fileName):
    echo "Replay file not found!"
    return
  
  # Load the replay
  let replay = loadTribalReplay(fileName)
  echo fmt"Loaded replay with {replay.numAgents} agents and {replay.maxSteps} steps"
  
  # Print some basic info
  echo fmt"Map size: {replay.mapSize[0]}x{replay.mapSize[1]}"
  echo fmt"Action types: {replay.actionNames}"
  echo fmt"Item types: {replay.itemNames}"
  echo fmt"Thing types: {replay.typeNames}"
  
  # Show replay states at key moments
  let keySteps = [0, replay.maxSteps div 4, replay.maxSteps div 2, replay.maxSteps - 1]
  
  for step in keySteps:
    echo fmt"\n=== Step {step} ==="
    
    # Show agent states
    echo "Agent states:"
    for agentId in 0 ..< min(3, replay.agents.len):  # Show first 3 agents
      let entity = replay.getEntityAtStep(agentId, step)
      if entity != nil and entity.location.len > 0:
        let pos = entity.location[0]
        echo fmt"  Agent {agentId}: pos=({pos.x}, {pos.y})"
        if entity.currentReward.len > 0:
          echo fmt"    Reward: {entity.currentReward[0]:.3f}"
        if entity.actionId.len > 0 and entity.actionId[0] < replay.actionNames.len:
          echo fmt"    Last action: {replay.actionNames[entity.actionId[0]]}"

proc analyzeReplay(fileName: string) =
  echo fmt"Analyzing replay: {fileName}"
  
  let replay = loadTribalReplay(fileName)
  
  # Calculate some statistics
  var totalRewards: array[MapAgents, float32]
  var actionCounts: array[6, int]  # Count of each action type
  
  for agentId in 0 ..< replay.agents.len:
    let agent = replay.agents[agentId]
    
    # Sum total rewards
    if agent.totalReward.len > 0:
      totalRewards[agentId] = agent.totalReward[^1]  # Final total reward
    
    # Count actions
    for actionId in agent.actionId:
      if actionId >= 0 and actionId < actionCounts.len:
        actionCounts[actionId] += 1
  
  # Print analysis
  echo "\nReward Analysis:"
  for agentId in 0 ..< min(5, replay.agents.len):
    echo fmt"  Agent {agentId}: {totalRewards[agentId]:.3f} total reward"
  
  echo "\nAction Distribution:"
  for actionId in 0 ..< actionCounts.len:
    let actionName = if actionId < replay.actionNames.len: replay.actionNames[actionId] else: "unknown"
    echo fmt"  {actionName}: {actionCounts[actionId]} times"

when isMainModule:
  echo "Tribal Replay System Example"
  echo "============================"
  
  # Record an episode
  recordEpisode()
  
  # Play it back
  let replayFile = "recorded_episode.tribal"
  playbackReplay(replayFile)
  
  # Analyze it
  analyzeReplay(replayFile)
  
  # Clean up
  if fileExists(replayFile):
    removeFile(replayFile)
    echo "\nCleaned up replay file"
  
  echo "\nExample completed!"