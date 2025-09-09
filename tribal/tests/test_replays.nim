import std/[os, strformat]
import vmath
import ../src/tribal/[environment, replays, common]

proc testReplayRecording() =
  echo "Testing tribal replay recording..."
  
  # Create a test environment
  let env = newEnvironment()
  
  # Create a replay recorder
  let recorder = newTribalReplayRecorder(env, "test_replay.tribal")
  
  # Start recording
  recorder.startRecording()
  echo fmt"Started recording with {recorder.replay.objects.len} objects and {recorder.replay.agents.len} agents"
  
  # Simulate some steps
  let numSteps = 10
  var actions: array[MapAgents, array[2, uint8]]
  
  for step in 0 ..< numSteps:
    # Set up some test actions (random movements)
    for agentId in 0 ..< MapAgents:
      actions[agentId][0] = 1  # move action
      actions[agentId][1] = uint8(step mod 8)  # direction varies by step
    
    # Record the step
    recorder.recordStep(actions.addr)
    
    # Step the environment
    env.step(actions.addr)
    
    echo fmt"Recorded step {step + 1}/{numSteps}"
  
  # Stop recording and save
  let fileName = "test_replay_output.tribal"
  recorder.saveReplay(fileName)
  echo fmt"Replay saved to {fileName}"
  
  # Test loading the replay
  if fileExists(fileName):
    echo "Testing replay loading..."
    let loadedReplay = loadTribalReplay(fileName)
    echo fmt"Loaded replay: {loadedReplay.numAgents} agents, {loadedReplay.maxSteps} steps, {loadedReplay.objects.len} objects"
    
    # Test rendering a few steps
    for step in 0 ..< min(3, loadedReplay.maxSteps):
      echo fmt"=== Rendering Step {step} ==="
      let stepOutput = loadedReplay.renderReplayStep(step)
      echo stepOutput
    
    # Clean up
    removeFile(fileName)
    echo "Test completed successfully!"
  else:
    echo "Error: Replay file was not created"

proc testReplayStepAccess() =
  echo "Testing replay step access..."
  
  # Create a simple environment and record a few steps
  let env = newEnvironment()
  let recorder = newTribalReplayRecorder(env, "step_test.tribal")
  
  recorder.startRecording()
  
  # Record 3 steps
  var actions: array[MapAgents, array[2, uint8]]
  for step in 0 ..< 3:
    for agentId in 0 ..< MapAgents:
      actions[agentId][0] = 0  # noop
      actions[agentId][1] = 0
    
    recorder.recordStep(actions.addr)
    env.step(actions.addr)
  
  let data = recorder.stopRecording()
  let replay = loadTribalReplay(data, "test")
  
  # Test accessing entity states at different steps
  echo fmt"Testing step access for {replay.objects.len} objects..."
  
  for entityId in 0 ..< min(3, replay.objects.len):
    echo fmt"Entity {entityId}:"
    for step in 0 ..< replay.maxSteps:
      let stepEntity = replay.getEntityAtStep(entityId, step)
      if stepEntity != nil and stepEntity.location.len > 0:
        let pos = stepEntity.location[0]
        echo fmt"  Step {step}: pos=({pos.x}, {pos.y})"
  
  echo "Step access test completed!"

when isMainModule:
  echo "Running tribal replay tests..."
  testReplayRecording()
  echo ""
  testReplayStepAccess()
  echo "All tests completed!"