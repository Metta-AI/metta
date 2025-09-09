## Example of manually integrating replay recording with tribal environment
import std/[times, strformat]
import ../src/tribal/[environment, replays, common]

proc runWithReplayRecording() =
  # Create environment
  let env = newEnvironment()
  
  # Create replay recorder with timestamp filename
  let timestamp = epochTime().int
  let fileName = fmt"tribal_replay_{timestamp}.tribal"
  let recorder = newTribalReplayRecorder(env, fileName)
  
  echo fmt"Starting replay recording to {fileName}"
  recorder.startRecording()
  
  # Run simulation for specified number of steps
  let maxSteps = 500  # Adjust as needed
  var actions: array[MapAgents, array[2, uint8]]
  
  for step in 0 ..< maxSteps:
    # Your agent logic here - this example uses random actions
    for agentId in 0 ..< MapAgents:
      # Example: agents cycle through different behaviors
      case step mod 30:
      of 0..9:    # Movement phase
        actions[agentId][0] = 1  # move
        actions[agentId][1] = uint8(agentId mod 8)  # direction
      of 10..19:  # Resource gathering phase
        actions[agentId][0] = 3  # get
        actions[agentId][1] = uint8(agentId mod 8)  # direction
      of 20..29:  # Building interaction phase
        actions[agentId][0] = 5  # put
        actions[agentId][1] = uint8(agentId mod 8)  # direction
      else:
        actions[agentId][0] = 0  # noop
        actions[agentId][1] = 0
    
    # Record this step BEFORE stepping the environment
    recorder.recordStep(actions.addr)
    
    # Step the environment
    env.step(actions.addr)
    
    # Optional: Print progress
    if step mod 100 == 0:
      echo fmt"Recorded step {step + 1}/{maxSteps}"
  
  # Save the replay
  recorder.saveReplay(fileName)
  echo fmt"Replay saved to {fileName}"
  
  # Show final environment stats
  echo env.getEpisodeStats()

when isMainModule:
  runWithReplayRecording()