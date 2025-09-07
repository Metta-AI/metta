import std/[strformat, sequtils, random, times, tables, strutils]
import vmath
import ../src/tribal/environment
import ../src/tribal/ai
import ../src/tribal/objects
import ../src/tribal/common

proc createTestEnvironment(): Environment =
  ## Use the actual newEnvironment which creates a full game
  result = newEnvironment()
  echo "Using full game environment for stuck detection test"

proc testStuckDetection() =
  echo "\n=== Stuck Detection Test ==="
  echo "Testing that agents detect when stuck and escape\n"
  
  var env = createTestEnvironment()
  var controller = newController(seed = 42)
  
  # Track a few agents (not all MapAgents)
  let numAgentsToTrack = min(4, env.agents.len)
  
  # Track agent positions over time
  var positionHistory: seq[seq[IVec2]] = @[]
  for i in 0 ..< numAgentsToTrack:
    positionHistory.add(@[])
  
  var stuckDetected = false
  var escapeDetected = false
  
  # Run simulation for 50 steps to see stuck behavior
  for step in 0..49:
    # Get actions for all agents
    var actions: array[MapAgents, array[2, uint8]]
    for i in 0 ..< env.agents.len:
      actions[i] = controller.decideAction(env, i)
      
      # Track positions for first few agents
      if i < numAgentsToTrack:
        positionHistory[i].add(env.agents[i].pos)
        
        # Check if stuck detection triggered
        if controller.agentStates.hasKey(i):
          let state = controller.agentStates[i]
          if state.stuckCounter >= 5:
            stuckDetected = true
            if not state.escapeMode:
              echo fmt"Step {step}: Agent {i} stuck at {env.agents[i].pos} (counter: {state.stuckCounter})"
          if state.escapeMode:
            escapeDetected = true
            echo fmt"Step {step}: Agent {i} in escape mode at {env.agents[i].pos} (steps remaining: {state.escapeStepsRemaining})"
    
    # Execute actions
    env.step(addr actions)
    
    # Show positions every 10 steps
    if step mod 10 == 0:
      echo fmt"\nStep {step} positions (first {numAgentsToTrack} agents):"
      for i in 0 ..< numAgentsToTrack:
        echo fmt"  Agent {i}: {env.agents[i].pos}"
  
  # Analyze results
  echo "\n=== Analysis ==="
  
  # Check if agents got stuck
  for i in 0 ..< numAgentsToTrack:
    let history = positionHistory[i]
    var maxStuckTime = 0
    var currentStuckTime = 0
    var lastPos = history[0]
    
    for pos in history:
      if pos == lastPos:
        currentStuckTime += 1
        maxStuckTime = max(maxStuckTime, currentStuckTime)
      else:
        currentStuckTime = 1
        lastPos = pos
    
    echo fmt"Agent {i}: Max time in same position: {maxStuckTime} steps"
  
  if stuckDetected:
    echo "✓ Stuck detection triggered successfully"
  else:
    echo "⚠ No stuck detection triggered (may not have gotten stuck)"
  
  if escapeDetected:
    echo "✓ Escape mode activated successfully"
  else:
    echo "⚠ No escape mode activated"
  
  # Check final positions to see if agents moved
  echo "\nFinal positions:"
  for i in 0 ..< numAgentsToTrack:
    let startPos = positionHistory[i][0]
    let endPos = env.agents[i].pos
    let distance = abs(endPos.x - startPos.x) + abs(endPos.y - startPos.y)
    echo fmt"  Agent {i}: moved {distance} tiles from {startPos} to {endPos}"

proc visualizeBottleneck() =
  echo "\n=== Bottleneck Visualization ==="
  echo "Showing the test scenario layout\n"
  
  # Create a small grid to show the bottleneck
  var grid = newSeq[seq[char]](20)
  for i in 0..19:
    grid[i] = newSeqWith(30, '.')
  
  # Add walls
  for x in 10..20:
    if x != 15:  # Gap at x=15
      grid[10][x] = '#'
      grid[12][x] = '#'
  
  # Mark key positions
  grid[8][15] = 'A'  # Altar
  grid[14][14] = '1'  # Initial agent positions
  grid[14][15] = '2'
  grid[15][14] = '3'
  grid[15][15] = '4'
  
  # Display
  echo "   0123456789012345678901234567890"
  for y, row in grid:
    let rowStr = row.join("")
    echo fmt"{y:2} {rowStr}"
  
  echo "\nLegend:"
  echo "  # = Wall"
  echo "  A = Altar (target)"
  echo "  1-4 = Initial agent positions"
  echo "  Gap at (15, 11) allows passage"

when isMainModule:
  testStuckDetection()