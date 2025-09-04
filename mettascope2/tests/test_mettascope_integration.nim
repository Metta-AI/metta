import std/[strformat, tables]
import vmath
import ../src/mettascope/tribal
import ../src/mettascope/controller
import ../src/mettascope/actions
import ../src/mettascope/common

proc main() =
  echo "Testing mettascope integration with controller..."
  
  # Initialize environment (normally done by mettascope main)
  env = newEnvironment()
  
  echo fmt"Environment initialized with {MapAgents} agents"
  
  # Test simStep with controller
  echo "\nRunning 10 simulation steps..."
  for i in 0 ..< 10:
    simStep()
    
    # Check if agents are doing something
    var agentsWithOre = 0
    var agentsWithBatteries = 0
    var frozenAgents = 0
    
    for agent in env.agents:
      if agent.inventory > 0:
        agentsWithOre += 1
      if agent.energy > MapObjectAgentInitialEnergy:
        agentsWithBatteries += 1
      if agent.frozen > 0:
        frozenAgents += 1
    
    echo fmt"Step {i}: Ore carriers={agentsWithOre}, Battery carriers={agentsWithBatteries}, Frozen={frozenAgents}"
  
  # Check environment state
  echo "\n=== Environment State Check ==="
  
  # Count entities
  var mines = 0
  var generators = 0
  var altars = 0
  var temples = 0
  var clippys = 0
  var walls = 0
  
  for thing in env.things:
    case thing.kind:
    of Mine: mines += 1
    of Generator: generators += 1
    of Altar: altars += 1
    of Temple: temples += 1
    of Clippy: clippys += 1
    of Wall: walls += 1
    of Agent: discard
  
  echo fmt"Mines: {mines}"
  echo fmt"Generators: {generators}"
  echo fmt"Altars: {altars}"
  echo fmt"Temples: {temples}"
  echo fmt"Clippys: {clippys}"
  echo fmt"Walls: {walls}"
  
  # Check if controller is working
  echo "\n=== Controller State Check ==="
  var stateCount = 0
  for key in agentController.agentStates.keys:
    stateCount += 1
  echo fmt"Controller has {stateCount} agent states initialized"
  
  # Sample first agent's controller state
  if agentController.agentStates.hasKey(0):
    let state = agentController.agentStates[0]
    echo fmt"Agent 0 controller state:"
    echo fmt"  Target type: {state.targetType}"
    echo fmt"  Has ore: {state.hasOre}"
    echo fmt"  Has battery: {state.hasBattery}"
    echo fmt"  Wander radius: {state.wanderRadius}"
  
  echo "\n✅ Integration test complete!"

when isMainModule:
  main()