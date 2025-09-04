import std/[strformat, os, tables]
import vmath
import ../src/mettascope/tribal
import ../src/mettascope/controller

proc main() =
  echo "Creating tribal environment with controller..."
  
  # Create environment
  var env = newEnvironment()
  echo fmt"Environment created with {MapAgents} agents"
  echo fmt"Map size: {MapWidth}x{MapHeight}"
  echo fmt"Houses: {MapRoomObjectsHouses}"
  echo fmt"Mines: {MapRoomObjectsMines}"
  echo fmt"Generators (Converters): {MapRoomObjectsGenerators}"
  
  # Create controller
  var controller = newController(seed = 2024)
  echo "Controller initialized"
  
  # Run simulation for N steps
  let numSteps = 100
  echo fmt"Running simulation for {numSteps} steps..."
  echo ""
  
  for step in 0 ..< numSteps:
    # Create actions array for all agents
    var actions: array[MapAgents, array[2, uint8]]
    
    # Get action from controller for each agent
    for agentId in 0 ..< MapAgents:
      actions[agentId] = controller.decideAction(env, agentId)
    
    # Step the environment
    env.step(addr actions)
    
    # Update controller
    controller.updateController()
    
    # Print status every 10 steps
    if step mod 10 == 0:
      echo fmt"Step {step}:"
      
      # Count agent states
      var totalOre = 0
      var totalBatteries = 0
      var totalDeposited = 0
      
      for agentId in 0 ..< MapAgents:
        let agent = env.agents[agentId]
        totalOre += agent.inventory
        totalBatteries += max(0, agent.energy - MapObjectAgentInitialEnergy)
      
      # Count altar hearts (deposited batteries)
      for thing in env.things:
        if thing.kind == Altar:
          # Hearts above initial value are deposited batteries
          totalDeposited += max(0, thing.hp - MapObjectAltarInitialHearts)
      
      echo fmt"  Total ore carried: {totalOre}"
      echo fmt"  Total batteries carried: {totalBatteries}"
      echo fmt"  Total batteries deposited at altars: {totalDeposited}"
      
      # Show sample agent states
      if MapAgents > 0:
        let agent = env.agents[0]
        if controller.agentStates.hasKey(0):
          let state = controller.agentStates[0]
          echo fmt"  Agent 0: pos=({agent.pos.x},{agent.pos.y}), " &
               fmt"ore={agent.inventory}, energy={agent.energy}, " &
               fmt"target={state.targetType}, targetPos=({state.currentTarget.x},{state.currentTarget.y})"
    
    # Check for early termination if all altars are full
    var allAltarsFull = true
    for thing in env.things:
      if thing.kind == Altar:
        if thing.hp < MapObjectAltarInitialHearts * 2:  # Max capacity is 2x initial
          allAltarsFull = false
          break
    
    if allAltarsFull:
      echo fmt"\nAll altars are full! Simulation complete at step {step}"
      break
  
  echo "\nFinal statistics:"
  echo env.getEpisodeStats()
  
  # Render final state
  echo "\nFinal map state:"
  echo env.render()

when isMainModule:
  main()