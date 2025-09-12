## Test lantern team coloring and production
import std/[strformat, strutils, tables]
import ../src/tribal/environment
import ../src/tribal/ai
import ../src/tribal/external_actions
import test_utils

proc testLanternProductionAndColoring() =
  echo "Testing Lantern Production and Team Coloring"
  echo "=" & repeat("=", 45)
  
  var env = setupTestEnvironment()
  setupTestController(seed = 42)
  
  # Find WeavingLoom specialists by running one step to initialize agent states
  stepTestEnvironment(env)  # Initialize agent states with actual roles
  
  var lanternMakers: seq[tuple[agentId: int, teamId: int, role: AgentRole]]
  
  # Access the global controller to get actual agent roles
  if globalController != nil and globalController.controllerType == BuiltinAI:
    let controller = globalController.aiController
    for i in 0 ..< env.agents.len:
      if controller.agentStates.hasKey(i):
        let agent = env.agents[i]
        let state = controller.agentStates[i]
        let agentId = agent.agentId
        let teamId = agentId div 5
        
        if state.role == WeavingLoomSpecialist:
          lanternMakers.add((agentId: agentId, teamId: teamId, role: state.role))
          echo fmt"  Agent {agentId} (Team {teamId}): WeavingLoom Specialist"
  
  echo fmt"\nFound {lanternMakers.len} lantern makers across teams"
  
  # Count initial lanterns
  var initialLanterns = 0
  for thing in env.things:
    if thing.kind == PlantedLantern:
      initialLanterns += 1
  echo fmt"Initial planted lanterns: {initialLanterns}"
  
  # Run simulation to see if lanterns get produced and planted
  echo "\nRunning 100 simulation steps to test lantern production..."
  
  for step in 1..100:
    # Use production-identical step method
    stepTestEnvironment(env)
    
    # Check lantern inventories and planted lanterns every 20 steps
    if step mod 20 == 0:
      var lanternInventory = 0
      var plantedLanterns = 0
      
      for agent in env.agents:
        lanternInventory += agent.inventoryLantern
      
      for thing in env.things:
        if thing.kind == PlantedLantern:
          plantedLanterns += 1
      
      echo fmt"  Step {step}: {lanternInventory} lanterns in inventory, {plantedLanterns} planted"
      
      # Show lantern maker status
      for maker in lanternMakers:
        let agent = env.agents[maker.agentId]
        echo fmt"    Agent {maker.agentId} (Team {maker.teamId}): wheat={agent.inventoryWheat}, lantern={agent.inventoryLantern}"
  
  # Final count
  var finalLanternInventory = 0
  var finalPlantedLanterns = 0
  
  for agent in env.agents:
    finalLanternInventory += agent.inventoryLantern
  
  for thing in env.things:
    if thing.kind == PlantedLantern:
      finalPlantedLanterns += 1
  
  echo fmt"\nFinal Results:"
  echo fmt"  Lanterns in inventory: {finalLanternInventory}"
  echo fmt"  Planted lanterns: {finalPlantedLanterns}"
  echo fmt"  Net lanterns created: {finalPlantedLanterns - initialLanterns + finalLanternInventory}"
  
  if finalPlantedLanterns > initialLanterns or finalLanternInventory > 0:
    echo "✓ Lantern production is working!"
  else:
    echo "⚠ No lanterns were produced - need to debug further"
  
  echo "\n" & "=" & repeat("=", 45)
  echo "Lantern production test completed"

when isMainModule:
  testLanternProductionAndColoring()