## Test lantern team coloring and production
import std/[strformat, strutils]
import ../src/tribal/environment
import ../src/tribal/ai

proc testLanternProductionAndColoring() =
  echo "Testing Lantern Production and Team Coloring"
  echo "=" & repeat("=", 45)
  
  var env = newEnvironment()
  let controller = newController(seed = 42)
  
  # Find WeavingLoom specialists across different teams
  var lanternMakers: seq[tuple[agentId: int, teamId: int, role: AgentRole]]
  
  for i in 0 ..< env.agents.len:
    let agent = env.agents[i]
    let agentId = agent.agentId
    let teamId = agentId div 5
    let role = AgentRole(agentId mod 5)
    
    if role == WeavingLoomSpecialist:
      lanternMakers.add((agentId: agentId, teamId: teamId, role: role))
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
  var actions: array[MapAgents, array[2, uint8]]
  
  for step in 1..100:
    # Let AI decide actions for all agents
    for i in 0 ..< env.agents.len:
      actions[i] = controller.decideAction(env, i)
    
    env.step(addr actions)
    controller.updateController()
    
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