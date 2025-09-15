## Test lantern team coloring and production
import std/[strformat, strutils]
import vmath
import ../src/tribal/environment
import ../src/tribal/ai
import test_utils

proc testLanternProductionAndColoring() =
  echo "Testing Lantern Production and Team Coloring"
  echo "=" & repeat("=", 45)
  
  var env = setupTestEnvironment()
  setupTestController(seed = 42)
  
  # Find WeavingLoom specialists by running one step to initialize agent states
  stepTestEnvironment(env)  # Initialize agent states with actual roles
  
  var lanternMakers: seq[tuple[agentId: int, teamId: int, role: AgentRole]]
  
  # Find WeavingLoom specialists using the simplified role assignment
  for i in 0 ..< env.agents.len:
    let agent = env.agents[i]
    let agentId = agent.agentId
    let teamId = agentId div 5
    
    # In simplified AI, WeavingLoomSpecialists are agents where (id mod 5) == 4
    if (agentId mod 5) == 4:
      lanternMakers.add((agentId: agentId, teamId: teamId, role: WeavingLoomSpecialist))
      echo fmt"  Agent {agentId} (Team {teamId}): WeavingLoom Specialist"
  
  echo fmt"\nFound {lanternMakers.len} lantern makers across teams"
  
  # Count initial lanterns and check for WeavingLooms
  var initialLanterns = 0
  var weavingLooms = 0
  for thing in env.things:
    if thing.kind == PlantedLantern:
      initialLanterns += 1
    elif thing.kind == WeavingLoom:
      weavingLooms += 1
      echo fmt"  Found WeavingLoom at ({thing.pos.x}, {thing.pos.y})"
  
  echo fmt"Initial planted lanterns: {initialLanterns}"
  echo fmt"Available WeavingLooms: {weavingLooms}"
  
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
        echo fmt"    Agent {maker.agentId} (Team {maker.teamId}): wheat={agent.inventoryWheat}, lantern={agent.inventoryLantern}, pos=({agent.pos.x},{agent.pos.y})"
        
        # Debug: Check if they can find a loom when they have wheat
        if agent.inventoryWheat > 0:
          var nearestLoom: Thing = nil
          var minDist = 999999
          for thing in env.things:
            if thing.kind == WeavingLoom:
              let dist = abs(thing.pos.x - agent.pos.x) + abs(thing.pos.y - agent.pos.y)
              if dist < minDist and dist < 30:
                minDist = dist
                nearestLoom = thing
          
          if nearestLoom != nil:
            echo fmt"      → Nearest loom at ({nearestLoom.pos.x},{nearestLoom.pos.y}), distance: {minDist}"
          else:
            echo fmt"      → No loom found!"
  
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