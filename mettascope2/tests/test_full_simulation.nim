import std/[strformat, tables, strutils]
import vmath
import ../src/mettascope/tribal
import ../src/mettascope/controller
import ../src/mettascope/actions
import ../src/mettascope/common

proc main() =
  echo "Testing full simulation with controller, mines, and clippys..."
  echo repeat("=", 60)
  
  # Initialize environment (normally done by mettascope main)
  env = newEnvironment()
  
  echo fmt"Environment initialized with:"
  echo fmt"  - {MapAgents} agents"
  echo fmt"  - {MapRoomObjectsMines} mines"
  echo fmt"  - {MapRoomObjectsGenerators} generators (converters)"
  echo fmt"  - {MapRoomObjectsHouses} houses with altars"
  echo ""
  
  # Run longer simulation to see actual behavior
  let simSteps = 50
  echo fmt"Running {simSteps} simulation steps..."
  echo ""
  
  var lastOreCount = 0
  var lastBatteryCount = 0
  var lastClippyCount = 0
  var finalClippyCount = 0
  
  for step in 0 ..< simSteps:
    simStep()
    
    # Collect statistics
    var agentsWithOre = 0
    var agentsWithBatteries = 0
    var frozenAgents = 0
    var totalOre = 0
    var totalBatteries = 0
    
    for agent in env.agents:
      if agent.inventory > 0:
        agentsWithOre += 1
        totalOre += agent.inventory
      if agent.energy > MapObjectAgentInitialEnergy:
        agentsWithBatteries += 1
        totalBatteries += agent.energy - MapObjectAgentInitialEnergy
      if agent.frozen > 0:
        frozenAgents += 1
    
    # Count clippys
    var clippyCount = 0
    var minesReady = 0
    var generatorsReady = 0
    var altarHearts = 0
    
    for thing in env.things:
      case thing.kind:
      of Clippy: 
        clippyCount += 1
      of Mine:
        if thing.cooldown == 0:
          minesReady += 1
      of Generator:
        if thing.cooldown == 0:
          generatorsReady += 1
      of Altar:
        altarHearts += thing.hp
      else:
        discard
    
    # Print updates when something changes
    if step mod 10 == 0 or 
       totalOre != lastOreCount or 
       totalBatteries != lastBatteryCount or 
       clippyCount != lastClippyCount:
      echo fmt"Step {step:3}: Ore={totalOre:2} (carriers={agentsWithOre:2}), " &
           fmt"Batteries={totalBatteries:2} (carriers={agentsWithBatteries:2}), " &
           fmt"Clippys={clippyCount:2}, " &
           fmt"Frozen={frozenAgents:2}, " &
           fmt"MinesReady={minesReady}/{MapRoomObjectsMines}, " &
           fmt"AltarHearts={altarHearts}"
      
      lastOreCount = totalOre
      lastBatteryCount = totalBatteries
      lastClippyCount = clippyCount
    
    # Save final clippy count
    finalClippyCount = clippyCount
  
  echo ""
  echo repeat("=", 60)
  echo "=== Final Controller States ==="
  
  # Show a few agent controller states
  var statesSampled = 0
  for agentId, state in agentController.agentStates.pairs:
    if statesSampled < 3:  # Show first 3 agents
      let agent = env.agents[agentId]
      echo fmt"Agent {agentId}:"
      echo fmt"  Position: ({agent.pos.x}, {agent.pos.y})"
      echo fmt"  Inventory: ore={agent.inventory}, energy={agent.energy}"
      echo fmt"  Controller: target={state.targetType}, hasOre={state.hasOre}, hasBattery={state.hasBattery}"
      echo fmt"  Target position: ({state.currentTarget.x}, {state.currentTarget.y})"
      statesSampled += 1
  
  echo ""
  echo "=== Verification ==="
  
  # Verify key behaviors
  var checks = 0
  var passed = 0
  
  # Check 1: Controller initialized all agents
  checks += 1
  var controllerStates = 0
  for _ in agentController.agentStates.keys:
    controllerStates += 1
  if controllerStates == MapAgents:
    echo "✅ Controller initialized all agents"
    passed += 1
  else:
    echo fmt"❌ Controller only initialized {controllerStates}/{MapAgents} agents"
  
  # Check 2: Mines have cooldowns (showing they can be used)
  checks += 1
  var minesWithCooldown = 0
  for thing in env.things:
    if thing.kind == Mine and thing.cooldown > 0:
      minesWithCooldown += 1
  if minesWithCooldown > 0 or lastOreCount > 0:
    echo "✅ Mines are being used (cooldowns active or ore collected)"
    passed += 1
  else:
    echo "⚠️  No mines show usage yet (may need more steps)"
  
  # Check 3: Clippys are spawning
  checks += 1
  if finalClippyCount > 3:  # Started with 3 (one per temple)
    echo fmt"✅ Clippys are spawning (count: {finalClippyCount})"
    passed += 1
  else:
    echo fmt"⚠️  Clippys not spawning much yet (count: {finalClippyCount})"
  
  # Check 4: Agents are moving
  checks += 1
  var agentsMoved = 0
  for agentId, state in agentController.agentStates.pairs:
    let agent = env.agents[agentId]
    if agent.pos != state.basePosition:
      agentsMoved += 1
  if agentsMoved > 0:
    echo fmt"✅ {agentsMoved} agents have moved from their starting positions"
    passed += 1
  else:
    echo "❌ No agents have moved from starting positions"
  
  echo ""
  echo fmt"Result: {passed}/{checks} checks passed"
  
  if passed == checks:
    echo "🎉 All systems working!"
  elif passed >= checks - 1:
    echo "✅ System mostly working (may need more simulation steps)"
  else:
    echo "⚠️  Some systems need investigation"

when isMainModule:
  main()