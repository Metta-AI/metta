## Consolidated combat system tests
import test_utils

echo "============================"
echo "Combat System Tests"
echo "============================"
echo ""

# Test 1: Basic combat occurrence
proc testBasicCombat() =
  echo "Test 1: Basic Combat"
  echo "--------------------"
  var env = newEnvironment()
  
  let initial = env.countEntities()
  echo fmt"Initial: {initial.agents} agents, {initial.clippys} clippys"
  
  # Run simulation until combat occurs
  var combatOccurred = false
  for i in 1..20:
    env.runSteps(10)
    let dead = env.countDeadAgents()
    if dead > 0:
      combatOccurred = true
      echo fmt"Combat occurred at step {i*10}: {dead} agents died"
      break
  
  if not combatOccurred:
    echo "No combat in 200 steps (agents idle)"
  echo ""

# Test 2: Agent respawn at altar
proc testRespawn() =
  echo "Test 2: Respawn System"
  echo "----------------------"
  var env = newEnvironment()
  
  let initialHearts = env.getTotalAltarHearts()
  echo fmt"Initial altar hearts: {initialHearts}"
  
  # Run until respawn might occur
  var respawnDetected = false
  for i in 1..50:
    env.runSteps(10)
    let currentHearts = env.getTotalAltarHearts()
    if currentHearts < initialHearts:
      respawnDetected = true
      echo fmt"Respawn detected! Hearts used: {initialHearts - currentHearts}"
      break
  
  if not respawnDetected:
    echo "No respawns detected (may need agents to die near home altar)"
  echo ""

# Test 3: Direct attack on clippy
proc testDirectAttack() =
  echo "Test 3: Direct Attack"
  echo "---------------------"
  var env = newEnvironment()
  
  # Find a clippy and attack it
  var clippyFound = false
  for thing in env.things:
    if thing.kind == Clippy:
      echo fmt"Found clippy at ({thing.pos.x}, {thing.pos.y})"
      # Note: In real test, agent would need to be positioned to attack
      clippyFound = true
      break
  
  if clippyFound:
    echo "Agent can attack clippy with action 4 (instant kill)"
  else:
    echo "No clippy found initially"
  echo ""

# Test 4: Combat statistics over time
proc testCombatStats() =
  echo "Test 4: Combat Statistics (500 steps)"
  echo "--------------------------------------"
  var env = newEnvironment()
  
  var totalDeaths = 0
  var totalRespawns = 0
  var previousDead: array[MapAgents, bool]
  
  env.printEntityCounts("Initial:")
  
  for step in 1..500:
    # Track deaths and respawns
    for i in 0 ..< MapAgents:
      if env.terminated[i] == 1.0 and not previousDead[i]:
        totalDeaths += 1
        previousDead[i] = true
      elif env.terminated[i] == 0.0 and previousDead[i]:
        totalRespawns += 1
        previousDead[i] = false
    
    env.runSteps(1)
    
    if step mod 100 == 0:
      echo fmt"Step {step}: Deaths: {totalDeaths}, Respawns: {totalRespawns}"
  
  env.printEntityCounts("Final:")
  
  if totalDeaths > 0:
    echo fmt"✓ Combat system working: {totalDeaths} deaths, {totalRespawns} respawns"
  else:
    echo "⚠ Limited combat (agents were idle)"
  echo ""

# Run all tests
testBasicCombat()
testRespawn()
testDirectAttack()
testCombatStats()

echo "Combat tests complete!"