import ../src/mettascope/tribal, vmath

# Test clippy vs agent combat

# Create environment
var env = newEnvironment()

# Initial count
var initialAgents = 0
var initialClippys = 0

for thing in env.things:
  if thing.kind == Agent:
    initialAgents += 1
  elif thing.kind == Clippy:
    initialClippys += 1

echo "Initial state:"
echo "  Agents: ", initialAgents
echo "  Clippys: ", initialClippys
echo ""

# Run 100 steps to allow combat to occur
for step in 1..100:
  # All agents do nothing (to test passive combat)
  var actions: array[MapAgents, array[2, uint8]]
  for i in 0 ..< MapAgents:
    actions[i] = [0'u8, 0'u8]  # NOOP
  
  env.step(actions.addr)
  
  if step mod 20 == 0 or step == 1:
    var currentAgents = 0
    var currentClippys = 0
    var deadAgents = 0
    
    for thing in env.things:
      if thing.kind == Agent:
        currentAgents += 1
      elif thing.kind == Clippy:
        currentClippys += 1
    
    for i in 0 ..< MapAgents:
      if env.terminated[i] == 1.0:
        deadAgents += 1
    
    echo "Step ", step, ":"
    echo "  Living agents: ", currentAgents, " (", deadAgents, " dead)"
    echo "  Clippys: ", currentClippys
    
    # Check for adjacent positions that would trigger combat
    var combatSpots = 0
    for clippy in env.things:
      if clippy.kind == Clippy:
        for dx in [-1, 0, 1]:
          for dy in [-1, 0, 1]:
            if abs(dx) + abs(dy) == 1:  # Only orthogonal
              let checkX = clippy.pos.x + dx
              let checkY = clippy.pos.y + dy
              if checkX >= 0 and checkX < MapWidth and checkY >= 0 and checkY < MapHeight:
                for agent in env.things:
                  if agent.kind == Agent and agent.pos.x == checkX and agent.pos.y == checkY:
                    combatSpots += 1
    
    if combatSpots > 0:
      echo "  Potential combat spots: ", combatSpots

echo "\nFinal result:"
var finalAgents = 0
var finalClippys = 0
var deadAgents = 0

for thing in env.things:
  if thing.kind == Agent:
    finalAgents += 1
  elif thing.kind == Clippy:
    finalClippys += 1

for i in 0 ..< MapAgents:
  if env.terminated[i] == 1.0:
    deadAgents += 1

echo "  Living agents: ", finalAgents, " / ", initialAgents, " (", deadAgents, " died)"
echo "  Clippys: ", finalClippys, " / ", initialClippys

if deadAgents > 0 or finalClippys < initialClippys:
  echo "\nCombat system is working! Agents died and/or clippys were destroyed."
else:
  echo "\nNo combat occurred (agents and clippys may not have met)."