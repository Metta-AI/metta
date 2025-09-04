import ../src/mettascope/tribal
import std/strformat

# Test combat by placing agent and clippy next to each other

# Create a simple environment
var env = newEnvironment()

echo "Looking for agents and clippys adjacent to each other..."
var combatFound = false

# Run for many steps to see if combat occurs naturally
for step in 1..200:
  var actions: array[MapAgents, array[2, uint8]]
  for i in 0 ..< MapAgents:
    actions[i] = [0'u8, 0'u8]  # All agents NOOP
  
  env.step(actions.addr)
  
  # Check if any agents are dead
  var deadAgents = 0
  for i in 0 ..< MapAgents:
    if env.terminated[i] == 1.0:
      deadAgents += 1
      combatFound = true
  
  if step mod 50 == 0 or deadAgents > 0:
    var clippyCount = 0
    for thing in env.things:
      if thing.kind == Clippy:
        clippyCount += 1
    
    echo fmt"Step {step}: Dead agents: {deadAgents}, Clippys: {clippyCount}"
    
    if deadAgents > 0:
      echo "Combat has occurred! Agents have died."
      break

if not combatFound:
  echo "\nNo combat occurred naturally. This is expected since:"
  echo "- Agents are doing NOOP (not moving)"
  echo "- Clippys are seeking altars, not agents"
  echo "- They may never become adjacent"
  echo "\nTo truly test combat, agents would need to:"
  echo "1. Move around (action 1 instead of 0)"
  echo "2. Attack clippys when they see them (action 4)"
  echo "3. Or we need to spawn them adjacent initially"