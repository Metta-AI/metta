import std/[strformat, random]
import ../src/mettascope/tribal

# Create environment for testing combat
var env = newEnvironment()

echo "Initial state:"
echo fmt"  Agents: {env.agents.len}"
echo fmt"  Clippys: {env.things.filterIt(it.kind == Clippy).len}"
echo fmt"  Temples: {env.things.filterIt(it.kind == Temple).len}"

# Run a few steps to let clippys move and potentially encounter agents
for step in 1..10:
  # Create dummy actions (all agents do nothing)
  var actions: array[MapAgents, array[2, uint8]]
  for i in 0 ..< MapAgents:
    actions[i] = [0'u8, 0'u8]  # NOOP
  
  env.step(actions.addr)
  
  let aliveAgents = env.agents.filterIt(env.terminated[it.agentId] == 0).len
  let clippyCount = env.things.filterIt(it.kind == Clippy).len
  
  echo fmt"Step {step}: Agents alive: {aliveAgents}/{MapAgents}, Clippys: {clippyCount}"

echo "\nChecking for adjacent clippys and agents:"
for clippy in env.things:
  if clippy.kind == Clippy:
    # Check adjacent positions
    for dx in [-1, 0, 1]:
      for dy in [-1, 0, 1]:
        if abs(dx) + abs(dy) == 1:  # Only orthogonal adjacency
          let checkPos = ivec2(clippy.pos.x + dx, clippy.pos.y + dy)
          if checkPos.x >= 0 and checkPos.x < MapWidth and checkPos.y >= 0 and checkPos.y < MapHeight:
            let thing = env.getThing(checkPos)
            if not isNil(thing) and thing.kind == Agent:
              echo fmt"  Combat potential: Clippy at ({clippy.pos.x},{clippy.pos.y}) adjacent to Agent at ({checkPos.x},{checkPos.y})"

# Run more steps to see combat
echo "\nRunning 40 more steps to observe combat..."
for step in 11..50:
  var actions: array[MapAgents, array[2, uint8]]
  for i in 0 ..< MapAgents:
    actions[i] = [0'u8, 0'u8]  # NOOP
  
  env.step(actions.addr)
  
  if step mod 10 == 0:
    let aliveAgents = env.agents.filterIt(env.terminated[it.agentId] == 0).len
    let clippyCount = env.things.filterIt(it.kind == Clippy).len
    echo fmt"Step {step}: Agents alive: {aliveAgents}/{MapAgents}, Clippys: {clippyCount}"