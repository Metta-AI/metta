import ../src/mettascope/tribal

# Very simple test - just create env and step once

var env = newEnvironment()
echo "Created environment"

# Count initial things
echo "Initial things: ", env.things.len
echo "Initial agents: ", env.agents.len

# Single step
var actions: array[MapAgents, array[2, uint8]]
for i in 0 ..< MapAgents:
  actions[i] = [0'u8, 0'u8]  # All NOOP

echo "About to step..."
env.step(actions.addr)
echo "Step complete"

# Count after
echo "Final things: ", env.things.len
echo "Final agents: ", env.agents.len