import ../src/mettascope/tribal

# Visual test - show the map with clippys

var env = newEnvironment()

echo "Initial map with temples (t) and clippys (C):"
echo env.render()

echo "\nLegend:"
echo "  A = Agent (alive)"
echo "  C = Clippy"
echo "  t = Temple (spawns clippys)"
echo "  a = Altar (village center)"
echo "  # = Wall"
echo "  m = Mine"
echo "  g = Generator"
echo "  T = Tree (terrain)"
echo "  . = Wheat (terrain)"
echo "  ~ = Water (terrain)"
echo ""

# Run a few steps
for i in 1..5:
  var actions: array[MapAgents, array[2, uint8]]
  for j in 0 ..< MapAgents:
    actions[j] = [0'u8, 0'u8]  # All agents NOOP
  env.step(actions.addr)

var deadCount = 0
for i in 0 ..< MapAgents:
  if env.terminated[i] == 1.0:
    deadCount += 1

var clippyCount = 0
for thing in env.things:
  if thing.kind == Clippy:
    clippyCount += 1

echo "After 5 steps:"
echo "  Living agents: ", 15 - deadCount, " / 15"
echo "  Dead agents: ", deadCount
echo "  Clippys: ", clippyCount