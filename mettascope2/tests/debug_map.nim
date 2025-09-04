import ../src/mettascope/tribal

# Create environment and show visual map
var env = newEnvironment()

echo "Map visualization ('T' = Temple, 'C' = Clippy, 'a' = Altar, 'A' = Agent):"
echo env.render()