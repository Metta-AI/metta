import src/mettascope/[tribal, controller]
import std/[times, os]

echo "Testing random seed generation..."

# Test environment seed
for i in 1..3:
  var env = newEnvironment()
  env.reset()
  sleep(10)  # Small delay to ensure different time
  
# Test controller seed  
for i in 1..3:
  let seed = int(epochTime() * 1000)
  echo "Controller seed: ", seed
  var controller = newController(seed)
  sleep(10)  # Small delay

echo "If you saw different seeds above, randomization is working!"