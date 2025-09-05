## Test to verify agent colors are warm (reds/oranges/yellows)
import std/[strformat, strutils, tables]
import vmath
import ../src/tribal/environment
import ../src/tribal/colors

proc testWarmColors() =
  echo "Test: Agent Warm Color Assignment"
  echo "-" & repeat("-", 40)
  
  var env = newEnvironment()
  
  echo fmt"  Total agents: {env.agents.len}"
  echo fmt"  Village colors assigned: {agentVillageColors.len}"
  echo ""
  
  # Check each agent's assigned color
  for i, agent in env.agents:
    if i < agentVillageColors.len:
      let color = agentVillageColors[i]
      echo fmt"  Agent {i}: R={color.r:.2f}, G={color.g:.2f}, B={color.b:.2f}"
      
      # Check if color is warm (high red component, moderate green, low blue)
      if color.r >= 0.8 and color.b <= 0.6:
        echo "    ✓ Warm color (red/orange/yellow range)"
      elif color.r >= 0.7:
        echo "    ✓ Moderately warm color"
      else:
        echo "    ✗ Not a warm color!"
  
  echo ""
  echo "  Altar colors:"
  for pos, color in altarColors:
    echo fmt"  Altar at ({pos[0]},{pos[1]}): R={color.r:.2f}, G={color.g:.2f}, B={color.b:.2f}"

when isMainModule:
  echo "\n" & "=" & repeat("=", 50)
  echo "Warm Color Assignment Test"
  echo "=" & repeat("=", 50) & "\n"
  
  testWarmColors()
  
  echo "=" & repeat("=", 50)
  echo "Test completed"