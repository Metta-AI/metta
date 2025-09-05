## Test that house tiles match their team colors
import std/[strformat, strutils, tables]
import vmath
import ../src/tribal/environment
import ../src/tribal/colors
import ../src/tribal/objects

proc testHouseColors() =
  echo "Test: House Tile Team Colors"
  echo "-" & repeat("-", 40)
  
  var env = newEnvironment()
  
  # Check each altar's house tiles
  for thing in env.things:
    if thing.kind == Altar and thing.houseSize > 0 and thing.houseTopLeft.x >= 0:
      let altarPos = thing.pos
      let teamColor = altarColors[altarPos]
      
      echo fmt"  Altar at ({altarPos[0]},{altarPos[1]}):"
      echo fmt"    Team color: R={teamColor.r:.2f}, G={teamColor.g:.2f}, B={teamColor.b:.2f}"
      
      # Check a sample house tile (center of house)
      let centerX = thing.houseTopLeft.x + 2
      let centerY = thing.houseTopLeft.y + 2
      
      if centerX >= 0 and centerX < MapWidth and centerY >= 0 and centerY < MapHeight:
        let tileColor = env.tileColors[centerX][centerY]
        let baseColor = env.baseTileColors[centerX][centerY]
        
        echo fmt"    House center tile ({centerX},{centerY}):"
        echo fmt"      Current: R={tileColor.r:.2f}, G={tileColor.g:.2f}, B={tileColor.b:.2f}"
        echo fmt"      Base:    R={baseColor.r:.2f}, G={baseColor.g:.2f}, B={baseColor.b:.2f}"
        
        # Check if base color matches team color
        if abs(baseColor.r - teamColor.r) < 0.01 and 
           abs(baseColor.g - teamColor.g) < 0.01 and
           abs(baseColor.b - teamColor.b) < 0.01:
          echo "    ✓ House base color matches team color!"
        else:
          echo "    ✗ House base color does NOT match team color"
        
        # Check if current color is close to team color (may have brightness modification)
        if abs(tileColor.r - teamColor.r) < 0.1 and 
           abs(tileColor.g - teamColor.g) < 0.1 and
           abs(tileColor.b - teamColor.b) < 0.1:
          echo "    ✓ House current color is close to team color"
        else:
          echo "    ⚠ House current color deviates from team color (brightness effect?)"
      
      echo ""

when isMainModule:
  echo "\n" & "=" & repeat("=", 50)
  echo "House Color Test"
  echo "=" & repeat("=", 50) & "\n"
  
  testHouseColors()
  
  echo "=" & repeat("=", 50)
  echo "Test completed"