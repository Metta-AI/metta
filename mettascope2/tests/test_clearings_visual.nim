## Test to visualize terrain clearings around structures
import ../src/mettascope/tribal
import vmath

proc visualizeClearings() =
  let env = newEnvironment()
  
  echo "=== Terrain Clearing Visualization ==="
  echo "Legend: . = wheat, t = tree, ~ = water, # = wall, a = altar, A = agent"
  echo "        (empty) = cleared ground, T = temple, C = Clippy"
  echo ""
  echo "Notice how houses and temples create clearings in the terrain:"
  echo ""
  
  # Create a zoomed-in view around the first house
  var firstHouse: IVec2
  var foundHouse = false
  
  for thing in env.things:
    if thing.kind == Altar and not foundHouse:
      # Altar is at center of 5x5 house, so house top-left is -2, -2
      firstHouse = ivec2(thing.pos.x - 2, thing.pos.y - 2)
      foundHouse = true
      break
  
  if foundHouse:
    echo "Close-up of a house (10x10 area around it):"
    echo "+" & "-".repeat(20) & "+"
    
    for dy in -3 .. 7:
      var line = "|"
      for dx in -3 .. 7:
        let x = firstHouse.x + dx
        let y = firstHouse.y + dy
        
        if x < 0 or x >= MapWidth or y < 0 or y >= MapHeight:
          line.add(" ")
          continue
        
        # Check for objects first
        var hasObject = false
        for thing in env.things:
          if thing.pos.x == x and thing.pos.y == y:
            case thing.kind:
            of Agent: line.add("A")
            of Wall: line.add("#")
            of Altar: line.add("a")
            of Temple: line.add("T")
            of Clippy: line.add("C")
            of Generator: line.add("G")
            of Mine: line.add("M")
            hasObject = true
            break
        
        if not hasObject:
          # Show terrain
          case env.terrain[x][y]:
          of Water: line.add("~")
          of Wheat: line.add(".")
          of Tree: line.add("t")
          of Empty: line.add(" ")  # Cleared area shows as empty
      
      line.add("|")
      
      # Add annotation for house rows
      if dy >= 0 and dy < 5:
        line.add(" <- House row " & $(dy + 1))
      
      echo line
    
    echo "+" & "-".repeat(20) & "+"
    echo ""
    echo "Notice: The 5x5 house area is completely clear of terrain"
    echo "        Trees and wheat that would have been inside are removed"
  
  # Show full map
  echo ""
  echo "Full map:"
  echo env.render()

visualizeClearings()