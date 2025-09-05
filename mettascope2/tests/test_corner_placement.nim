import std/[math, tables, strformat, sequtils, sets, strutils]
import ../src/tribal/tribal
import ../src/tribal/placement
import ../src/tribal/terrain
import vmath
import std/random

proc testCornerPlacement() =
  echo "\n=== Corner Placement Test ==="
  echo "Testing that villages are placed in different corners\n"
  
  # Create multiple environments to test consistency
  for testRun in 1..5:
    var env = newEnvironment()
    
    # Track house positions
    var housePositions: seq[IVec2] = @[]
    var altarPositions: seq[IVec2] = @[]
    
    # Find all altars (centers of houses)
    for thing in env.things:
      if thing.kind == Altar:
        altarPositions.add(thing.pos)
    
    echo fmt"Test run {testRun}:"
    echo fmt"  Found {altarPositions.len} houses/altars"
    
    if altarPositions.len > 0:
      # Determine which corner each house is in
      let mapCenter = ivec2(MapWidth div 2, MapHeight div 2)
      var corners: array[4, seq[IVec2]]  # TL, TR, BL, BR
      
      for pos in altarPositions:
        let cornerIdx = 
          if pos.x < mapCenter.x and pos.y < mapCenter.y: 0  # Top-left
          elif pos.x >= mapCenter.x and pos.y < mapCenter.y: 1  # Top-right
          elif pos.x < mapCenter.x and pos.y >= mapCenter.y: 2  # Bottom-left
          else: 3  # Bottom-right
        
        corners[cornerIdx].add(pos)
      
      # Report distribution
      echo "  Corner distribution:"
      echo fmt"    Top-left: {corners[0].len} houses"
      echo fmt"    Top-right: {corners[1].len} houses"
      echo fmt"    Bottom-left: {corners[2].len} houses"
      echo fmt"    Bottom-right: {corners[3].len} houses"
      
      # Check if houses are in different corners (when possible)
      var occupiedCorners = 0
      for i in 0..3:
        if corners[i].len > 0:
          occupiedCorners += 1
      
      let maxPossibleCorners = min(4, altarPositions.len)
      if occupiedCorners == maxPossibleCorners:
        echo "  ✓ Houses optimally distributed across corners!"
      elif occupiedCorners >= maxPossibleCorners - 1:
        echo "  ✓ Houses mostly well distributed"
      else:
        echo fmt"  ⚠ Houses could be better distributed ({occupiedCorners}/{maxPossibleCorners} corners used)"
      
      # Calculate average distance between houses
      if altarPositions.len > 1:
        var totalDist = 0.0
        var pairCount = 0
        for i in 0 ..< altarPositions.len:
          for j in i + 1 ..< altarPositions.len:
            let dx = (altarPositions[i].x - altarPositions[j].x).float
            let dy = (altarPositions[i].y - altarPositions[j].y).float
            totalDist += sqrt(dx * dx + dy * dy)
            pairCount += 1
        
        let avgDist = totalDist / pairCount.float
        echo fmt"  Average distance between houses: {avgDist:.1f} tiles"
        
        # Good separation is at least 30% of map diagonal
        let mapDiagonal = sqrt((MapWidth * MapWidth + MapHeight * MapHeight).float)
        if avgDist > mapDiagonal * 0.3:
          echo "  ✓ Houses are well separated"
    
    echo ""

proc visualizeCornerRegions() =
  echo "\n=== Corner Region Visualization ==="
  echo "Showing the 4 corner regions on the map\n"
  
  # Create a grid visualization
  const gridSize = 40
  var grid = newSeq[seq[char]](gridSize)
  for i in 0..gridSize-1:
    grid[i] = newSeqWith(gridSize, '.')
  
  # Mark corner regions (25% of map from each corner)
  let cornerSize = gridSize div 4
  
  # Top-left
  for y in 0 ..< cornerSize:
    for x in 0 ..< cornerSize:
      grid[y][x] = '1'
  
  # Top-right
  for y in 0 ..< cornerSize:
    for x in gridSize - cornerSize ..< gridSize:
      grid[y][x] = '2'
  
  # Bottom-left
  for y in gridSize - cornerSize ..< gridSize:
    for x in 0 ..< cornerSize:
      grid[y][x] = '3'
  
  # Bottom-right
  for y in gridSize - cornerSize ..< gridSize:
    for x in gridSize - cornerSize ..< gridSize:
      grid[y][x] = '4'
  
  # Display grid
  echo "Corner regions (1=TL, 2=TR, 3=BL, 4=BR):"
  for row in grid:
    echo row.join("")
  
  echo "\nEach region is 25% of the map size from its corner"

when isMainModule:
  testCornerPlacement()
  visualizeCornerRegions()