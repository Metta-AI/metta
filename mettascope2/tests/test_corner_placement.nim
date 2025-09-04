## Test to verify agents spawn at house corners
import ../src/mettascope/tribal
import std/strformat
import vmath

proc visualizeAgentPlacement() =
  let env = newEnvironment()
  
  echo "=== Agent Placement Test - Corners Priority ==="
  echo ""
  
  # Find all houses and their associated agents
  var houseCount = 0
  var housePositions: seq[tuple[topLeft: IVec2, altar: IVec2]] = @[]
  
  for thing in env.things:
    if thing.kind == Altar:
      # Find the house this altar belongs to (altar is at center of 5x5 house)
      let houseTopLeft = ivec2(thing.pos.x - 2, thing.pos.y - 2)
      housePositions.add((topLeft: houseTopLeft, altar: thing.pos))
      houseCount += 1
  
  echo fmt"Found {houseCount} houses with altars"
  echo ""
  
  # For each house, show where agents are relative to it
  for i, house in housePositions:
    echo fmt"House {i+1} at position ({house.topLeft.x}, {house.topLeft.y}):"
    echo fmt"  Altar at ({house.altar.x}, {house.altar.y})"
    
    # Expected corner positions (outside the 5x5 house)
    let expectedCorners = @[
      ivec2(house.topLeft.x - 1, house.topLeft.y - 1),         # Top-left
      ivec2(house.topLeft.x + 5, house.topLeft.y - 1),         # Top-right  
      ivec2(house.topLeft.x - 1, house.topLeft.y + 5),         # Bottom-left
      ivec2(house.topLeft.x + 5, house.topLeft.y + 5)          # Bottom-right
    ]
    
    echo "  Expected corner positions:"
    for j, corner in expectedCorners:
      let cornerName = case j
        of 0: "Top-left"
        of 1: "Top-right"
        of 2: "Bottom-left"
        of 3: "Bottom-right"
        else: "Unknown"
      
      # Check if there's an agent at this corner
      var hasAgent = false
      for thing in env.things:
        if thing.kind == Agent and thing.pos == corner:
          hasAgent = true
          break
      
      let status = if hasAgent: "✓ Agent present" else: "✗ Empty"
      echo fmt"    {cornerName:12} ({corner.x:3}, {corner.y:3}): {status}"
    
    # Find all agents near this house (within 3 tiles of altar)
    echo "  Agents near this house:"
    var agentCount = 0
    for thing in env.things:
      if thing.kind == Agent:
        let dx = abs(thing.pos.x - house.altar.x)
        let dy = abs(thing.pos.y - house.altar.y)
        if dx <= 3 and dy <= 3:
          agentCount += 1
          # Check if at a corner
          var location = "other position"
          for j, corner in expectedCorners:
            if thing.pos == corner:
              location = case j
                of 0: "at TOP-LEFT corner"
                of 1: "at TOP-RIGHT corner"
                of 2: "at BOTTOM-LEFT corner"
                of 3: "at BOTTOM-RIGHT corner"
                else: "at unknown corner"
              break
          
          echo fmt"    Agent {thing.agentId:2} at ({thing.pos.x:3}, {thing.pos.y:3}) - {location}"
    
    echo fmt"  Total agents near house: {agentCount}"
    echo ""

visualizeAgentPlacement()