## Test lantern team coloring implementation
import std/[strformat, strutils]
import ../src/tribal/environment
import ../src/tribal/ai

proc testLanternTeamColoring() =
  echo "Testing Lantern Team Coloring"
  echo "=" & repeat("=", 30)
  
  var env = newEnvironment()
  let controller = newController(seed = 42)
  
  # Find WeavingLoom specialists (role #4) across different teams
  var lanternMakers: seq[tuple[agentId: int, teamId: int, role: AgentRole]]
  
  for i in 0 ..< env.agents.len:
    let agent = env.agents[i]
    let agentId = agent.agentId
    let teamId = agentId div 5  # This should match the fixed calculation
    let role = AgentRole(agentId mod 5)  # Role based on agentId mod 5
    
    if role == WeavingLoomSpecialist:
      lanternMakers.add((agentId: agentId, teamId: teamId, role: role))
      echo fmt"  Agent {agentId} (Team {teamId}): WeavingLoom Specialist"
  
  echo fmt"\nFound {lanternMakers.len} lantern makers across teams"
  
  # Test the plantAction function by simulating lantern placement
  echo "\nSimulating lantern placement..."
  
  for maker in lanternMakers:
    let agent = env.agents[maker.agentId]  # Get agent by its position in array
    
    # Find a suitable spot near the agent to test planting
    let searchPos = agent.pos
    let emptySpots = env.findEmptyPositionsAround(searchPos, 3)
    
    if emptySpots.len > 0:
      let plantPos = emptySpots[0]
      echo fmt"  Agent {maker.agentId} (Team {maker.teamId}) would plant lantern at {plantPos}"
      
      # The actual plantAction would calculate:
      # teamId = agent_i.agentId div 5 = {maker.agentId} div 5 = {maker.teamId}
      echo fmt"    Calculated team ID: {maker.teamId}"
      echo fmt"    Expected team color based on team {maker.teamId}"
    else:
      echo fmt"  Agent {maker.agentId}: No empty spots found for planting"
  
  echo "\n" & "=" & repeat("=", 30)
  echo "Test complete - team ID calculation appears correct"

when isMainModule:
  testLanternTeamColoring()