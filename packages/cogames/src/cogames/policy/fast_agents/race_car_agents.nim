import
  std/[strformat, strutils, tables, random, sets],
  genny, jsony,
  common

type
  RaceCarAgent* = ref object
    agentId*: int

    map: Table[Location, seq[FeatureValue]]
    seen: HashSet[Location]
    cfg: Config
    random: Rand
    location: Location

proc newRaceCarAgent*(agentId: int, environmentConfig: string): RaceCarAgent {.raises: [].} =
  echo "Creating new heuristic agent ", agentId

  var config = parseConfig(environmentConfig)
  result = RaceCarAgent(agentId: agentId, cfg: config)
  result.random = initRand(agentId)

proc reset*(agent: RaceCarAgent) =
  echo "Resetting heuristic agent ", agent.agentId
  agent.map.clear()
  agent.seen.clear()
  agent.location = Location(x: 0, y: 0)

proc updateMap(agent: RaceCarAgent, visible: Table[Location, seq[FeatureValue]]) =
  ## Update the big map with the small visible map.

  if agent.map.len == 0:
    # First time we're called, just copy the visible map to the big map.
    agent.map = visible
    agent.location = Location(x: 0, y: 0)
    return

  # Try to best guess where the new map is located in the big map.
  # We can only move cardinal directions, so we only need to check staying put
  # and the 4 cardinal directions.
  var
    bestScore = 0
    bestLocation = agent.location
    possibleOffsets: seq[Location]

  # We know we can only have N states, either stay put or move.
  # If the last actions was not known check all possible offsets.
  # Note: Checking all possible offsets does not always work!
  # Last action is nearly required.
  var lastAction = agent.cfg.getLastAction(visible)
  if lastAction == agent.cfg.actions.moveNorth or lastAction == -1: # Moving north.
    possibleOffsets.add(Location(x: 0, y: -1))
  if lastAction == agent.cfg.actions.moveSouth or lastAction == -1: # Moving south.
    possibleOffsets.add(Location(x: 0, y: 1))
  if lastAction == agent.cfg.actions.moveWest or lastAction == -1: # Moving west.
    possibleOffsets.add(Location(x: -1, y: 0))
  if lastAction == agent.cfg.actions.moveEast or lastAction == -1: # Moving east.
    possibleOffsets.add(Location(x: 1, y: 0))
  possibleOffsets.add(Location(x: 0, y: 0)) # Staying put.

  for offset in possibleOffsets:
    var score = 0
    let location = Location(x: agent.location.x + offset.x, y: agent.location.y + offset.y)
    for x in -5 .. 5:
      for y in -5 .. 5:
        let visibleTag = agent.cfg.getTag(visible, Location(x: x, y: y))
        let mapTag = agent.cfg.getTag(agent.map, Location(x: x + location.x, y: y + location.y))
        if visibleTag == mapTag:
          if visibleTag == agent.cfg.tags.agent or visibleTag == -1:
            # No points for empty or agent locations.
            discard
          elif visibleTag == agent.cfg.tags.assembler:
            # There is only one assembler per map, so this is a good score.
            score += 100
          elif visibleTag == agent.cfg.tags.wall:
            # Walls can repeat and cause issues, so not worth much.
            score += 1
          else:
            # Other types of features are worth 1 point each.
            score += 10
    if score > bestScore:
      bestScore = score
      bestLocation = location

  # Update the big map with the small visible map.
  if bestScore < 2:
    echo "Looks like we are lost?"
    echo "  current location: ", agent.location.x, ", ", agent.location.y
    echo "  best location: ", bestLocation.x, ", ", bestLocation.y
  else:
    agent.location =  bestLocation
    for x in -5 .. 5:
      for y in -5 .. 5:
        let visibleLocation = Location(x: x, y: y)
        let location = Location(x: x + agent.location.x, y: y + agent.location.y)
        if visibleLocation in visible:
          agent.map[location] = visible[visibleLocation]
        else:
          agent.map[location] = @[]
        agent.seen.incl(location)

proc raceCarStepInternal(
  agent: RaceCarAgent,
  numTokens: int,
  sizeToken: int,
  rawObservation: pointer,
  actions: ptr UncheckedArray[int32],
  actionIndex: int
) {.raises: [].} =
  try:
    echo "Driving race car agent ", agent.agentId
    # echo "  numAgents", numAgents
    # echo "  numTokens", numTokens
    # echo "  sizeToken", sizeToken
    # echo "  numActions", numActions
    let observations = cast[ptr UncheckedArray[uint8]](rawObservation)

    var map: Table[Location, seq[FeatureValue]]
    for token in 0 ..< numTokens:
      let locationPacked = observations[token * sizeToken]
      let featureId = observations[token * sizeToken + 1]
      let value = observations[token * sizeToken + 2]
      if locationPacked == 255 and featureId == 255 and value == 255:
        break
      var location: Location
      if locationPacked != 0xFF:
        location.y = (locationPacked shr 4).int - 5
        location.x = (locationPacked and 0x0F).int - 5
      #echo "  token ", token, " loc ", location.x, ", ", location.y, " featureId ", featureId, " value ", value
      if location notin map:
        map[location] = @[]
      map[location].add(FeatureValue(featureId: featureId.int, value: value.int))

    echo "current location: ", agent.location.x, ", ", agent.location.y
    echo "visible map:"
    agent.cfg.drawMap(map, initHashSet[Location]())
    updateMap(agent, map)
    echo "updated map:"
    agent.cfg.drawMap(agent.map, agent.seen)

    let vibe = agent.cfg.getVibe(map)
    echo "vibe: ", vibe

    let invEnergy = agent.cfg.getInventory(map, agent.cfg.features.invEnergy)
    let invCarbon = agent.cfg.getInventory(map, agent.cfg.features.invCarbon)
    let invOxygen = agent.cfg.getInventory(map, agent.cfg.features.invOxygen)
    let invGermanium = agent.cfg.getInventory(map, agent.cfg.features.invGermanium)
    let invSilicon = agent.cfg.getInventory(map, agent.cfg.features.invSilicon)
    let invHeart = agent.cfg.getInventory(map, agent.cfg.features.invHeart)
    let invDecoder = agent.cfg.getInventory(map, agent.cfg.features.invDecoder)
    let invModulator = agent.cfg.getInventory(map, agent.cfg.features.invModulator)
    let invResonator = agent.cfg.getInventory(map, agent.cfg.features.invResonator)
    let invScrambler = agent.cfg.getInventory(map, agent.cfg.features.invScrambler)

    echo &"H:{invHeart} E:{invEnergy} C:{invCarbon} O2:{invOxygen} Ge:{invGermanium} Si:{invSilicon} D:{invDecoder} M:{invModulator} R:{invResonator} S:{invScrambler}"

    let action = agent.random.rand(1 .. 4).int32
    actions[actionIndex] = action
    echo "taking action ", action

  except:
    echo getCurrentException().getStackTrace()
    echo getCurrentExceptionMsg()
    quit()

proc step*(
  agent: RaceCarAgent,
  numAgents: int,
  numTokens: int,
  sizeToken: int,
  rawObservations: pointer,
  numActions: int,
  rawActions: pointer
) {.raises: [].} =
  discard numAgents
  discard numActions
  let observations = cast[ptr UncheckedArray[uint8]](rawObservations)
  let actions = cast[ptr UncheckedArray[int32]](rawActions)
  let agentObservation = cast[pointer](observations[agent.agentId * numTokens * sizeToken].addr)
  raceCarStepInternal(agent, numTokens, sizeToken, agentObservation, actions, agent.agentId)
