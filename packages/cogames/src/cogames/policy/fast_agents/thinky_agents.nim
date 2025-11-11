import
  std/[strformat, strutils, tables, random, sets, options],
  genny, jsony,
  common

type
  ThinkyAgent* = ref object
    agentId*: int

    map: Table[Location, seq[FeatureValue]]
    seen: HashSet[Location]
    cfg: Config
    random: Rand
    location: Location

proc newThinkyAgent*(agentId: int, environmentConfig: string): ThinkyAgent {.raises: [].} =
  #echo "Creating new heuristic agent ", agentId

  var config = parseConfig(environmentConfig)
  result = ThinkyAgent(agentId: agentId, cfg: config)
  result.random = initRand(agentId)

proc reset*(agent: ThinkyAgent) =
  #echo "Resetting heuristic agent ", agent.agentId
  agent.map.clear()
  agent.seen.clear()
  agent.location = Location(x: 0, y: 0)

proc updateMap(agent: ThinkyAgent, visible: Table[Location, seq[FeatureValue]]) =
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
  # Note: Checking all possible offsets does not allows work!
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
    # echo "Looks like we are lost?"
    # echo "  current location: ", agent.location.x, ", ", agent.location.y
    # echo "  best location: ", bestLocation.x, ", ", bestLocation.y
    discard
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

proc thinkyStepInternal(
  agent: ThinkyAgent,
  numTokens: int,
  sizeToken: int,
  rawObservation: pointer,
  actions: ptr UncheckedArray[int32],
  actionIndex: int
) {.raises: [].} =
  try:
    # echo "Thinking heuristic agent ", agent.agentId
    let observations = cast[ptr UncheckedArray[uint8]](rawObservation)

    var map: Table[Location, seq[FeatureValue]]
    for token in 0 ..< numTokens:
      let baseIdx = token * sizeToken
      let locationPacked = observations[baseIdx]
      let featureId = observations[baseIdx + 1]
      let value = observations[baseIdx + 2]
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

    #echo "current location: ", agent.location.x, ", ", agent.location.y
    #echo "visible map:"
    #agent.cfg.drawMap(map, initHashSet[Location]())
    updateMap(agent, map)
    #echo "updated map:"
    #agent.cfg.drawMap(agent.map, agent.seen)

    let vibe = agent.cfg.getVibe(map)
    #echo "vibe: ", vibe

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

    #echo &"H:{invHeart} E:{invEnergy} C:{invCarbon} O2:{invOxygen} Ge:{invGermanium} Si:{invSilicon} D:{invDecoder} M:{invModulator} R:{invResonator} S:{invScrambler}"

    # If not vibing here at heart then vibe heart.
    if vibe != agent.cfg.vibes.heart:
      actions[actionIndex] = agent.cfg.actions.vibeHeart.int32
      #echo "vibing heart"
      return

    # Is there an energy charger nearby?
    let chargerNearby = agent.cfg.getNearby(agent.location, agent.map, agent.cfg.tags.charger)
    if invEnergy < 100 and chargerNearby.isSome():
      let action = agent.cfg.aStar(agent.location, chargerNearby.get(), agent.map)
      if action.isSome():
        actions[actionIndex] = action.get().int32
        #echo "going to charger"
        return

    # Explore locations around the assembler.
    let keyLocations = [
      Location(x: -10, y: -10),
      Location(x: -10, y: +10),
      Location(x: +10, y: -10),
      Location(x: +10, y: +10),
    ]
    let assemblerNearby = agent.cfg.getNearby(agent.location, agent.map, agent.cfg.tags.assembler)
    if assemblerNearby.isSome():
      for keyLocation in keyLocations:
        let location = assemblerNearby.get() + keyLocation
        if location notin agent.seen:
          let action = agent.cfg.aStar(agent.location, location, agent.map)
          if action.isSome():
            actions[actionIndex] = action.get().int32
            #echo "going to key location to explore"
            return

    # Is there carbon nearby?
    let carbonNearby = agent.cfg.getNearby(agent.location, agent.map, agent.cfg.tags.carbonExtractor)
    if invCarbon == 0 and carbonNearby.isSome():
      let action = agent.cfg.aStar(agent.location, carbonNearby.get(), agent.map)
      if action.isSome():
        actions[actionIndex] = action.get().int32
        #echo "going to carbon"
        return

    # Is there oxygen nearby?
    let oxygenNearby = agent.cfg.getNearby(agent.location, agent.map, agent.cfg.tags.oxygenExtractor)
    if invOxygen == 0 and oxygenNearby.isSome():
      let action = agent.cfg.aStar(agent.location, oxygenNearby.get(), agent.map)
      if action.isSome():
        actions[actionIndex] = action.get().int32
        #echo "going to oxygen"
        return

    # Is there germanium nearby?
    let germaniumNearby = agent.cfg.getNearby(agent.location, agent.map, agent.cfg.tags.germaniumExtractor)
    if invGermanium == 0 and germaniumNearby.isSome():
      let action = agent.cfg.aStar(agent.location, germaniumNearby.get(), agent.map)
      if action.isSome():
        actions[actionIndex] = action.get().int32
        #echo "going to germanium"
        return

    # Is there silicon nearby?
    let siliconNearby = agent.cfg.getNearby(agent.location, agent.map, agent.cfg.tags.siliconExtractor)
    if invSilicon == 0 and siliconNearby.isSome():
      let action = agent.cfg.aStar(agent.location, siliconNearby.get(), agent.map)
      if action.isSome():
        actions[actionIndex] = action.get().int32
        #echo "going to silicon"
        return

    if invSilicon > 0 and invOxygen > 0 and invCarbon > 0 and invGermanium > 0:
      # We have all the resources we need, so we can build an assembler.
      let assemblerNearby = agent.cfg.getNearby(agent.location, agent.map, agent.cfg.tags.assembler)
      if assemblerNearby.isSome():
        let action = agent.cfg.aStar(agent.location, assemblerNearby.get(), agent.map)
        if action.isSome():
          actions[actionIndex] = action.get().int32
          #echo "going to assembler to build heart"
          return

    # Try to find a nearby unseen location nearest to the agent.
    let unseenNearby = agent.cfg.getNearbyUnseen(agent.location, agent.map, agent.seen)
    if unseenNearby.isSome():
      let action = agent.cfg.aStar(agent.location, unseenNearby.get(), agent.map)
      if action.isSome():
        actions[actionIndex] = action.get().int32
        #echo "going to unseen location nearest to agent"
        return

    # # Find the nearest unexplored location to the assembler.
    # let assemblerNearby = agent.cfg.getNearby(agent.location, agent.map, agent.cfg.tags.assembler)
    # if assemblerNearby.isSome():
    #   echo "assembler nearby"
    #   let unseenNearby = agent.cfg.getNearbyUnseen(assemblerNearby.get(), agent.map, agent.seen)
    #   if unseenNearby.isSome():
    #     echo "unseen nearby"
    #     let action = agent.cfg.aStar(agent.location, unseenNearby.get(), agent.map)
    #     if action.isSome():
    #       actions[actionIndex] = action.get().int32
    #       echo "going to unseen location"
    #       return

    # If all else fails, take a random move to explore the map or get unstuck.
    let action = agent.random.rand(1 .. 4).int32
    actions[actionIndex] = action
    #echo "taking random action ", action

  except:
    echo getCurrentException().getStackTrace()
    echo getCurrentExceptionMsg()
    quit()

proc step*(
  agent: ThinkyAgent,
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
  thinkyStepInternal(agent, numTokens, sizeToken, agentObservation, actions, agent.agentId)
