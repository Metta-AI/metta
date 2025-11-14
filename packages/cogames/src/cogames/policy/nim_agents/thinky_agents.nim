import
  std/[strformat, tables, random, sets, options, json],
  common

type
  ThinkyAgent* = ref object
    agentId*: int

    map: Table[Location, seq[FeatureValue]]
    seen: HashSet[Location]
    cfg: Config
    random: Rand
    location: Location
    lastActions: seq[int]

  ThinkyPolicy* = ref object
    agents*: seq[ThinkyAgent]

proc getAssemblerRequirements(agent: ThinkyAgent): Table[string, int] =
  ## Read the assembler protocol inputs from the cached map to understand heart costs.
  result = initTable[string, int]()
  let assemblerLocation = agent.cfg.getNearby(agent.location, agent.map, agent.cfg.tags.assembler)
  if assemblerLocation.isNone():
    return
  let location = assemblerLocation.get()
  if location notin agent.map:
    return
  let features = agent.map[location]

  var heartProtocolSeen = false
  for featureValue in features:
    for resource, featureId in agent.cfg.features.protocolOutputs.pairs:
      if resource == "heart" and featureValue.featureId == featureId and featureValue.value > 0:
        heartProtocolSeen = true
        break
    if heartProtocolSeen:
      break

  if not heartProtocolSeen:
    return

  for resource, featureId in agent.cfg.features.protocolInputs.pairs:
    for featureValue in features:
      if featureValue.featureId == featureId and featureValue.value > 0:
        result[resource] = featureValue.value
        break

proc requirementOrFallback(
  requirements: Table[string, int],
  resource: string,
  fallback: int
): int =
  ## Return the required amount for a resource, defaulting when the assembler recipe is unknown.
  if requirements.hasKey(resource):
    let value = requirements[resource]
    if value > 0:
      return value
    return 0
  return fallback

proc hasResourcesForHeart(
  requirements: Table[string, int],
  invEnergy: int,
  invCarbon: int,
  invOxygen: int,
  invGermanium: int,
  invSilicon: int
): bool =
  ## Determine if the agent currently holds enough materials (and energy) to build the next heart.
  if requirements.len == 0:
    return invCarbon > 0 and invOxygen > 0 and invGermanium > 0 and invSilicon > 0

  for resource, required in requirements.pairs:
    if required <= 0:
      continue
    case resource
    of "carbon":
      if invCarbon < required: return false
    of "oxygen":
      if invOxygen < required: return false
    of "germanium":
      if invGermanium < required: return false
    of "silicon":
      if invSilicon < required: return false
    of "energy":
      if invEnergy < required: return false
    else:
      discard
  return true

proc newThinkyAgent*(agentId: int, environmentConfig: string): ThinkyAgent =
  #echo "Creating new heuristic agent ", agentId

  var config = parseConfig(environmentConfig)
  result = ThinkyAgent(agentId: agentId, cfg: config)
  result.random = initRand(agentId)
  result.map = initTable[Location, seq[FeatureValue]]()
  result.seen = initHashSet[Location]()
  result.location = Location(x: 0, y: 0)
  result.lastActions = @[]

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

proc step*(
  agent: ThinkyAgent,
  numAgents: int,
  numTokens: int,
  sizeToken: int,
  rawObservation: pointer,
  numActions: int,
  agentAction: ptr int32
) =
  try:
    discard numAgents
    discard numActions
    # echo "  numTokens", numTokens
    # echo "  sizeToken", sizeToken
    # echo "  numActions", numActions
    let observations = cast[ptr UncheckedArray[uint8]](rawObservation)

    proc doAction(action: int) =

      # Stuck prevention: if last 2 actions are left, right and this is left.
      if agent.lastActions.len >= 2 and
        action == agent.cfg.actions.moveWest and
        agent.lastActions[^1] == agent.cfg.actions.moveEast and
        agent.lastActions[^2] == agent.cfg.actions.moveWest:
        # Noop 50% of the time.
        if agent.random.rand(1 .. 2) == 1:
          # echo "Stuck prevention: left, right, left"
          doAction(agent.cfg.actions.noop.int32)
          return
      if agent.lastActions.len >= 2 and
        action == agent.cfg.actions.moveEast and
        agent.lastActions[^1] == agent.cfg.actions.moveWest and
        agent.lastActions[^2] == agent.cfg.actions.moveEast:
        # Noop 50% of the time.
        if agent.random.rand(1 .. 2) == 1:
          # echo "Stuck prevention: right, left, right"
          doAction(agent.cfg.actions.noop.int32)
          return
      if agent.lastActions.len >= 2 and
        action == agent.cfg.actions.moveNorth and
        agent.lastActions[^1] == agent.cfg.actions.moveSouth and
        agent.lastActions[^2] == agent.cfg.actions.moveNorth:
        # Noop 50% of the time.
        if agent.random.rand(1 .. 2) == 1:
          # echo "Stuck prevention: north, south, north"
          doAction(agent.cfg.actions.noop.int32)
          return
      if agent.lastActions.len >= 2 and
        action == agent.cfg.actions.moveSouth and
        agent.lastActions[^1] == agent.cfg.actions.moveNorth and
        agent.lastActions[^2] == agent.cfg.actions.moveSouth:
        # Noop 50% of the time.
        if agent.random.rand(1 .. 2) == 1:
          # echo "Stuck prevention: south, north, south"
          doAction(agent.cfg.actions.noop.int32)
          return

      agent.lastActions.add(action)
      agentAction[] = action.int32

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

    let heartRequirements = agent.getAssemblerRequirements()
    let carbonTarget = requirementOrFallback(heartRequirements, "carbon", 1)
    let oxygenTarget = requirementOrFallback(heartRequirements, "oxygen", 1)
    let germaniumTarget = requirementOrFallback(heartRequirements, "germanium", 1)
    let siliconTarget = requirementOrFallback(heartRequirements, "silicon", 1)

    # Is there an energy charger nearby?
    let chargerNearby = agent.cfg.getNearby(agent.location, agent.map, agent.cfg.tags.charger)
    if invEnergy < 50 and chargerNearby.isSome():
      let action = agent.cfg.aStar(agent.location, chargerNearby.get(), agent.map)
      if action.isSome():
        doAction(action.get().int32)
        #echo "going to charger"
        return

    # Deposit heart into the chest.
    if invHeart > 0:
      let depositAction = agent.cfg.actions.vibeHeartB
      let depositVibe = agent.cfg.vibes.heartB
      if depositAction != 0 and vibe != depositVibe:
        doAction(depositAction.int32)
        # echo "adjusting vibe for heart deposit"
        return
      let chestNearby = agent.cfg.getNearby(agent.location, agent.map, agent.cfg.tags.chest)
      if chestNearby.isSome():
        let action = agent.cfg.aStar(agent.location, chestNearby.get(), agent.map)
        if action.isSome():
          doAction(action.get().int32)
          # echo "going to chest"
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
            doAction(action.get().int32)
            #echo "going to key location to explore"
            return

    # See if there is a chest nearby and it has resources we need.
    # TODO: Waiting for chest features to be fixed.
    # let chestNearby = agent.cfg.getNearby(agent.location, agent.map, agent.cfg.tags.chest)
    # if chestNearby.isSome():
    #   echo "chest features: ", agent.map[chestNearby.get()]
    #   let
    #     chestCarbon = agent.cfg.getOtherInventory(agent.map, chestNearby.get(), agent.cfg.features.invCarbon)
    #     chestOxygen = agent.cfg.getOtherInventory(agent.map, chestNearby.get(), agent.cfg.features.invOxygen)
    #     chestGermanium = agent.cfg.getOtherInventory(agent.map, chestNearby.get(), agent.cfg.features.invGermanium)
    #     chestSilicon = agent.cfg.getOtherInventory(agent.map, chestNearby.get(), agent.cfg.features.invSilicon)
    #     chestHeart = agent.cfg.getOtherInventory(agent.map, chestNearby.get(), agent.cfg.features.invHeart)
    #   echo "chest nearby C ", chestCarbon, " O2 ", chestOxygen, " Ge ", chestGermanium, " Si ", chestSilicon, " H ", chestHeart
    #   if invCarbon == 0 and chestCarbon > 0:
    #     echo "carbon in chest, i want it"
    #     if vibe != agent.cfg.vibes.carbon:
    #       doAction(agent.cfg.actions.vibeCarbon.int32)
    #       echo "vibing carbon"
    #       return
    #     let action = agent.cfg.aStar(agent.location, chestNearby.get(), agent.map)
    #     if action.isSome():
    #       doAction(action.get().int32)
    #       echo "going to chest to get carbon"
    #       return

    # Is there carbon nearby?
    if carbonTarget > 0 and invCarbon < carbonTarget:
      let carbonNearby = agent.cfg.getNearby(agent.location, agent.map, agent.cfg.tags.carbonExtractor)
      if carbonNearby.isSome():
        let action = agent.cfg.aStar(agent.location, carbonNearby.get(), agent.map)
        if action.isSome():
          doAction(action.get().int32)
          #echo "going to carbon"
          return

    # Is there oxygen nearby?
    if oxygenTarget > 0 and invOxygen < oxygenTarget:
      let oxygenNearby = agent.cfg.getNearby(agent.location, agent.map, agent.cfg.tags.oxygenExtractor)
      if oxygenNearby.isSome():
        let action = agent.cfg.aStar(agent.location, oxygenNearby.get(), agent.map)
        if action.isSome():
          doAction(action.get().int32)
          #echo "going to oxygen"
          return

    # Is there germanium nearby?
    if germaniumTarget > 0 and invGermanium < germaniumTarget:
      let germaniumNearby = agent.cfg.getNearby(agent.location, agent.map, agent.cfg.tags.germaniumExtractor)
      if germaniumNearby.isSome():
        let action = agent.cfg.aStar(agent.location, germaniumNearby.get(), agent.map)
        if action.isSome():
          doAction(action.get().int32)
          #echo "going to germanium"
          return

    # Is there silicon nearby?
    if siliconTarget > 0 and invSilicon < siliconTarget:
      let siliconNearby = agent.cfg.getNearby(agent.location, agent.map, agent.cfg.tags.siliconExtractor)
      if siliconNearby.isSome():
        let action = agent.cfg.aStar(agent.location, siliconNearby.get(), agent.map)
        if action.isSome():
          doAction(action.get().int32)
          #echo "going to silicon"
          return

    if hasResourcesForHeart(heartRequirements, invEnergy, invCarbon, invOxygen, invGermanium, invSilicon):
      # We have all the resources we need, so we can build an assembler.

      var assembleAction = agent.cfg.actions.vibeHeartA
      var assembleVibe = agent.cfg.vibes.heartA
      if assembleAction == 0:
        assembleAction = agent.cfg.actions.vibeHeartB
        assembleVibe = agent.cfg.vibes.heartB
      if assembleAction != 0 and vibe != assembleVibe:
        doAction(assembleAction.int32)
        # echo "vibing heart for assembler"
        return

      let assemblerNearby = agent.cfg.getNearby(agent.location, agent.map, agent.cfg.tags.assembler)
      if assemblerNearby.isSome():
        let action = agent.cfg.aStar(agent.location, assemblerNearby.get(), agent.map)
        if action.isSome():
          doAction(action.get().int32)
          #echo "going to assembler to build heart"
          return

    # Try to find a nearby unseen location nearest to the agent.
    let unseenNearby = agent.cfg.getNearbyUnseen(agent.location, agent.map, agent.seen)
    if unseenNearby.isSome():
      let action = agent.cfg.aStar(agent.location, unseenNearby.get(), agent.map)
      if action.isSome():
        doAction(action.get().int32)
        #echo "going to unseen location nearest to agent"
        return

    # If all else fails, take a random move to explore the map or get unstuck.
    let action = agent.random.rand(1 .. 4).int32
    doAction(action.int32)
    #echo "taking random action ", action

  except:
    echo getCurrentException().getStackTrace()
    echo getCurrentExceptionMsg()
    quit()


proc newThinkyPolicy*(environmentConfig: string): ThinkyPolicy =
  let cfg = parseConfig(environmentConfig)
  var agents: seq[ThinkyAgent] = @[]
  for id in 0 ..< cfg.config.numAgents:
    agents.add(newThinkyAgent(id, environmentConfig))
  ThinkyPolicy(agents: agents)

proc stepBatch*(
    policy: ThinkyPolicy,
    agentIds: pointer,
    numAgentIds: int,
    numAgents: int,
    numTokens: int,
    sizeToken: int,
    rawObservations: pointer,
    numActions: int,
    rawActions: pointer
) =
  let ids = cast[ptr UncheckedArray[int32]](agentIds)
  let obsArray = cast[ptr UncheckedArray[uint8]](rawObservations)
  let actionArray = cast[ptr UncheckedArray[int32]](rawActions)
  let obsStride = numTokens * sizeToken

  for i in 0 ..< numAgentIds:
    let idx = int(ids[i])
    let obsPtr = cast[pointer](obsArray[idx * obsStride].addr)
    let actPtr = cast[ptr int32](actionArray[idx].addr)
    step(policy.agents[idx], numAgents, numTokens, sizeToken, obsPtr, numActions, actPtr)
