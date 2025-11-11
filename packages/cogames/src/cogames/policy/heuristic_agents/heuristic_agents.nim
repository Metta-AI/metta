import
  std/[strformat, strutils, tables, random, sets],
  genny, jsony

when not defined(heuristic_logging):
  template echo(args: varargs[untyped]) = discard

type

  ConfigFeature* = object
    id*: int
    name*: string
    normalization*: float

  Config* = object
    numAgents*: int
    obsWidth*: int
    obsHeight*: int
    actions*: seq[string]
    tags*: seq[string]
    obsFeatures*: seq[ConfigFeature]

  FeatureValue* = object
    featureId*: int
    value*: int

  Location* = object
    x*: int
    y*: int

  MapBounds* = object
    minX*: int
    maxX*: int
    minY*: int
    maxY*: int

  Actions = object
    noop*: int
    moveNorth*: int
    moveSouth*: int
    moveWest*: int
    moveEast*: int
    vibeDefault*: int
    vibeCharger*: int
    vibeCarbon*: int
    vibeOxygen*: int
    vibeGermanium*: int
    vibeSilicon*: int
    vibeHeart*: int
    vibeGear*: int
    vibeAssembler*: int
    vibeChest*: int
    vibeWall*: int

  Tags = object
    agent*: int
    assembler*: int
    carbonExtractor*: int
    charger*: int
    chest*: int
    germaniumExtractor*: int
    oxygenExtractor*: int
    siliconExtractor*: int
    wall*: int

  Features = object
    group*: int
    frozen*: int
    orientation*: int
    reservedForFutureUse*: int
    converting*: int
    swappable*: int
    episodeCompletionPct*: int
    lastAction*: int
    lastActionArg*: int
    lastReward*: int
    vibe*: int
    visitationCounts*: int
    tag*: int
    cooldownRemaining*: int
    clipped*: int
    remainingUses*: int
    invEnergy*: int
    invCarbon*: int
    invOxygen*: int
    invGermanium*: int
    invSilicon*: int
    invHeart*: int
    invDecoder*: int
    invModulator*: int
    invResonator*: int
    invScrambler*: int

  HeuristicAgent* = ref object
    agentId: int

    map: Table[Location, seq[FeatureValue]]
    seen: HashSet[Location]
    config: Config
    actions: Actions
    features: Features
    tags: Tags
    random: Rand
    location: Location

proc ctrlCHandler() {.noconv.} =
  ## Handle ctrl-c signal to exit cleanly.
  echo "\nNim DLL caught ctrl-c, exiting..."
  quit(0)

proc initCHook() =
  setControlCHook(ctrlCHandler)
  echo "HeuristicAgents initialized"

proc newHeuristicAgent(agentId: int, environmentConfig: string): HeuristicAgent {.raises: [].} =
  echo "Creating new heuristic agent ", agentId
  try:
    var config = environmentConfig.fromJson(Config)
    result = HeuristicAgent(agentId: agentId, config: config)
    echo "  numAgents", config.numAgents
    echo "  obsWidth", config.obsWidth
    echo "  obsHeight", config.obsHeight
    echo "  actions", config.actions
    echo "  tags", config.tags
    echo "  obsFeatures", config.obsFeatures

    for feature in config.obsFeatures:
      echo "    feature ", feature.id, " ", feature.name, " ", feature.normalization
      case feature.name:
      of "agent:group":
        result.features.group = feature.id
      of "agent:frozen":
        result.features.frozen = feature.id
      of "agent:orientation":
        result.features.orientation = feature.id
      of "agent:reserved_for_future_use":
        result.features.reservedForFutureUse = feature.id
      of "converting":
        result.features.converting = feature.id
      of "swappable":
        result.features.swappable = feature.id
      of "episode_completion_pct":
        result.features.episodeCompletionPct = feature.id
      of "last_action":
        result.features.lastAction = feature.id
      of "last_action_arg":
        result.features.lastActionArg = feature.id
      of "last_reward":
        result.features.lastReward = feature.id
      of "vibe":
        result.features.vibe = feature.id
      of "agent:visitation_counts":
        result.features.visitationCounts = feature.id
      of "tag":
        result.features.tag = feature.id
      of "cooldown_remaining":
        result.features.cooldownRemaining = feature.id
      of "clipped":
        result.features.clipped = feature.id
      of "remaining_uses":
        result.features.remainingUses = feature.id
      of "inv:energy":
        result.features.invEnergy = feature.id
      of "inv:carbon":
        result.features.invCarbon = feature.id
      of "inv:oxygen":
        result.features.invOxygen = feature.id
      of "inv:germanium":
        result.features.invGermanium = feature.id
      of "inv:silicon":
        result.features.invSilicon = feature.id
      of "inv:heart":
        result.features.invHeart = feature.id
      of "inv:decoder":
        result.features.invDecoder = feature.id
      of "inv:modulator":
        result.features.invModulator = feature.id
      of "inv:resonator":
        result.features.invResonator = feature.id
      of "inv:scrambler":
        result.features.invScrambler = feature.id
      else:
        echo "Unknown feature: ", feature.name

    for id, name in config.actions:
      case name:
      of "noop":
        result.actions.noop = id
      of "move_north":
        result.actions.moveNorth = id
      of "move_south":
        result.actions.moveSouth = id
      of "move_west":
        result.actions.moveWest = id
      of "move_east":
        result.actions.moveEast = id
      of "change_vibe_default":
        result.actions.vibeDefault = id
      of "change_vibe_charger":
        result.actions.vibeCharger = id
      of "change_vibe_carbon":
        result.actions.vibeCarbon = id
      of "change_vibe_oxygen":
        result.actions.vibeOxygen = id
      of "change_vibe_germanium":
        result.actions.vibeGermanium = id
      of "change_vibe_silicon":
        result.actions.vibeSilicon = id
      of "change_vibe_heart":
        result.actions.vibeHeart = id
      of "change_vibe_gear":
        result.actions.vibeGear = id
      of "change_vibe_assembler":
        result.actions.vibeAssembler = id
      of "change_vibe_chest":
        result.actions.vibeChest = id
      of "change_vibe_wall":
        result.actions.vibeWall = id
      else:
        discard
    echo "  actions", result.actions
    for id, name in config.tags:
      echo "    tag name ", name, " id ", id
      case name:
      of "agent":
        result.tags.agent = id
      of "assembler":
        result.tags.assembler = id
      of "carbon_extractor":
        result.tags.carbonExtractor = id
      of "charger":
        result.tags.charger = id
      of "chest":
        result.tags.chest = id
      of "germanium_extractor":
        result.tags.germaniumExtractor = id
      of "oxygen_extractor":
        result.tags.oxygenExtractor = id
      of "silicon_extractor":
        result.tags.siliconExtractor = id
      of "wall":
        result.tags.wall = id
      else:
        discard
    echo "  types_names", result.tags
    result.random = initRand(agentId)
  except JsonError, ValueError:
    echo "Error parsing environment config: ", getCurrentExceptionMsg()

proc reset(agent: HeuristicAgent) =
  echo "Resetting heuristic agent ", agent.agentId

proc computeMapBounds(map: Table[Location, seq[FeatureValue]]): MapBounds =
  ## Compute the bounds of the map.
  result.minX = -5
  result.maxX = 5
  result.minY = -5
  result.maxY = 5
  for location, featureValues in map:
    if location.x < result.minX:
      result.minX = location.x
    if location.x > result.maxX:
      result.maxX = location.x
    if location.y < result.minY:
      result.minY = location.y
    if location.y > result.maxY:
      result.maxY = location.y

proc drawMap(agent: HeuristicAgent, map: Table[Location, seq[FeatureValue]], seen: HashSet[Location]) =
  ## Draw the map to the console.
  let bounds = computeMapBounds(map)
  var line = "+"
  for x in bounds.minX .. bounds.maxX:
    line.add "--"
  line.add "+"
  echo line
  for y in bounds.minY .. bounds.maxY:
    line = "|"
    for x in bounds.minX .. bounds.maxX:
      var cell = "  "
      let location = Location(x: x, y: y)
      if location notin seen:
        cell = "~~"
      if location in map:
        for featureValue in map[location]:
          if featureValue.featureId == agent.features.tag:
            if featureValue.value == agent.tags.agent:
              cell = "@@"
            elif featureValue.value == agent.tags.assembler:
              cell = "As"
            elif featureValue.value == agent.tags.carbonExtractor:
              cell = "Ca"
            elif featureValue.value == agent.tags.charger:
              cell = "En"
            elif featureValue.value == agent.tags.chest:
              cell = "Ch"
            elif featureValue.value == agent.tags.germaniumExtractor:
              cell = "Ge"
            elif featureValue.value == agent.tags.oxygenExtractor:
              cell = "O2"
            elif featureValue.value == agent.tags.siliconExtractor:
              cell = "Si"
            elif featureValue.value == agent.tags.wall:
              cell = "##"
            else:
              cell = &"{featureValue.value:2d}"
      line.add cell
    line.add "|"
    echo line
  line = "+"
  for x in bounds.minX .. bounds.maxX:
    line.add "--"
  line.add "+"
  echo line

proc getTag(agent: HeuristicAgent, map: Table[Location, seq[FeatureValue]], location: Location): int =
  ## Get the type id of the location in the map.
  if location in map:
    for featureValue in map[location]:
      if featureValue.featureId == agent.features.tag:
        return featureValue.value
  return -1

proc getFeature(agent: HeuristicAgent, visible: Table[Location, seq[FeatureValue]], featureId: int): int =
  ## Get the feature of the visible map.
  if Location(x: 0, y: 0) in visible:
    for featureValue in visible[Location(x: 0, y: 0)]:
      if featureValue.featureId == featureId:
        return featureValue.value
  return -1

proc getLastAction(agent: HeuristicAgent, visible: Table[Location, seq[FeatureValue]]): int =
  ## Get the last action of the visible map.
  agent.getFeature(visible, agent.features.lastAction)

proc getInventory(agent: HeuristicAgent, visible: Table[Location, seq[FeatureValue]], inventoryId: int): int =
  ## Get the inventory of the visible map.
  result = agent.getFeature(visible, inventoryId)
  # Missing inventory is 0.
  if result == -1:
    result = 0

proc getVibe(agent: HeuristicAgent, visible: Table[Location, seq[FeatureValue]]): int =
  ## Get the vibe of the visible map.
  agent.getFeature(visible, agent.features.vibe)

proc updateMap(agent: HeuristicAgent, visible: Table[Location, seq[FeatureValue]]) =
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
  var lastAction = agent.getLastAction(visible)
  if lastAction == agent.actions.moveNorth or lastAction == -1: # Moving north.
    possibleOffsets.add(Location(x: 0, y: -1))
  if lastAction == agent.actions.moveSouth or lastAction == -1: # Moving south.
    possibleOffsets.add(Location(x: 0, y: 1))
  if lastAction == agent.actions.moveWest or lastAction == -1: # Moving west.
    possibleOffsets.add(Location(x: -1, y: 0))
  if lastAction == agent.actions.moveEast or lastAction == -1: # Moving east.
    possibleOffsets.add(Location(x: 1, y: 0))
  possibleOffsets.add(Location(x: 0, y: 0)) # Staying put.

  for offset in possibleOffsets:
    var score = 0
    let location = Location(x: agent.location.x + offset.x, y: agent.location.y + offset.y)
    for x in -5 .. 5:
      for y in -5 .. 5:
        let visibleTag = agent.getTag(visible, Location(x: x, y: y))
        let mapTag = agent.getTag(agent.map, Location(x: x + location.x, y: y + location.y))
        if visibleTag == mapTag:
          if visibleTag == agent.tags.agent or visibleTag == -1:
            # No points for empty or agent locations.
            discard
          elif visibleTag == agent.tags.assembler:
            # There is only one assembler per map, so this is a good score.
            score += 100
          elif visibleTag == agent.tags.wall:
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

proc step(
  agent: HeuristicAgent,
  numAgents: int,
  numTokens: int,
  sizeToken: int,
  rawObservations: pointer,
  numActions: int,
  rawActions: pointer
) {.raises: [].} =
  try:
    echo "Thinking heuristic agent ", agent.agentId
    # echo "  numAgents", numAgents
    # echo "  numTokens", numTokens
    # echo "  sizeToken", sizeToken
    # echo "  numActions", numActions
    let observations = cast[ptr UncheckedArray[uint8]](rawObservations)

    var map: Table[Location, seq[FeatureValue]]
    for token in 0 ..< numTokens:
      let locationPacked = observations[token * sizeToken + agent.agentId * numTokens * sizeToken]
      let featureId = observations[token * sizeToken + agent.agentId * numTokens * sizeToken + 1]
      let value = observations[token * sizeToken + agent.agentId * numTokens * sizeToken + 2]
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
    agent.drawMap(map, initHashSet[Location]())
    updateMap(agent, map)
    echo "updated map:"
    agent.drawMap(agent.map, agent.seen)

    let vibe = agent.getVibe(map)
    echo "vibe: ", vibe

    let invEnergy = agent.getInventory(map, agent.features.invEnergy)
    let invCarbon = agent.getInventory(map, agent.features.invCarbon)
    let invOxygen = agent.getInventory(map, agent.features.invOxygen)
    let invGermanium = agent.getInventory(map, agent.features.invGermanium)
    let invSilicon = agent.getInventory(map, agent.features.invSilicon)
    let invHeart = agent.getInventory(map, agent.features.invHeart)
    let invDecoder = agent.getInventory(map, agent.features.invDecoder)
    let invModulator = agent.getInventory(map, agent.features.invModulator)
    let invResonator = agent.getInventory(map, agent.features.invResonator)
    let invScrambler = agent.getInventory(map, agent.features.invScrambler)

    echo &"H:{invHeart} E:{invEnergy} C:{invCarbon} O2:{invOxygen} Ge:{invGermanium} Si:{invSilicon} D:{invDecoder} M:{invModulator} R:{invResonator} S:{invScrambler}"

    let actions = cast[ptr UncheckedArray[int32]](rawActions)
    let action = agent.random.rand(1 .. 4).int32
    actions[agent.agentId] = action
    echo "taking action ", action

  except:
    echo getCurrentException().getStackTrace()
    echo getCurrentExceptionMsg()
    quit()

exportRefObject HeuristicAgent:
  constructor:
    newHeuristicAgent(int, string)
  fields:
    agentId
  procs:
    reset(HeuristicAgent)
    step

exportProcs:
  initCHook

writeFiles("bindings/generated", "HeuristicAgents")

include bindings/generated/internal
