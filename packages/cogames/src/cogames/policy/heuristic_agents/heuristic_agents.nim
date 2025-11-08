import
  std/[strformat, strutils, tables, random, sets],
  genny, jsony

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
    typeNames*: seq[string]
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

  HeuristicAgent* = ref object
    agentId: int

    map: Table[Location, seq[FeatureValue]]
    seen: HashSet[Location]
    config: Config
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
    echo "  numAgents", config.numAgents
    echo "  obsWidth", config.obsWidth
    echo "  obsHeight", config.obsHeight
    echo "  actions", config.actions
    if config.typeNames.len == 0:
      config.typeNames = @[
        "wall",
        "assembler",
        "chest",
        "charger",
        "carbon_extractor",
        "oxygen_extractor",
        "germanium_extractor",
        "silicon_extractor",
        "clipped_carbon_extractor",
        "clipped_oxygen_extractor",
        "clipped_germanium_extractor",
        "clipped_silicon_extractor",
      ]
    echo "  typeNames", config.typeNames
    for feature in config.obsFeatures:
      echo "    feature ", feature.id, " ", feature.name, " ", feature.normalization
    result = HeuristicAgent(agentId: agentId, config: config)
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

proc drawMap(map: Table[Location, seq[FeatureValue]], seen: HashSet[Location]) =
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
          if featureValue.featureId == 0:
            case featureValue.value
            of 0:
              cell = "@@"
            of 1:
              cell = "As"
            of 2:
              cell = "Ca"
            of 3:
              cell = "En"
            of 4:
              cell = "Ch"
            of 9:
              cell = "Ge"
            of 10:
              cell = "O2"
            of 11:
              cell = "Si"
            of 12:
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

proc getTypeId(map: Table[Location, seq[FeatureValue]], location: Location): int =
  ## Get the type id of the location in the map.
  if location in map:
    for featureValue in map[location]:
      if featureValue.featureId == 0:
        return featureValue.value
  return -1

proc getLastAction(visible: Table[Location, seq[FeatureValue]]): int =
  ## Get the last action of the visible map.
  let lastActionId = 8 # last_action
  echo "searching for last action in visible map"
  if Location(x: 0, y: 0) in visible:
    for featureValue in visible[Location(x: 0, y: 0)]:
      echo "featureValue: ", featureValue.featureId, " ", featureValue.value
      if featureValue.featureId == lastActionId:
        echo "found last action: ", featureValue.value
        return featureValue.value
  return -1

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
  var lastAction = visible.getLastAction()
  if lastAction == 1 or lastAction == -1: # Moving north.
    possibleOffsets.add(Location(x: 0, y: -1))
  if lastAction == 2 or lastAction == -1: # Moving south.
    possibleOffsets.add(Location(x: 0, y: 1))
  if lastAction == 3 or lastAction == -1: # Moving west.
    possibleOffsets.add(Location(x: -1, y: 0))
  if lastAction == 4 or lastAction == -1: # Moving east.
    possibleOffsets.add(Location(x: 1, y: 0))
  possibleOffsets.add(Location(x: 0, y: 0)) # Staying put.

  for offset in possibleOffsets:
    var score = 0
    let location = Location(x: agent.location.x + offset.x, y: agent.location.y + offset.y)
    for x in -5 .. 5:
      for y in -5 .. 5:
        let visibleTypeId = visible.getTypeId(Location(x: x, y: y))
        let mapTypeId = agent.map.getTypeId(Location(x: x + location.x, y: y + location.y))
        if visibleTypeId == mapTypeId:
          if visibleTypeId == 0 or visibleTypeId == -1:
            # No points for empty or agent locations.
            discard
          elif visibleTypeId == 1:
            # There is only one assembler per map, so this is a good score.
            score += 100
          elif visibleTypeId == 12:
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
  rowObservations: pointer,
  numActions: int,
  rawActions: pointer
) {.raises: [].} =
  try:
    echo "Thinking heuristic agent ", agent.agentId
    # echo "  numAgents", numAgents
    # echo "  numTokens", numTokens
    # echo "  sizeToken", sizeToken
    # echo "  numActions", numActions
    let observations = cast[ptr UncheckedArray[uint8]](rowObservations)

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
    drawMap(map, initHashSet[Location]())
    updateMap(agent, map)
    echo "updated map:"
    drawMap(agent.map, agent.seen)

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
