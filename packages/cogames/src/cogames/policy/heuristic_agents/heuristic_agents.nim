import
  std/[strformat, strutils, tables, random],
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

  HeuristicAgent* = ref object
    agentId: int
    map: Table[Location, seq[FeatureValue]]
    config: Config
    random: Rand

proc ctrlCHandler() {.noconv.} =
  ## Handle ctrl-c signal to exit cleanly.
  echo "\nNim DLL caught ctrl-c, exiting..."
  quit(0)

proc init() =
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

proc drawMap(map: Table[Location, seq[FeatureValue]]) =
  ## Draw the map to the console.
  var
    minX = -5
    maxX = 5
    minY = -5
    maxY = 5
  for location, featureValues in map:
    if location.x < minX:
      minX = location.x
    if location.x > maxX:
      maxX = location.x
    if location.y < minY:
      minY = location.y
    if location.y > maxY:
      maxY = location.y
  echo "Map: ", minX, ", ", maxX, ", ", minY, ", ", maxY
  var line = "+"
  for x in minX .. maxX:
    line.add "--"
  line.add "+"
  echo line
  for y in minY .. maxY:
    line = "|"
    for x in minX .. maxX:
      var cell = "  "
      if Location(x: x, y: y) in map:
        for featureValue in map[Location(x: x, y: y)]:
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
  for x in minX .. maxX:
    line.add "--"
  line.add "+"
  echo line

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
    echo "  numAgents", numAgents
    echo "  numTokens", numTokens
    echo "  sizeToken", sizeToken
    echo "  numActions", numActions
    let observations = cast[ptr UncheckedArray[uint8]](rowObservations)

    agent.map.clear()
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
      if not agent.map.hasKey(location):
        agent.map[location] = @[]
      agent.map[location].add(FeatureValue(featureId: featureId.int, value: value.int))

    drawMap(agent.map)

    let actions = cast[ptr UncheckedArray[int32]](rawActions)

    actions[agent.agentId] = agent.random.rand(1 .. 4).int32


  except:
    quit("should't happen: " & getCurrentExceptionMsg())

exportRefObject HeuristicAgent:
  constructor:
    newHeuristicAgent(int, string)
  fields:
    agentId
  procs:
    reset(HeuristicAgent)
    step

exportProcs:
  init

writeFiles("bindings/generated", "HeuristicAgents")

include bindings/generated/internal
