
import
  std/[algorithm, strformat, strutils, tables, sets, options],
  fidget2/measure,
  jsony

type
  ConfigFeature* = object
    id*: int
    name*: string
    normalization*: float

  AssemblerProtocol* = object
    inputResources*: Table[string, int]
    outputResources*: Table[string, int]

  PolicyConfig* = object
    numAgents*: int
    obsWidth*: int
    obsHeight*: int
    actions*: seq[string]
    tags*: seq[string]
    obsFeatures*: seq[ConfigFeature]
    assemblerProtocols*: seq[AssemblerProtocol]

  Config* = object
    config*: PolicyConfig
    actions*: Actions
    features*: Features
    tags*: Tags
    vibes*: Vibes
    assemblerProtocols*: seq[AssemblerProtocol]
    inventoryTokenBase*: int
    inventoryPowerFeatures*: Table[int, array[2, int]]

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

  Actions*  = object
    noop*: int
    moveNorth*: int
    moveSouth*: int
    moveWest*: int
    moveEast*: int
    vibeDefault*: int
    vibeCharger*: int
    vibeCarbonA*: int
    vibeCarbonB*: int
    vibeOxygenA*: int
    vibeOxygenB*: int
    vibeGermaniumA*: int
    vibeGermaniumB*: int
    vibeSiliconA*: int
    vibeSiliconB*: int
    vibeHeartA*: int
    vibeHeartB*: int
    vibeGear*: int
    vibeAssembler*: int
    vibeChest*: int
    vibeWall*: int

  Tags* = object
    agent*: int
    assembler*: int
    carbonExtractor*: int
    charger*: int
    chest*: int
    germaniumExtractor*: int
    oxygenExtractor*: int
    siliconExtractor*: int
    wall*: int

  Vibes* = object
    # TODO: Pass with vibes from config.
    default*: int = 0
    charger*: int = 1
    carbonA*: int = 2
    carbonB*: int = 3
    oxygenA*: int = 4
    oxygenB*: int = 5
    germaniumA*: int = 6
    germaniumB*: int = 7
    siliconA*: int = 8
    siliconB*: int = 9
    heartA*: int = 10
    heartB*: int = 11
    gear*: int = 12
    assembler*: int = 13
    chest*: int = 14
    wall*: int = 15
    paperclip*: int = 16

  Features* = object
    group*: int
    frozen*: int
    episodeCompletionPct*: int
    goal*: int
    lastAction*: int
    lastReward*: int
    vibe*: int
    compass*: int
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

    protocolInputEnergy*: int
    protocolInputCarbon*: int
    protocolInputOxygen*: int
    protocolInputGermanium*: int
    protocolInputSilicon*: int
    protocolInputHeart*: int
    protocolInputDecoder*: int
    protocolInputModulator*: int
    protocolInputResonator*: int
    protocolInputScrambler*: int

    protocolOutputEnergy*: int
    protocolOutputCarbon*: int
    protocolOutputOxygen*: int
    protocolOutputGermanium*: int
    protocolOutputSilicon*: int
    protocolOutputHeart*: int
    protocolOutputDecoder*: int
    protocolOutputModulator*: int
    protocolOutputResonator*: int
    protocolOutputScrambler*: int

  RecipeInfo* = object
    pattern*: seq[int] # In vibe indices

    energyCost*: int
    carbonCost*: int
    oxygenCost*: int
    germaniumCost*: int
    siliconCost*: int
    heartCost*: int
    decoderCost*: int
    modulatorCost*: int
    resonatorCost*: int
    scramblerCost*: int

    energyOutput*: int
    carbonOutput*: int
    oxygenOutput*: int
    germaniumOutput*: int
    siliconOutput*: int
    heartOutput*: int
    decoderOutput*: int
    modulatorOutput*: int
    resonatorOutput*: int
    scramblerOutput*: int
    cooldown*: int

proc `+`*(location1: Location, location2: Location): Location =
  ## Add two locations.
  result.x = location1.x + location2.x
  result.y = location1.y + location2.y

proc `-`*(location1: Location, location2: Location): Location =
  ## Subtract two locations.
  result.x = location1.x - location2.x
  result.y = location1.y - location2.y

proc manhattan*(a, b: Location): int =
  ## Get the Manhattan distance between two locations.
  abs(a.x - b.x) + abs(a.y - b.y)

proc generateSpiral*(count: int): seq[Location] =
  ## Generate a square spiral starting at (0,0) and spiraling outwards.
  result = @[]
  var
    x = 0
    y = 0
    dx = 1
    dy = 0
    stepSize = 1
    stepsTaken = 0
    directionChanges = 0
  for i in 0 ..< count:
    result.add(Location(x: x, y: y))
    x += dx
    y += dy
    inc stepsTaken
    if stepsTaken == stepSize:
      stepsTaken = 0
      inc(directionChanges)
      # Rotate direction: (dx, dy) -> (-dy, dx)
      let tmp = dx
      dx = -dy
      dy = tmp
      if directionChanges mod 2 == 0:
        inc stepSize
  return result

const spiral* = generateSpiral(1000)

proc `$`*(recipe: RecipeInfo): string =
  ## Stringify the recipe.
  result = "Recipe(pattern: ["
  for vibe in recipe.pattern:
    case vibe:
    of 0:
      result.add("Default")
      result.add(", ")
    of 1:
      result.add("Charger")
      result.add(", ")
    of 2:
      result.add("CarbonA")
      result.add(", ")
    of 3:
      result.add("CarbonB")
      result.add(", ")
    of 4:
      result.add("OxygenA")
      result.add(", ")
    of 5:
      result.add("OxygenB")
      result.add(", ")
    of 6:
      result.add("GermaniumA")
      result.add(", ")
    of 7:
      result.add("GermaniumB")
      result.add(", ")
    of 8:
      result.add("SiliconA")
      result.add(", ")
    of 9:
      result.add("SiliconB")
      result.add(", ")
    of 10:
      result.add("HeartA")
      result.add(", ")
    of 11:
      result.add("HeartB")
      result.add(", ")
    of 12:
      result.add("Gear")
      result.add(", ")
    of 13:
      result.add("Assembler")
      result.add(", ")
    of 14:
      result.add("Chest")
      result.add(", ")
    of 15:
      result.add("Wall")
      result.add(", ")
    of 16:
      result.add("Paperclip")
      result.add(", ")
    else:
      result.add("???")
      result.add(", ")
  result.removeSuffix(", ")
  result.add("]")
  if recipe.energyCost != 0:
    result.add(" E:")
    result.add($recipe.energyCost)
  if recipe.carbonCost != 0:
    result.add(" C:")
    result.add($recipe.carbonCost)
  if recipe.oxygenCost != 0:
    result.add(" O2:")
    result.add($recipe.oxygenCost)
  if recipe.germaniumCost != 0:
    result.add(" Ge:")
    result.add($recipe.germaniumCost)
  if recipe.siliconCost != 0:
    result.add(" Si:")
    result.add($recipe.siliconCost)
  if recipe.heartCost != 0:
    result.add(" Heart:")
    result.add($recipe.heartCost)
  if recipe.decoderCost != 0:
    result.add(" Decoder:")
    result.add($recipe.decoderCost)
  if recipe.modulatorCost != 0:
    result.add(" Modulator:")
    result.add($recipe.modulatorCost)
  if recipe.resonatorCost != 0:
    result.add(" Resonator:")
    result.add($recipe.resonatorCost)
  if recipe.scramblerCost != 0:
    result.add(" Scrambler:")
    result.add($recipe.scramblerCost)
  result.add(" -> ")
  if recipe.energyOutput != 0:
    result.add(" E:")
    result.add($recipe.energyOutput)
  if recipe.carbonOutput != 0:
    result.add(" C:")
    result.add($recipe.carbonOutput)
  if recipe.oxygenOutput != 0:
    result.add(" O2:")
    result.add($recipe.oxygenOutput)
  if recipe.germaniumOutput != 0:
    result.add(" Ge:")
    result.add($recipe.germaniumOutput)
  if recipe.siliconOutput != 0:
    result.add(" Si:")
    result.add($recipe.siliconOutput)
  if recipe.heartOutput != 0:
    result.add(" Heart:")
    result.add($recipe.heartOutput)
  if recipe.decoderOutput != 0:
    result.add(" Decoder:")
    result.add($recipe.decoderOutput)
  if recipe.modulatorOutput != 0:
    result.add(" Modulator:")
    result.add($recipe.modulatorOutput)
  if recipe.resonatorOutput != 0:
    result.add(" Resonator:")
    result.add($recipe.resonatorOutput)
  if recipe.scramblerOutput != 0:
    result.add(" Scrambler:")
    result.add($recipe.scramblerOutput)
  result.add(")")

proc registerProtocolFeature(feature: ConfigFeature; prefix: string;
    dest: var Table[string, int]): bool =
  ## Store protocol input/output features keyed by their resource suffix.
  if not feature.name.startsWith(prefix):
    return false
  if feature.name.len <= prefix.len:
    echo "Protocol feature missing resource suffix: ", feature.name
    return true

  let resource = feature.name[prefix.len .. ^1]
  dest[resource] = feature.id
  return true

proc ctrlCHandler*() {.noconv.} =
  ## Handle ctrl-c signal to exit cleanly.
  echo "\nNim DLL caught ctrl-c, exiting..."
  quit(0)

proc initCHook*() =
  setControlCHook(ctrlCHandler)
  echo "NimAgents initialized"

proc parseConfig*(environmentConfig: string): Config {.raises: [].} =
  try:
    var config = environmentConfig.fromJson(PolicyConfig)
    result = Config(config: config)
    result.assemblerProtocols = config.assemblerProtocols
    result.inventoryPowerFeatures = initTable[int, array[2, int]]()
    var inventoryBaseIds = initTable[string, int]()
    var inventoryPowerIds = initTable[string, array[2, int]]()

    for feature in config.obsFeatures:
      if feature.name.startsWith("inv:"):
        if result.inventoryTokenBase == 0:
          result.inventoryTokenBase = int(feature.normalization)
        let suffix = feature.name[4 .. ^1]
        let powerIndex = suffix.rfind(":p")
        if powerIndex != -1:
          let resource = suffix[0 ..< powerIndex]
          let powerStr = suffix[powerIndex + 2 .. ^1]
          if resource.len > 0 and powerStr.len > 0 and powerStr.allCharsInSet({'0' .. '9'}):
            let power = parseInt(powerStr)
            if power > 0:
              var powers = inventoryPowerIds.getOrDefault(resource, [-1, -1])
              if power <= 2:
                powers[power - 1] = feature.id
              inventoryPowerIds[resource] = powers
              continue
        else:
          inventoryBaseIds[suffix] = feature.id
      case feature.name:
      of "agent:group":
        result.features.group = feature.id
      of "agent:frozen":
        result.features.frozen = feature.id
      of "episode_completion_pct":
        result.features.episodeCompletionPct = feature.id
      of "goal":
        result.features.goal = feature.id
      of "last_action":
        result.features.lastAction = feature.id
      of "last_reward":
        result.features.lastReward = feature.id
      of "vibe":
        result.features.vibe = feature.id
      of "agent:compass":
        result.features.compass = feature.id
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
      of "protocol_input:energy":
        result.features.protocolInputEnergy = feature.id
      of "protocol_input:carbon":
        result.features.protocolInputCarbon = feature.id
      of "protocol_input:oxygen":
        result.features.protocolInputOxygen = feature.id
      of "protocol_input:germanium":
        result.features.protocolInputGermanium = feature.id
      of "protocol_input:silicon":
        result.features.protocolInputSilicon = feature.id
      of "protocol_input:heart":
        result.features.protocolInputHeart = feature.id
      of "protocol_input:decoder":
        result.features.protocolInputDecoder = feature.id
      of "protocol_input:modulator":
        result.features.protocolInputModulator = feature.id
      of "protocol_input:resonator":
        result.features.protocolInputResonator = feature.id
      of "protocol_input:scrambler":
        result.features.protocolInputScrambler = feature.id
      of "protocol_output:energy":
        result.features.protocolOutputEnergy = feature.id
      of "protocol_output:carbon":
        result.features.protocolOutputCarbon = feature.id
      of "protocol_output:oxygen":
        result.features.protocolOutputOxygen = feature.id
      of "protocol_output:germanium":
        result.features.protocolOutputGermanium = feature.id
      of "protocol_output:silicon":
        result.features.protocolOutputSilicon = feature.id
      of "protocol_output:heart":
        result.features.protocolOutputHeart = feature.id
      of "protocol_output:decoder":
        result.features.protocolOutputDecoder = feature.id
      of "protocol_output:modulator":
        result.features.protocolOutputModulator = feature.id
      of "protocol_output:resonator":
        result.features.protocolOutputResonator = feature.id
      of "protocol_output:scrambler":
        result.features.protocolOutputScrambler = feature.id
      else:
        echo "Unknown feature: ", feature.name

    for resource, powers in inventoryPowerIds:
      if resource in inventoryBaseIds:
        result.inventoryPowerFeatures[inventoryBaseIds[resource]] = powers

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
      of "change_vibe_carbon_a":
        result.actions.vibeCarbonA = id
      of "change_vibe_carbon_b":
        result.actions.vibeCarbonB = id
      of "change_vibe_oxygen_a":
        result.actions.vibeOxygenA = id
      of "change_vibe_oxygen_b":
        result.actions.vibeOxygenB = id
      of "change_vibe_germanium_a":
        result.actions.vibeGermaniumA = id
      of "change_vibe_germanium_b":
        result.actions.vibeGermaniumB = id
      of "change_vibe_silicon_a":
        result.actions.vibeSiliconA = id
      of "change_vibe_silicon_b":
        result.actions.vibeSiliconB = id
      of "change_vibe_heart_a":
        result.actions.vibeHeartA = id
      of "change_vibe_heart_b":
        result.actions.vibeHeartB = id
      of "change_vibe_carbon":
        result.actions.vibeCarbonA = id
      of "change_vibe_oxygen":
        result.actions.vibeOxygenA = id
      of "change_vibe_germanium":
        result.actions.vibeGermaniumA = id
      of "change_vibe_silicon":
        result.actions.vibeSiliconA = id
      of "change_vibe_heart":
        result.actions.vibeHeartA = id
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

    for id, name in config.tags:
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
  except JsonError, ValueError:
    echo "Error parsing environment config: ", getCurrentExceptionMsg()


proc computeMapBounds*(map: Table[Location, seq[FeatureValue]]): MapBounds =
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

proc drawMap*(cfg: Config, map: Table[Location, seq[FeatureValue]], seen: HashSet[Location]) =
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
          if featureValue.featureId == cfg.features.group:
            if featureValue.value == 0:
              cell = "@" & ($featureValue.value)[0]
          if featureValue.featureId == cfg.features.tag:
            if featureValue.value == cfg.tags.agent:
              cell = "@@"
            elif featureValue.value == cfg.tags.assembler:
              cell = "As"
            elif featureValue.value == cfg.tags.carbonExtractor:
              cell = "Ca"
            elif featureValue.value == cfg.tags.charger:
              cell = "En"
            elif featureValue.value == cfg.tags.chest:
              cell = "Ch"
            elif featureValue.value == cfg.tags.germaniumExtractor:
              cell = "Ge"
            elif featureValue.value == cfg.tags.oxygenExtractor:
              cell = "O2"
            elif featureValue.value == cfg.tags.siliconExtractor:
              cell = "Si"
            elif featureValue.value == cfg.tags.wall:
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

proc getTag*(cfg: Config, map: Table[Location, seq[FeatureValue]], location: Location): int =
  ## Get the type id of the location in the map.
  if location in map:
    for featureValue in map[location]:
      if featureValue.featureId == cfg.features.tag:
        return featureValue.value
  return -1

proc getFeature*(
  cfg: Config,
  visible: Table[Location,
  seq[FeatureValue]], featureId: int,
  location: Location = Location(x: 0, y: 0)
): int =
  ## Get the feature of the visible map.
  if location in visible:
    for featureValue in visible[location]:
      if featureValue.featureId == featureId:
        return featureValue.value
  return -1

proc getLastAction*(cfg: Config, visible: Table[Location, seq[FeatureValue]]): int =
  ## Get the last action of the visible map.
  cfg.getFeature(visible, cfg.features.lastAction)

proc getInventory*(
  cfg: Config,
  visible: Table[Location, seq[FeatureValue]],
  inventoryId: int,
  location: Location = Location(x: 0, y: 0)
): int =
  ## Get the inventory of the visible map.
  result = cfg.getFeature(visible, inventoryId, location)
  # Missing inventory is 0.
  if result == -1:
    result = 0
  if cfg.inventoryTokenBase > 1 and inventoryId in cfg.inventoryPowerFeatures:
    let powers = cfg.inventoryPowerFeatures[inventoryId]
    if powers[0] > -1:
      let powerValue = cfg.getFeature(visible, powers[0], location)
      if powerValue > -1:
        result += powerValue * cfg.inventoryTokenBase
    if powers[1] > -1:
      let powerValue = cfg.getFeature(visible, powers[1], location)
      if powerValue > -1:
        result += powerValue * cfg.inventoryTokenBase * cfg.inventoryTokenBase

proc getOtherInventory*(
  cfg: Config,
  map: Table[Location, seq[FeatureValue]],
  location: Location,
  inventoryId: int
): int =
  ## Get the other inventory of the visible map.
  cfg.getInventory(map, inventoryId, location)

proc getVibe*(cfg: Config, visible: Table[Location, seq[FeatureValue]], location: Location): int =
  ## Get the vibe of the visible map.
  result = cfg.getFeature(visible, cfg.features.vibe, location)

proc getNearby*(
  cfg: Config,
  currentLocation: Location,
  map: Table[Location, seq[FeatureValue]],
  tagId: int
): Option[Location] =
  ## Get if there is a nearby location with the given tag.
  var
    found = false
    closestLocation = Location(x: 0, y: 0)
    closestDistance = 9999
  for location, featureValues in map:
    for featureValue in featureValues:
      if featureValue.featureId == cfg.features.tag and featureValue.value == tagId:
        let distance = manhattan(location, currentLocation)
        if distance < closestDistance:
          closestDistance = distance
          closestLocation = location
          found = true
  if found:
    return some(closestLocation)
  return none(Location)

proc getNearbyUnseen*(
  cfg: Config,
  currentLocation: Location,
  map: Table[Location, seq[FeatureValue]],
  seen: HashSet[Location],
  unreachables: HashSet[Location]
): Option[Location] =
  ## Get if there is a nearby location that is unseen.
  var
    found = false
    closestLocation = Location(x: 0, y: 0)
    closestDistance = 9999
  for spiralLocation in spiral:
    let location = spiralLocation + currentLocation
    if location notin seen and location notin unreachables:
      let distance = manhattan(location, currentLocation)
      if distance < closestDistance:
        closestDistance = distance
        closestLocation = location
        found = true
  if found:
    return some(closestLocation)
  else:
    return none(Location)

proc simpleGoTo*(cfg: Config, currentLocation: Location, targetLocation: Location): int =
  ## Navigate to the given location.
  echo "currentLocation: ", currentLocation.x, ", ", currentLocation.y
  echo "targetLocation: ", targetLocation.x, ", ", targetLocation.y
  if currentLocation.x < targetLocation.x:
    echo "moving east"
    return cfg.actions.moveEast
  elif currentLocation.x > targetLocation.x:
    echo "moving west"
    return cfg.actions.moveWest
  elif currentLocation.y < targetLocation.y:
    echo "moving south"
    return cfg.actions.moveSouth
  elif currentLocation.y > targetLocation.y:
    echo "moving north"
    return cfg.actions.moveNorth
  else:
    echo "no action"
    return cfg.actions.noop

proc isWalkable*(cfg: Config, map: Table[Location, seq[FeatureValue]], loc: Location): bool =
  # Default: tiles not present are walkable; present tiles are walkable unless you decide otherwise.
  if loc in map:
    for featureValue in map[loc]:
      if featureValue.featureId == cfg.features.tag:
        # Its something that blocks movement.
        return false
      if featureValue.featureId == cfg.features.group:
        # If the group there, then its an agent.
        return false
  return true

proc neighbors(loc: Location): array[4, Location] =
  [
    Location(x: loc.x + 1, y: loc.y), # East
    Location(x: loc.x - 1, y: loc.y), # West
    Location(x: loc.x, y: loc.y - 1), # North (assuming y-1 is north)
    Location(x: loc.x, y: loc.y + 1)  # South
  ]

proc reconstructPath(cameFrom: Table[Location, Location], current: Location): seq[Location] =
  var cur = current
  result = @[cur]
  var cf = cameFrom
  while cf.hasKey(cur):
    cur = cf[cur]
    result.add(cur)
  result.reverse()

proc stepToAction(cfg: Config, fromLoc, toLoc: Location): int =
  # Translate the first step along the path into an action id.
  if toLoc.x == fromLoc.x + 1 and toLoc.y == fromLoc.y:
    return cfg.actions.moveEast
  elif toLoc.x == fromLoc.x - 1 and toLoc.y == fromLoc.y:
    return cfg.actions.moveWest
  elif toLoc.y == fromLoc.y - 1 and toLoc.x == fromLoc.x:
    return cfg.actions.moveNorth
  elif toLoc.y == fromLoc.y + 1 and toLoc.x == fromLoc.x:
    return cfg.actions.moveSouth
  else:
    # Not an adjacent cardinal move; noop as a safeguard.
    return cfg.actions.noop

proc aStar*(
  cfg: Config,
  currentLocation: Location,
  targetLocation: Location,
  map: Table[Location, seq[FeatureValue]]
): Option[int] {.measure.} =
  ## Navigate to the given location using A*. Returns the next action to take.
  if currentLocation == targetLocation:
    return none(int)

  # Open set: nodes to evaluate
  var openSet = initHashSet[Location]()
  openSet.incl(currentLocation)

  # For path reconstruction
  var cameFrom = initTable[Location, Location]()

  # gScore: cost from start
  var gScore = initTable[Location, int]()
  gScore[currentLocation] = 0

  # fScore: g + heuristic
  var fScore = initTable[Location, int]()
  fScore[currentLocation] = manhattan(currentLocation, targetLocation)

  # Utility to get fScore with default "infinite"
  proc getF(loc: Location): int =
    if fScore.hasKey(loc): fScore[loc] else: high(int)

  while openSet.len > 0:

    if openSet.len > 100:
      # Too far... bail out.
      return none(int)

    # Pick node in openSet with lowest fScore
    var currentIter = false
    var current: Location
    var bestF = high(int)
    for n in openSet:
      let f = getF(n)
      if not currentIter or f < bestF:
        bestF = f
        current = n
        currentIter = true

    # (Optional sanity guard)
    if not currentIter:
      # openSet was somehow empty; break out safely
      return none(int)
    if current == targetLocation:
      let path = reconstructPath(cameFrom, current)
      # path[0] is currentLocation; path[1] is our next step (if exists)
      if path.len >= 2:
        return some(stepToAction(cfg, path[0], path[1]))
      else:
        return none(int)

    openSet.excl(current)

    # Explore neighbors
    for nb in neighbors(current).items:
      # Allow stepping onto the goal even if it's "blocked" (e.g., extractor tile).
      if nb != targetLocation and not cfg.isWalkable(map, nb):
        continue

      let tentativeG = (if gScore.hasKey(current): gScore[current] else: high(int)) + 1
      # If nb has no gScore or this path is better, record it
      let nbG = (if gScore.hasKey(nb): gScore[nb] else: high(int))
      if tentativeG < nbG:
        cameFrom[nb] = current
        gScore[nb] = tentativeG
        fScore[nb] = tentativeG + manhattan(nb, targetLocation)
        if nb notin openSet:
          openSet.incl(nb)

  # No path found — fall back to greedy single-step
  return none(int)

proc remove*[T](seq: var seq[T], item: T) =
  let index = seq.find(item)
  if index != -1:
    seq.delete(index)
