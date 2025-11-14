
import
  std/[strformat, strutils, tables, sets, options, algorithm],
  jsony

type
  ConfigFeature* = object
    id*: int
    name*: string
    normalization*: float

  PolicyConfig* = object
    numAgents*: int
    obsWidth*: int
    obsHeight*: int
    actions*: seq[string]
    tags*: seq[string]
    obsFeatures*: seq[ConfigFeature]

  Config* = object
    config*: PolicyConfig
    actions*: Actions
    features*: Features
    tags*: Tags
    vibes*: Vibes

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
    vibeCarbon*: int
    vibeOxygen*: int
    vibeGermanium*: int
    vibeSilicon*: int
    vibeHeart*: int
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
    carbon*: int = 2
    oxygen*: int = 3
    germanium*: int = 4
    silicon*: int = 5
    heart*: int = 6
    gear*: int = 7
    assembler*: int = 8
    chest*: int = 9
    wall*: int = 10
    paperclip*: int = 11

  Features* = object
    group*: int
    frozen*: int
    reservedForFutureUse*: int
    converting*: int
    episodeCompletionPct*: int
    lastAction*: int
    lastActionArg*: int
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
    protocolInputs*: Table[string, int]
    protocolOutputs*: Table[string, int]

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
    result.features.protocolInputs = initTable[string, int]()
    result.features.protocolOutputs = initTable[string, int]()

    for feature in config.obsFeatures:
      case feature.name:
      of "agent:group":
        result.features.group = feature.id
      of "agent:frozen":
        result.features.frozen = feature.id
      of "agent:reserved_for_future_use":
        result.features.reservedForFutureUse = feature.id
      of "converting":
        result.features.converting = feature.id
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
      else:
        if registerProtocolFeature(feature, "protocol_input:",
            result.features.protocolInputs):
          discard
        elif registerProtocolFeature(feature, "protocol_output:",
            result.features.protocolOutputs):
          discard
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

proc getFeature*(cfg: Config, visible: Table[Location, seq[FeatureValue]], featureId: int): int =
  ## Get the feature of the visible map.
  if Location(x: 0, y: 0) in visible:
    for featureValue in visible[Location(x: 0, y: 0)]:
      if featureValue.featureId == featureId:
        return featureValue.value
  return -1

proc getLastAction*(cfg: Config, visible: Table[Location, seq[FeatureValue]]): int =
  ## Get the last action of the visible map.
  cfg.getFeature(visible, cfg.features.lastAction)

proc getInventory*(cfg: Config, visible: Table[Location, seq[FeatureValue]], inventoryId: int): int =
  ## Get the inventory of the visible map.
  result = cfg.getFeature(visible, inventoryId)
  # Missing inventory is 0.
  if result == -1:
    result = 0

proc getOtherInventory*(
  cfg: Config,
  map: Table[Location, seq[FeatureValue]],
  location: Location,
  inventoryId: int
): int =
  ## Get the other inventory of the visible map.
  if location in map:
    for featureValue in map[location]:
      if featureValue.featureId == inventoryId:
        return featureValue.value
  return 0

proc getVibe*(cfg: Config, visible: Table[Location, seq[FeatureValue]]): int =
  ## Get the vibe of the visible map.
  result = cfg.getFeature(visible, cfg.features.vibe)
  if result == -1:
    result = cfg.vibes.default

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
  seen: HashSet[Location]
): Option[Location] =
  ## Get if there is a nearby location that is unseen.
  var
    found = false
    closestLocation = Location(x: 0, y: 0)
    closestDistance = 9999
  for spiralLocation in spiral:
    let location = spiralLocation + currentLocation
    if location notin seen:
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
      if featureValue.featureId == cfg.features.orientation:
        # Its the agent's orientation, so an agent can't move through it.
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
): Option[int] =
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
