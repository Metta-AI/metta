
import
  std/[strformat, strutils, tables, sets],
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

  Features* = object
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

proc ctrlCHandler*() {.noconv.} =
  ## Handle ctrl-c signal to exit cleanly.
  echo "\nNim DLL caught ctrl-c, exiting..."
  quit(0)

proc initCHook*() =
  setControlCHook(ctrlCHandler)
  echo "FastAgents initialized"

proc parseConfig*(environmentConfig: string): Config {.raises: [].} =
  try:
    var config = environmentConfig.fromJson(PolicyConfig)
    result = Config(config: config)

    for feature in config.obsFeatures:
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

proc getVibe*(cfg: Config, visible: Table[Location, seq[FeatureValue]]): int =
  ## Get the vibe of the visible map.
  cfg.getFeature(visible, cfg.features.vibe)
