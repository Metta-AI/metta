import
  std/[strformat, tables, random, sets, options, deques],
  fidget2/measure,
  common

# Disable debug output (comment out to enable)
template echo(args: varargs[string, `$`]) = discard

const
  MaxEnergy = 100
  MaxResourceInventory = 100
  MaxToolInventory = 100

  PutCarbonAmount = 10
  PutOxygenAmount = 10
  PutGermaniumAmount = 1
  PutSiliconAmount = 25

type
  RaceCarAgent* = ref object
    agentId*: int

    map: Table[Location, seq[FeatureValue]]
    seen: HashSet[Location]
    cfg: Config
    random: Rand
    location: Location
    lastActions: seq[int]
    recipes: seq[RecipeInfo]

    clippedExtractors: HashSet[Location]

    stepCount: int
    prevCompletion: int

    carbonTarget: int
    oxygenTarget: int
    germaniumTarget: int
    siliconTarget: int

    bump: bool
    offsets4: seq[Location]  # 4 cardinal but random for each agent
    seenAssembler: bool
    seenChest: bool
    exploreLocations: seq[Location]

    dwellAssemblerTicks: int  # stay on assembler to let crafting finish
    dwellChestTicks: int      # stay on chest to let heart deposit register
    hasHeartMission: bool     # hard lock to deliver heart without other tasks
    heartsDelivered: int      # heuristic count of hearts turned in
    prevHeartInv: int         # last seen heart inventory

    role: string              # simple role tag: "runner", "harvest_cosi", "harvest_og", "support"

  RaceCarPolicy* = ref object
    agents*: seq[RaceCarAgent]

const Offsets4 = [
  Location(x: -1, y: +0),
  Location(x: +0, y: +1),
  Location(x: +0, y: -1),
  Location(x: +1, y: +0),
]

const Offsets8 = [
  Location(x: +0, y: +0),
  Location(x: +0, y: +1),
  Location(x: +0, y: -1),
  Location(x: +1, y: +0),
  Location(x: +1, y: +1),
  Location(x: +1, y: -1),
  Location(x: -1, y: +0),
  Location(x: -1, y: +1),
  Location(x: -1, y: -1),
]

proc getActiveRecipe(agent: RaceCarAgent): RecipeInfo {.measure.} =
  ## Get the recipes form the assembler protocol inputs.
  let assemblerLocation = agent.cfg.getNearby(agent.location, agent.map, agent.cfg.tags.assembler)
  if assemblerLocation.isSome():
    let location = assemblerLocation.get()
    # Get the vibe key.
    result.pattern = @[]
    for offsets in Offsets8:
      let vibeKey = agent.cfg.getVibe(agent.map, location + offsets)
      if vibeKey != -1:
        result.pattern.add(vibeKey)

    # Get the required resources.
    let assemblerFeatures = agent.map[location]
    for feature in assemblerFeatures:
      if feature.featureId == agent.cfg.features.protocolInputEnergy:
        result.energyCost = feature.value
      elif feature.featureId == agent.cfg.features.protocolInputCarbon:
        result.carbonCost = feature.value
      elif feature.featureId == agent.cfg.features.protocolInputOxygen:
        result.oxygenCost = feature.value
      elif feature.featureId == agent.cfg.features.protocolInputGermanium:
        result.germaniumCost = feature.value
      elif feature.featureId == agent.cfg.features.protocolInputSilicon:
        result.siliconCost = feature.value
      elif feature.featureId == agent.cfg.features.protocolInputHeart:
        result.heartCost = feature.value
      elif feature.featureId == agent.cfg.features.protocolInputDecoder:
        result.decoderCost = feature.value
      elif feature.featureId == agent.cfg.features.protocolInputModulator:
        result.modulatorCost = feature.value
      elif feature.featureId == agent.cfg.features.protocolInputResonator:
        result.resonatorCost = feature.value
      elif feature.featureId == agent.cfg.features.protocolInputScrambler:
        result.scramblerCost = feature.value

      elif feature.featureId == agent.cfg.features.protocolOutputEnergy:
        result.energyCost = feature.value
      elif feature.featureId == agent.cfg.features.protocolOutputCarbon:
        result.carbonCost = feature.value
      elif feature.featureId == agent.cfg.features.protocolOutputOxygen:
        result.oxygenCost = feature.value
      elif feature.featureId == agent.cfg.features.protocolOutputGermanium:
        result.germaniumCost = feature.value
      elif feature.featureId == agent.cfg.features.protocolOutputSilicon:
        result.siliconCost = feature.value
      elif feature.featureId == agent.cfg.features.protocolOutputHeart:
        result.heartCost = feature.value
      elif feature.featureId == agent.cfg.features.protocolOutputDecoder:
        result.decoderCost = feature.value
      elif feature.featureId == agent.cfg.features.protocolOutputModulator:
        result.modulatorCost = feature.value
      elif feature.featureId == agent.cfg.features.protocolOutputResonator:
        result.resonatorCost = feature.value
      elif feature.featureId == agent.cfg.features.protocolOutputScrambler:
        result.scramblerCost = feature.value

proc resetAgent(agent: RaceCarAgent) =
  agent.map = initTable[Location, seq[FeatureValue]]()
  agent.seen = initHashSet[Location]()
  agent.clippedExtractors = initHashSet[Location]()
  agent.location = Location(x: 0, y: 0)
  agent.lastActions = @[]
  agent.stepCount = 0
  agent.prevCompletion = 0
  agent.carbonTarget = PutCarbonAmount
  agent.oxygenTarget = PutOxygenAmount
  agent.germaniumTarget = PutGermaniumAmount
  agent.siliconTarget = PutSiliconAmount
  agent.bump = false
  agent.seenAssembler = false
  agent.seenChest = false
  agent.exploreLocations = @[]
  agent.dwellAssemblerTicks = 0
  agent.dwellChestTicks = 0
  agent.hasHeartMission = false
  agent.heartsDelivered = 0
  agent.prevHeartInv = 0

proc newRaceCarAgent*(agentId: int, environmentConfig: string): RaceCarAgent =
  ## Create a new thinky agent, the fastest and the smartest agent.

  var config = parseConfig(environmentConfig)
  result = RaceCarAgent(agentId: agentId, cfg: config)
  result.random = initRand(agentId)
  resetAgent(result)
  # Randomize the offsets4 for each agent, so they take different directions.
  var offsets4 = Offsets4
  result.random.shuffle(offsets4)
  for i in 0 ..< offsets4.len:
    offsets4[i] = offsets4[i]

  result.exploreLocations = @[
    Location(x: -7, y: 0),
    Location(x: 0, y: +7),
    Location(x: +7, y: 0),
    Location(x: 0, y: -7),
  ]
  result.random.shuffle(result.exploreLocations)

  # Assign simple roles:
  # 0: runner/crafter, 1: carbon+silicon, 2: oxygen+germanium, 3: support/unclipping
  # Default role for now: no specialization (reverts earlier experiment)
  result.role = "support"

proc updateMap(agent: RaceCarAgent, visible: Table[Location, seq[FeatureValue]]) {.measure.} =
  ## Update the big map with the small visible map.

  if agent.map.len == 0:
    # First time we're called, just copy the visible map to the big map.
    agent.map = visible
    agent.location = Location(x: 0, y: 0)
    for loc, features in visible:
      for fv in features:
        if fv.featureId == agent.cfg.features.clipped and fv.value > 0:
          agent.clippedExtractors.incl(loc)
    for x in -5 .. 5:
      for y in -5 .. 5:
        let visibleLocation = Location(x: x, y: y)
        agent.seen.incl(visibleLocation)
    return

  var newLocation = agent.location
  let lastAction = agent.cfg.getLastAction(visible)
  if lastAction == agent.cfg.actions.moveNorth:
    newLocation.y -= 1
  elif lastAction == agent.cfg.actions.moveSouth:
    newLocation.y += 1
  elif lastAction == agent.cfg.actions.moveWest:
    newLocation.x -= 1
  elif lastAction == agent.cfg.actions.moveEast:
    newLocation.x += 1

  # Does the new location have force any tags to no be there?
  # If so we did a bump instead of a move.
  block bumpCheck:
    agent.bump = false
    for x in -5 .. 5:
      for y in -5 .. 5:
        let visibleLocation = Location(x: x, y: y)
        let mapLocation = Location(x: x + newLocation.x, y: y + newLocation.y)
        if mapLocation notin agent.seen:
          continue
        var visibleTag = agent.cfg.getTag(visible, visibleLocation)
        if visibleTag == agent.cfg.tags.agent:
          # Ignore agents.
          visibleTag = -1
        var mapTag = agent.cfg.getTag(agent.map, mapLocation)
        if mapTag == agent.cfg.tags.agent:
          # Ignore agents.
          mapTag = -1
        if visibleTag != mapTag:
          newLocation = agent.location
          agent.bump = true
          break bumpCheck

  # Update the seen set.
  agent.location = newLocation
  for x in -5 .. 5:
    for y in -5 .. 5:
      let visibleLocation = Location(x: x, y: y)
      let mapLocation = Location(x: x + agent.location.x, y: y + agent.location.y)
      if visibleLocation in visible:
        agent.map[mapLocation] = visible[visibleLocation]
        var isClipped = false
        for fv in visible[visibleLocation]:
          if fv.featureId == agent.cfg.features.clipped and fv.value > 0:
            isClipped = true
            break
        if isClipped:
          agent.clippedExtractors.incl(mapLocation)
        else:
          agent.clippedExtractors.excl(mapLocation)
      else:
        agent.map[mapLocation] = @[]
      agent.seen.incl(mapLocation)

  # agent.cfg.drawMap(agent.map, agent.seen)

proc getNumAgentsNearby*(
  cfg: Config,
  location: Location,
  map: Table[Location, seq[FeatureValue]]
): int {.measure.} =
  ## Get the number of agents nearby.
  for offset in Offsets8:
    let at = location + offset
    if at in map:
      for featureValue in map[at]:
        if featureValue.featureId == cfg.features.group:
          result += 1
          break

proc getNearbyExtractor*(
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
        let agentsNearby = cfg.getNumAgentsNearby(location, map)
        if agentsNearby > 1:
          continue
        var skip = false
        for f in map[location]:
          if f.featureId == cfg.features.remainingUses and f.value == 0:
            skip = true
            break
          if f.featureId == cfg.features.clipped and f.value > 0:
            skip = true
            break
          # if f.featureId == cfg.features.cooldownRemaining and f.value > 50:
          #   skip = true
          #   break
        if skip:
          continue
        let distance = manhattan(location, currentLocation)
        if distance < closestDistance:
          closestDistance = distance
          closestLocation = location
          found = true
  if found:
    return some(closestLocation)
  return none(Location)

proc getResourceForTag(cfg: Config, tagId: int): string =
  if tagId == cfg.tags.carbonExtractor:
    return "carbon"
  elif tagId == cfg.tags.oxygenExtractor:
    return "oxygen"
  elif tagId == cfg.tags.germaniumExtractor:
    return "germanium"
  elif tagId == cfg.tags.siliconExtractor:
    return "silicon"
  else:
    return ""

proc isClipped(agent: RaceCarAgent, location: Location): bool =
  if location notin agent.map:
    return false
  for f in agent.map[location]:
    if f.featureId == agent.cfg.features.clipped and f.value > 0:
      return true
  false

proc findNearestClipped(agent: RaceCarAgent): Option[Location] =
  var found = false
  var best = Location(x: 0, y: 0)
  var bestDist = high(int)
  for loc in agent.clippedExtractors:
    let dist = manhattan(agent.location, loc)
    if dist < bestDist:
      bestDist = dist
      best = loc
      found = true
  if found:
    return some(best)
  return none(Location)

proc getClipRequirement(agent: RaceCarAgent, location: Location): (string, int) =
  if location notin agent.map:
    return ("", 0)
  for f in agent.map[location]:
    if f.featureId == agent.cfg.features.protocolInputDecoder and f.value > 0:
      return ("decoder", f.value)
    if f.featureId == agent.cfg.features.protocolInputModulator and f.value > 0:
      return ("modulator", f.value)
    if f.featureId == agent.cfg.features.protocolInputResonator and f.value > 0:
      return ("resonator", f.value)
    if f.featureId == agent.cfg.features.protocolInputScrambler and f.value > 0:
      return ("scrambler", f.value)
  return ("", 0)

proc getExtractorResource(agent: RaceCarAgent, location: Location): string =
  if location notin agent.map:
    return ""
  for f in agent.map[location]:
    if f.featureId == agent.cfg.features.tag:
      return agent.cfg.getResourceForTag(f.value)
  return ""

proc getToolInventory(agent: RaceCarAgent, tool: string): int =
  case tool
  of "decoder":
    agent.cfg.getInventory(agent.map, agent.cfg.features.invDecoder, agent.location)
  of "modulator":
    agent.cfg.getInventory(agent.map, agent.cfg.features.invModulator, agent.location)
  of "resonator":
    agent.cfg.getInventory(agent.map, agent.cfg.features.invResonator, agent.location)
  of "scrambler":
    agent.cfg.getInventory(agent.map, agent.cfg.features.invScrambler, agent.location)
  else:
    0

proc getToolCraftInputs(agent: RaceCarAgent, tool: string): Table[string, int] =
  result = initTable[string, int]()
  if tool.len == 0:
    return

  # Prefer explicit assembler protocol that outputs the tool
  for proto in agent.cfg.assemblerProtocols:
    if tool in proto.outputResources:
      for k, v in proto.inputResources:
        result[k] = v
      return

  # Fallback: standard mapping (one resource crafts the tool)
  case tool
  of "decoder":
    result["carbon"] = 1
  of "modulator":
    result["oxygen"] = 1
  of "resonator":
    result["silicon"] = 1
  of "scrambler":
    result["germanium"] = 1
  else:
    discard

proc step*(
  agent: RaceCarAgent,
  numAgents: int,
  numTokens: int,
  sizeToken: int,
  rawObservation: pointer,
  numActions: int,
  agentAction: ptr int32
) {.measure.} =
  try:

    let observations = cast[ptr UncheckedArray[uint8]](rawObservation)

    # Parse the tokens into a vision map.
    var visible: Table[Location, seq[FeatureValue]]
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
      if location notin visible:
        visible[location] = @[]
      visible[location].add(FeatureValue(featureId: featureId.int, value: value.int))

    # Detect episode resets via episodeCompletionPct wrapping to 0.
    let completionPct = agent.cfg.getFeature(visible, agent.cfg.features.episodeCompletionPct)
    if agent.stepCount > 0 and (completionPct == 0 or completionPct < agent.prevCompletion):
      resetAgent(agent)
    if completionPct >= 0:
      agent.prevCompletion = completionPct
    let episodeEndingSoon = completionPct >= 90

    proc doAction(action: int) {.measure.} =

      # Get last action from observations, in case something else moved us.
      agent.lastActions.add(agent.cfg.getLastAction(visible))

      # Stuck prevention: if last 2 actions are left, right and this is left.
      if agent.lastActions.len >= 2 and
        action == agent.cfg.actions.moveWest and
        agent.lastActions[^1] == agent.cfg.actions.moveEast and
        agent.lastActions[^2] == agent.cfg.actions.moveWest:
        # Noop 50% of the time.
        if agent.random.rand(1 .. 2) == 1:
          echo "Stuck prevention: left, right, left"
          doAction(agent.cfg.actions.noop.int32)
          return
      if agent.lastActions.len >= 2 and
        action == agent.cfg.actions.moveEast and
        agent.lastActions[^1] == agent.cfg.actions.moveWest and
        agent.lastActions[^2] == agent.cfg.actions.moveEast:
        # Noop 50% of the time.
        if agent.random.rand(1 .. 2) == 1:
          echo "Stuck prevention: right, left, right"
          doAction(agent.cfg.actions.noop.int32)
          return
      if agent.lastActions.len >= 2 and
        action == agent.cfg.actions.moveNorth and
        agent.lastActions[^1] == agent.cfg.actions.moveSouth and
        agent.lastActions[^2] == agent.cfg.actions.moveNorth:
        # Noop 50% of the time.
        if agent.random.rand(1 .. 2) == 1:
          echo "Stuck prevention: north, south, north"
          doAction(agent.cfg.actions.noop.int32)
          return
      if agent.lastActions.len >= 2 and
        action == agent.cfg.actions.moveSouth and
        agent.lastActions[^1] == agent.cfg.actions.moveNorth and
        agent.lastActions[^2] == agent.cfg.actions.moveSouth:
        # Noop 50% of the time.
        if agent.random.rand(1 .. 2) == 1:
          echo "Stuck prevention: south, north, south"
          doAction(agent.cfg.actions.noop.int32)
          return

      agentAction[] = action.int32

    updateMap(agent, visible)

    let
      vibe = agent.cfg.getVibe(visible, Location(x: 0, y: 0))
      invEnergy = agent.cfg.getInventory(visible, agent.cfg.features.invEnergy)
      invCarbon = agent.cfg.getInventory(visible, agent.cfg.features.invCarbon)
      invOxygen = agent.cfg.getInventory(visible, agent.cfg.features.invOxygen)
      invGermanium = agent.cfg.getInventory(visible, agent.cfg.features.invGermanium)
      invSilicon = agent.cfg.getInventory(visible, agent.cfg.features.invSilicon)
      invHeart = agent.cfg.getInventory(visible, agent.cfg.features.invHeart)
      invDecoder = agent.cfg.getInventory(visible, agent.cfg.features.invDecoder)
      invModulator = agent.cfg.getInventory(visible, agent.cfg.features.invModulator)
      invResonator = agent.cfg.getInventory(visible, agent.cfg.features.invResonator)
      invScrambler = agent.cfg.getInventory(visible, agent.cfg.features.invScrambler)

    # Track heart inventory delta to estimate deliveries.
    if invHeart == 0 and agent.prevHeartInv > 0:
      inc agent.heartsDelivered, agent.prevHeartInv
    agent.prevHeartInv = invHeart

    proc vibeUnavailable(v: int, action: int): bool =
      ## True when the environment exposes neither the vibe value nor an action to change it.
      v == -1 or action == 0

    echo &"vibe:{vibe} H:{invHeart} E:{invEnergy} C:{invCarbon} O2:{invOxygen} Ge:{invGermanium} Si:{invSilicon} D:{invDecoder} M:{invModulator} R:{invResonator} S:{invScrambler}"

    let activeRecipe = agent.getActiveRecipe()
    echo "active recipe: " & $activeRecipe

    # Adjust targets after each delivered heart (later hearts are cheaper on machina_1).
    if agent.heartsDelivered > 0:
      agent.carbonTarget = min(agent.carbonTarget, 6)
      agent.oxygenTarget = min(agent.oxygenTarget, 6)
      agent.germaniumTarget = min(agent.germaniumTarget, 1)
      agent.siliconTarget = min(agent.siliconTarget, 15)
    else:
      # Make the first heart more reliable: modestly raise O2/Ge targets, allow a bit more silicon buffer.
      agent.oxygenTarget = max(agent.oxygenTarget, PutOxygenAmount + 2)   # was 10, now 12
      agent.germaniumTarget = max(agent.germaniumTarget, PutGermaniumAmount + 1) # was 1, now 2
      agent.siliconTarget = max(agent.siliconTarget, PutSiliconAmount + 8) # allow up to 33 early

    # If we're waiting on assembler crafting to finish, stay put until heart appears or timer expires.
    if agent.dwellAssemblerTicks > 0 and invHeart == 0:
      dec agent.dwellAssemblerTicks
      doAction(agent.cfg.actions.noop.int32)
      echo "dwelling at assembler (" & $agent.dwellAssemblerTicks & " left)"
      return

    # If we're on chest with a heart, dwell a few ticks so deposit registers.
    if agent.dwellChestTicks > 0:
      if invHeart == 0:
        agent.dwellChestTicks = 0
        agent.hasHeartMission = false
      else:
        dec agent.dwellChestTicks
        doAction(agent.cfg.actions.noop.int32)
        echo "dwelling at chest (" & $agent.dwellChestTicks & " left)"
        return

    # Heart mission: absolute priority until deposited.
    if invHeart > 0:
      agent.hasHeartMission = true
    if agent.hasHeartMission:
      # Optional safety: top off if critically low while carrying.
      if invEnergy < MaxEnergy div 3:
        let charger = agent.cfg.getNearbyExtractor(agent.location, agent.map, agent.cfg.tags.charger)
        if charger.isSome():
          let action = agent.cfg.aStar(agent.location, charger.get(), agent.map)
          if action.isSome():
            doAction(action.get().int32)
            echo "heart mission: detouring to charger"
            return
      elif invEnergy < (MaxEnergy * 7) div 10:
        let charger = agent.cfg.getNearbyExtractor(agent.location, agent.map, agent.cfg.tags.charger)
        if charger.isSome():
          let action = agent.cfg.aStar(agent.location, charger.get(), agent.map)
          if action.isSome():
            doAction(action.get().int32)
            echo "heart mission: topping to 70%"
            return

      # Ensure we're in deposit vibe.
      let depositAction = agent.cfg.actions.vibeHeartB
      let depositVibe = agent.cfg.vibes.heartB
      if not vibeUnavailable(depositVibe, depositAction) and vibe != depositVibe:
        doAction(depositAction.int32)
        echo "heart mission: switching to heart deposit vibe"
        return

      let chestNearby = agent.cfg.getNearby(agent.location, agent.map, agent.cfg.tags.chest)
      if chestNearby.isSome():
        if agent.location == chestNearby.get():
          agent.dwellChestTicks = max(agent.dwellChestTicks, 8)
          doAction(agent.cfg.actions.noop.int32)
          echo "heart mission: depositing at chest"
          return
        let action = agent.cfg.aStar(agent.location, chestNearby.get(), agent.map)
        if action.isSome():
          doAction(action.get().int32)
          echo "heart mission: heading to chest"
          return
      # Chest unknown: bias toward assembler as a hub, else cautious explore.
      let assemblerNearby = agent.cfg.getNearby(agent.location, agent.map, agent.cfg.tags.assembler)
      if assemblerNearby.isSome():
        let action = agent.cfg.aStar(agent.location, assemblerNearby.get(), agent.map)
        if action.isSome():
          doAction(action.get().int32)
          echo "heart mission: seeking chest via assembler"
          return
      # fallback small random step to continue revealing map
      let action = agent.random.rand(1 .. 4).int32
      echo "heart mission: random explore while holding heart"
      doAction(action.int32)
      return

    if activeRecipe.pattern.len > 0:
      # Split the cost evenly across the agents.
      proc divUp(a, b: int): int =
        ## Like div, but rounds up instead of down.
        let extra = if a mod b > 0: 1 else: 0
        return a div b + extra
      agent.carbonTarget = max(agent.carbonTarget, activeRecipe.carbonCost.divUp(activeRecipe.pattern.len))
      agent.oxygenTarget = max(agent.oxygenTarget, activeRecipe.oxygenCost.divUp(activeRecipe.pattern.len))
      agent.germaniumTarget = max(agent.germaniumTarget, activeRecipe.germaniumCost.divUp(activeRecipe.pattern.len))
      agent.siliconTarget = max(agent.siliconTarget, activeRecipe.siliconCost.divUp(activeRecipe.pattern.len))
    else:
      agent.carbonTarget = max(agent.carbonTarget, activeRecipe.carbonCost)
      agent.oxygenTarget = max(agent.oxygenTarget, activeRecipe.oxygenCost)
      agent.germaniumTarget = max(agent.germaniumTarget, activeRecipe.germaniumCost)
      agent.siliconTarget = max(agent.siliconTarget, activeRecipe.siliconCost)

    proc findAndTakeResource(
      agent: RaceCarAgent,
      vibe: int,
      resource: int,
      target: int,
      inventory: int,
      vibeGetResource: int,
      vibeAction: int,
      extractorTag: int,
      name: string
    ): bool {.measure.} =
      # Check the chest first.
      var closeChest = agent.cfg.getNearby(agent.location, agent.map, agent.cfg.tags.chest)
      if closeChest.isSome():
        let chestInventory = agent.cfg.getInventory(agent.map, resource, closeChest.get())
        if chestInventory > 0:
          if not vibeUnavailable(vibeGetResource, vibeAction) and vibe != vibeGetResource:
            doAction(vibeAction.int32)
            echo "vibing " & name & " to take from chest"
            return true
          measurePush("chest nearby to take " & name)
          let action = agent.cfg.aStar(agent.location, closeChest.get(), agent.map)
          measurePop()
          if action.isSome():
            doAction(action.get().int32)
            echo "going to chest to take " & name & " from chest"
            return true

      # Then try the extractor.
      let extractorNearby = agent.cfg.getNearbyExtractor(agent.location, agent.map, extractorTag)
      if extractorNearby.isSome():
        measurePush("extractor nearby to take " & name)
        let action = agent.cfg.aStar(agent.location, extractorNearby.get(), agent.map)
        measurePop()
        if action.isSome():
          doAction(action.get().int32)
          echo "going to " & name & ", need: " & $target & " have: " & $inventory
          return true
      false

    # Unclipping: if we see a clipped extractor, grab the required tool and go unclip it.
    block unclipping:
      let clipTarget = agent.findNearestClipped()
      if clipTarget.isSome():
        let (toolName, needed) = agent.getClipRequirement(clipTarget.get())
        let resourceBlocked = agent.getExtractorResource(clipTarget.get())
        # Fallback mapping if protocol inputs are missing.
        var tool = toolName
        if tool.len == 0:
          case resourceBlocked
          of "oxygen":
            tool = "decoder"
          of "carbon":
            tool = "modulator"
          of "germanium":
            tool = "resonator"
          of "silicon":
            tool = "scrambler"
        else:
          discard
        var invTool = agent.getToolInventory(tool)

        let neededAmount = if needed > 0: needed else: 1
        # Skip crafting tools; only unclip if we already have the needed one.

        # If we have the tool now, move to the clipped extractor and use it (stay in gear vibe).
        invTool = agent.getToolInventory(tool)
        if tool.len > 0 and invTool >= neededAmount:
          if not vibeUnavailable(agent.cfg.vibes.gear, agent.cfg.actions.vibeGear) and vibe != agent.cfg.vibes.gear:
            doAction(agent.cfg.actions.vibeGear.int32)
            return
          if agent.location == clipTarget.get():
            # Already on target; wait to let unclipping resolve.
            doAction(agent.cfg.actions.noop.int32)
            return
          let action = agent.cfg.aStar(agent.location, clipTarget.get(), agent.map)
          if action.isSome():
            doAction(action.get().int32)
            return

    # Are we running low on energy?
    if invEnergy < MaxEnergy div 4:
      let chargerNearby = agent.cfg.getNearbyExtractor(agent.location, agent.map, agent.cfg.tags.charger)
      if chargerNearby.isSome():
        # If we are already on the charger, stay put to actually recharge.
        if agent.location == chargerNearby.get():
          doAction(agent.cfg.actions.noop.int32)
          echo "charging in place"
          return
        measurePush("charger nearby")
        let action = agent.cfg.aStar(agent.location, chargerNearby.get(), agent.map)
        measurePop()
        if action.isSome():
          doAction(action.get().int32)
          echo "going to charger"
          return

    # Charge opportunistically when close to a charger.
    if invEnergy < MaxEnergy - 20:
      let chargerNearby = agent.cfg.getNearbyExtractor(agent.location, agent.map, agent.cfg.tags.charger)
      if chargerNearby.isSome():
        if manhattan(agent.location, chargerNearby.get()) < 2:
          # If already on the charger tile, wait; otherwise walk over.
          if agent.location == chargerNearby.get():
            doAction(agent.cfg.actions.noop.int32)
            echo "topping off energy in place"
            return
          measurePush("charge nearby")
          let action = agent.cfg.aStar(agent.location, chargerNearby.get(), agent.map)
          measurePop()
          if action.isSome():
            doAction(action.get().int32)
            echo "charge nearby might as well charge"
            return


    if invCarbon >= agent.carbonTarget and invOxygen >= agent.oxygenTarget and invGermanium >= agent.germaniumTarget and invSilicon >= agent.siliconTarget:
      # We have all the resources we need, so we can build a heart.
      echo "trying to build a heart"

      if not vibeUnavailable(agent.cfg.vibes.heartA, agent.cfg.actions.vibeHeartA) and vibe != agent.cfg.vibes.heartA:
        doAction(agent.cfg.actions.vibeHeartA.int32)
        echo "vibing heart for assembler"
        return

      let assemblerNearby = agent.cfg.getNearby(agent.location, agent.map, agent.cfg.tags.assembler)
      if assemblerNearby.isSome():
        # If already on assembler with proper vibe and resources, dwell to let craft resolve.
        if agent.location == assemblerNearby.get():
          agent.dwellAssemblerTicks = max(agent.dwellAssemblerTicks, 4)
          doAction(agent.cfg.actions.noop.int32)
          echo "crafting heart at assembler"
          return
        measurePush("assembler nearby")
        let action = agent.cfg.aStar(agent.location, assemblerNearby.get(), agent.map)
        measurePop()
        if action.isSome():
          doAction(action.get().int32)
          echo "going to assembler to build heart"
          return

    # Simplified: don’t dump; keep focus on heart inputs.

    # Harvesting prioritised for heart inputs (favor bottlenecks first).
    let haveAll =
      invCarbon >= agent.carbonTarget and
      invOxygen >= agent.oxygenTarget and
      invGermanium >= agent.germaniumTarget and
      invSilicon >= agent.siliconTarget

    if not haveAll:
      # Order: germanium, oxygen, carbon, silicon (silicon last since often plentiful).
      if agent.germaniumTarget > 0 and invGermanium < agent.germaniumTarget:
        if agent.findAndTakeResource(
          vibe,
          agent.cfg.features.invGermanium,
          agent.germaniumTarget,
          invGermanium,
          agent.cfg.vibes.germaniumA,
          agent.cfg.actions.vibeGermaniumA,
          agent.cfg.tags.germaniumExtractor,
          "germanium"
        ):
          return

      if agent.oxygenTarget > 0 and invOxygen < agent.oxygenTarget:
        if agent.findAndTakeResource(
          vibe,
          agent.cfg.features.invOxygen,
          agent.oxygenTarget,
          invOxygen,
          agent.cfg.vibes.oxygenA,
          agent.cfg.actions.vibeOxygenA,
          agent.cfg.tags.oxygenExtractor,
          "oxygen"
        ):
          return

      if agent.carbonTarget > 0 and invCarbon < agent.carbonTarget:
        if agent.findAndTakeResource(
          vibe,
          agent.cfg.features.invCarbon,
          agent.carbonTarget,
          invCarbon,
          agent.cfg.vibes.carbonA,
          agent.cfg.actions.vibeCarbonA,
          agent.cfg.tags.carbonExtractor,
          "carbon"
        ):
          return

      let siliconBuffer = if agent.heartsDelivered > 0: 2 else: 8
      if agent.siliconTarget > 0 and invSilicon < agent.siliconTarget and invSilicon < agent.siliconTarget + siliconBuffer:
        if agent.findAndTakeResource(
          vibe,
          agent.cfg.features.invSilicon,
          agent.siliconTarget,
          invSilicon,
          agent.cfg.vibes.siliconA,
          agent.cfg.actions.vibeSiliconA,
          agent.cfg.tags.siliconExtractor,
          "silicon"
        ):
          return
    if agent.carbonTarget > 0 and invCarbon < agent.carbonTarget:
      if agent.findAndTakeResource(
        vibe,
        agent.cfg.features.invCarbon,
        agent.carbonTarget,
        invCarbon,
        agent.cfg.vibes.carbonA,
        agent.cfg.actions.vibeCarbonA,
        agent.cfg.tags.carbonExtractor,
        "carbon"
      ):
        return

    if agent.siliconTarget > 0 and invSilicon < agent.siliconTarget:
      if agent.findAndTakeResource(
        vibe,
        agent.cfg.features.invSilicon,
        agent.siliconTarget,
        invSilicon,
        agent.cfg.vibes.siliconA,
        agent.cfg.actions.vibeSiliconA,
        agent.cfg.tags.siliconExtractor,
        "silicon"
      ):
        return

    if agent.oxygenTarget > 0 and invOxygen < agent.oxygenTarget:
      if agent.findAndTakeResource(
        vibe,
        agent.cfg.features.invOxygen,
        agent.oxygenTarget,
        invOxygen,
        agent.cfg.vibes.oxygenA,
        agent.cfg.actions.vibeOxygenA,
        agent.cfg.tags.oxygenExtractor,
        "oxygen"
      ):
        return

    if agent.germaniumTarget > 0 and invGermanium < agent.germaniumTarget:
      if agent.findAndTakeResource(
        vibe,
        agent.cfg.features.invGermanium,
        agent.germaniumTarget,
        invGermanium,
        agent.cfg.vibes.germaniumA,
        agent.cfg.actions.vibeGermaniumA,
        agent.cfg.tags.germaniumExtractor,
        "germanium"
      ):
        return

    # Explore locations around the assembler.
    block:
      if not agent.seenAssembler:
        let assemblerNearby = agent.cfg.getNearby(agent.location, agent.map, agent.cfg.tags.assembler)
        if assemblerNearby.isSome():
          agent.seenAssembler = true
          let keyLocations = [
            Location(x: -10, y: -10),
            Location(x: -10, y: +10),
            Location(x: +10, y: -10),
            Location(x: +10, y: +10),
          ]
          for keyLocation in keyLocations:
            let location = assemblerNearby.get() + keyLocation
            agent.exploreLocations.add(location)
      if not agent.seenChest:
        let chestNearby = agent.cfg.getNearby(agent.location, agent.map, agent.cfg.tags.chest)
        if chestNearby.isSome():
          agent.seenChest = true
          let keyLocations = [
            Location(x: -3, y: 0),
            Location(x: 0, y: +3),
            Location(x: +3, y: 0),
            Location(x: 0, y: -3),
          ]
          for keyLocation in keyLocations:
            let location = chestNearby.get() + keyLocation
            agent.exploreLocations.add(location)

    # Do Exploration.
    if agent.exploreLocations.len == 0:
      measurePush("exploration")
      var locationFound = false
      var unexploredLocation: Location
      var visited: HashSet[Location]
      block exploration:
        var seedLocation = agent.location
        let assemblerNearby = agent.cfg.getNearby(agent.location, agent.map, agent.cfg.tags.assembler)
        if assemblerNearby.isSome():
          seedLocation = assemblerNearby.get()
        var queue: Deque[Location]
        queue.addLast(seedLocation)
        visited.incl(seedLocation)
        while queue.len > 0:
          let location = queue.popLast()
          if location notin agent.seen:
            locationFound = true
            unexploredLocation = location
            break exploration
          for i, offset in Offsets4:
            let neighbor = location + offset
            # passable check
            if agent.cfg.isWalkable(agent.map, neighbor):
              if neighbor notin visited:
                visited.incl(neighbor)
                queue.addLast(neighbor)
      if locationFound:
        agent.exploreLocations.add(unexploredLocation)
      else:
        echo "no unseen location found"
        # agent.cfg.drawMap(agent.map, visited)
      measurePop()

    measurePush("explore locations")
    echo "explore locations: " & $agent.exploreLocations
    if agent.exploreLocations.len > 0:
      for location in agent.exploreLocations:
        if location notin agent.seen:
          let action = agent.cfg.aStar(agent.location, location, agent.map)
          if action.isSome():
            doAction(action.get().int32)
            echo "going to explore location: " & $location
            return
          else:
            agent.exploreLocations.remove(location)
            break
        else:
          agent.exploreLocations.remove(location)
          break
    measurePop()

    # If all else fails, take a random move to explore the map or get unstuck.
    let action = agent.random.rand(1 .. 4).int32
    echo "taking random action " & $action
    doAction(action.int32)

    inc agent.stepCount

  except:
    echo getCurrentException().getStackTrace()
    echo getCurrentExceptionMsg()
    quit()

proc newRaceCarPolicy*(environmentConfig: string): RaceCarPolicy =
  let cfg = parseConfig(environmentConfig)
  var agents: seq[RaceCarAgent] = @[]
  for id in 0 ..< cfg.config.numAgents:
    agents.add(newRaceCarAgent(id, environmentConfig))
  return RaceCarPolicy(agents: agents)

proc stepBatch*(
    policy: RaceCarPolicy,
    agentIds: pointer,
    numAgentIds: int,
    numAgents: int,
    numTokens: int,
    sizeToken: int,
    rawObservations: pointer,
    numActions: int,
    rawActions: pointer
) {.measure.} =
  let ids = cast[ptr UncheckedArray[int32]](agentIds)
  let obsArray = cast[ptr UncheckedArray[uint8]](rawObservations)
  let actionArray = cast[ptr UncheckedArray[int32]](rawActions)
  let obsStride = numTokens * sizeToken

  for i in 0 ..< numAgentIds:
    let idx = int(ids[i])
    let obsPtr = cast[pointer](obsArray[idx * obsStride].addr)
    let actPtr = cast[ptr int32](actionArray[idx].addr)
    step(policy.agents[idx], numAgents, numTokens, sizeToken, obsPtr, numActions, actPtr)
