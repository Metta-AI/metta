import
  std/[strformat, tables, random, sets, options, json, deques],
  fidget2/measure,
  common

const
  MaxEnergy = 100
  MaxResourceInventory = 100
  MaxToolInventory = 100

  PutCarbonAmount = 10
  PutOxygenAmount = 10
  PutGermaniumAmount = 1
  PutSiliconAmount = 25

type
  ThinkyAgent* = ref object
    agentId*: int

    map: Table[Location, seq[FeatureValue]]
    seen: HashSet[Location]
    cfg: Config
    random: Rand
    location: Location
    lastActions: seq[int]
    recipes: seq[RecipeInfo]

    carbonTarget: int
    oxygenTarget: int
    germaniumTarget: int
    siliconTarget: int

    bump: bool
    offsets4: seq[Location]  # 4 cardinal but random for each agent
    seenAssembler: bool
    seenChest: bool
    exploreLocations: seq[Location]

  ThinkyPolicy* = ref object
    agents*: seq[ThinkyAgent]

proc log(message: string) =
  when defined(debug):
    echo message

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

proc getActiveRecipe(agent: ThinkyAgent): RecipeInfo {.measure.} =
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
        result.energyOutput = feature.value
      elif feature.featureId == agent.cfg.features.protocolOutputCarbon:
        result.carbonOutput = feature.value
      elif feature.featureId == agent.cfg.features.protocolOutputOxygen:
        result.oxygenOutput = feature.value
      elif feature.featureId == agent.cfg.features.protocolOutputGermanium:
        result.germaniumOutput = feature.value
      elif feature.featureId == agent.cfg.features.protocolOutputSilicon:
        result.siliconOutput = feature.value
      elif feature.featureId == agent.cfg.features.protocolOutputHeart:
        result.heartOutput = feature.value
      elif feature.featureId == agent.cfg.features.protocolOutputDecoder:
        result.decoderOutput = feature.value
      elif feature.featureId == agent.cfg.features.protocolOutputModulator:
        result.modulatorOutput = feature.value
      elif feature.featureId == agent.cfg.features.protocolOutputResonator:
        result.resonatorOutput = feature.value
      elif feature.featureId == agent.cfg.features.protocolOutputScrambler:
        result.scramblerOutput = feature.value

proc newThinkyAgent*(agentId: int, environmentConfig: string): ThinkyAgent =
  ## Create a new thinky agent, the fastest and the smartest agent.

  var config = parseConfig(environmentConfig)
  result = ThinkyAgent(agentId: agentId, cfg: config)
  result.random = initRand(agentId)
  result.map = initTable[Location, seq[FeatureValue]]()
  result.seen = initHashSet[Location]()
  result.location = Location(x: 0, y: 0)
  result.lastActions = @[]
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

proc updateMap(agent: ThinkyAgent, visible: Table[Location, seq[FeatureValue]]) {.measure.} =
  ## Update the big map with the small visible map.

  if agent.map.len == 0:
    # First time we're called, just copy the visible map to the big map.
    agent.map = visible
    agent.location = Location(x: 0, y: 0)
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

proc step*(
  agent: ThinkyAgent,
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
          log "Stuck prevention: left, right, left"
          doAction(agent.cfg.actions.noop.int32)
          return
      if agent.lastActions.len >= 2 and
        action == agent.cfg.actions.moveEast and
        agent.lastActions[^1] == agent.cfg.actions.moveWest and
        agent.lastActions[^2] == agent.cfg.actions.moveEast:
        # Noop 50% of the time.
        if agent.random.rand(1 .. 2) == 1:
          log "Stuck prevention: right, left, right"
          doAction(agent.cfg.actions.noop.int32)
          return
      if agent.lastActions.len >= 2 and
        action == agent.cfg.actions.moveNorth and
        agent.lastActions[^1] == agent.cfg.actions.moveSouth and
        agent.lastActions[^2] == agent.cfg.actions.moveNorth:
        # Noop 50% of the time.
        if agent.random.rand(1 .. 2) == 1:
          log "Stuck prevention: north, south, north"
          doAction(agent.cfg.actions.noop.int32)
          return
      if agent.lastActions.len >= 2 and
        action == agent.cfg.actions.moveSouth and
        agent.lastActions[^1] == agent.cfg.actions.moveNorth and
        agent.lastActions[^2] == agent.cfg.actions.moveSouth:
        # Noop 50% of the time.
        if agent.random.rand(1 .. 2) == 1:
          log "Stuck prevention: south, north, south"
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

    log &"vibe:{vibe} H:{invHeart} E:{invEnergy} C:{invCarbon} O2:{invOxygen} Ge:{invGermanium} Si:{invSilicon} D:{invDecoder} M:{invModulator} R:{invResonator} S:{invScrambler}"

    let activeRecipe = agent.getActiveRecipe()
    log "active recipe: " & $activeRecipe

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

    # Are we running low on energy?
    if invEnergy < MaxEnergy div 4:
      let chargerNearby = agent.cfg.getNearbyExtractor(agent.location, agent.map, agent.cfg.tags.charger)
      if chargerNearby.isSome():
        measurePush("charger nearby")
        let action = agent.cfg.aStar(agent.location, chargerNearby.get(), agent.map)
        measurePop()
        if action.isSome():
          doAction(action.get().int32)
          log "going to charger"
          return

    # Charge opportunistically.
    if invEnergy < MaxEnergy - 20:
      let chargerNearby = agent.cfg.getNearbyExtractor(agent.location, agent.map, agent.cfg.tags.charger)
      if chargerNearby.isSome():
        if manhattan(agent.location, chargerNearby.get()) < 2:
          measurePush("charge nearby")
          let action = agent.cfg.aStar(agent.location, chargerNearby.get(), agent.map)
          measurePop()
          if action.isSome():
            doAction(action.get().int32)
            log "charge nearby might as well charge"
            return

    # Deposit heart into the chest.
    if invHeart > 0:

      # Reset the targets when we deposit hearts.
      # Reset to 0 (not hardcoded defaults) so the recipe-reading logic
      # can set correct targets from activeRecipe on the next cycle.
      log "depositing hearts"
      agent.carbonTarget = 0
      agent.oxygenTarget = 0
      agent.germaniumTarget = 0
      agent.siliconTarget = 0

      let depositAction = agent.cfg.actions.vibeHeartB
      let depositVibe = agent.cfg.vibes.heartB
      if depositAction != 0 and vibe != depositVibe:
        doAction(depositAction.int32)
        return
      let chestNearby = agent.cfg.getNearby(agent.location, agent.map, agent.cfg.tags.chest)
      if chestNearby.isSome():
        measurePush("chest nearby")
        let action = agent.cfg.aStar(agent.location, chestNearby.get(), agent.map)
        measurePop()
        if action.isSome():
          doAction(action.get().int32)
          log "going to chest"
          return

    if invCarbon >= agent.carbonTarget and invOxygen >= agent.oxygenTarget and invGermanium >= agent.germaniumTarget and invSilicon >= agent.siliconTarget:
      # We have all the resources we need, so we can build a heart.
      log "trying to build a heart"

      if vibe != agent.cfg.vibes.heartA:
        doAction(agent.cfg.actions.vibeHeartA.int32)
        log "vibing heart for assembler"
        return

      let assemblerNearby = agent.cfg.getNearby(agent.location, agent.map, agent.cfg.tags.assembler)
      if assemblerNearby.isSome():
        measurePush("assembler nearby")
        let action = agent.cfg.aStar(agent.location, assemblerNearby.get(), agent.map)
        measurePop()
        if action.isSome():
          doAction(action.get().int32)
          log "going to assembler to build heart"
          return

    # Dump excess resources.
    let atMaxInventory = invCarbon + invOxygen + invGermanium + invSilicon >= MaxResourceInventory
    let avgResource = (invCarbon + invOxygen + invGermanium + invSilicon) div 4
    if atMaxInventory:
      log "at max inventory"

    if atMaxInventory and invCarbon > avgResource and invCarbon > agent.carbonTarget + PutCarbonAmount:
      let chestNearby = agent.cfg.getNearby(agent.location, agent.map, agent.cfg.tags.chest)
      if chestNearby.isSome():
        if vibe != agent.cfg.vibes.carbonB:
          doAction(agent.cfg.actions.vibeCarbonB.int32)
          log "vibing carbon B to dump excess carbon"
          return
        measurePush("chest nearby excess carbon")
        let action = agent.cfg.aStar(agent.location, chestNearby.get(), agent.map)
        measurePop()
        if action.isSome():
          doAction(action.get().int32)
          log "going to chest to dump excess carbon"
          return

    if atMaxInventory and invSilicon > avgResource and invSilicon > agent.siliconTarget + PutSiliconAmount:
      let chestNearby = agent.cfg.getNearby(agent.location, agent.map, agent.cfg.tags.chest)
      if chestNearby.isSome():
        if vibe != agent.cfg.vibes.siliconB:
          doAction(agent.cfg.actions.vibeSiliconB.int32)
          log "vibing silicon B to dump excess silicon"
          return
        let action = agent.cfg.aStar(agent.location, chestNearby.get(), agent.map)
        if action.isSome():
          doAction(action.get().int32)
          log "going to chest to dump excess silicon"
          return

    if atMaxInventory and invOxygen > avgResource and invOxygen > agent.oxygenTarget + PutOxygenAmount:
      let chestNearby = agent.cfg.getNearby(agent.location, agent.map, agent.cfg.tags.chest)
      if chestNearby.isSome():
        if vibe != agent.cfg.vibes.oxygenB:
          doAction(agent.cfg.actions.vibeOxygenB.int32)
          log "vibing oxygen B to dump excess oxygen"
          return
        measurePush("chest nearby excess oxygen")
        let action = agent.cfg.aStar(agent.location, chestNearby.get(), agent.map)
        measurePop()
        if action.isSome():
          doAction(action.get().int32)
          log "going to chest to dump excess oxygen"
          return

    if atMaxInventory and invGermanium > avgResource and invGermanium > agent.germaniumTarget + PutGermaniumAmount:
      let chestNearby = agent.cfg.getNearby(agent.location, agent.map, agent.cfg.tags.chest)
      if chestNearby.isSome():
        if vibe != agent.cfg.vibes.germaniumB:
          doAction(agent.cfg.actions.vibeGermaniumB.int32)
          log "vibing germanium B to dump excess germanium"
          return
        measurePush("chest nearby excess germanium")
        let action = agent.cfg.aStar(agent.location, chestNearby.get(), agent.map)
        measurePop()
        if action.isSome():
          doAction(action.get().int32)
          log "going to chest to dump excess germanium"
          return

    proc findAndTakeResource(
      agent: ThinkyAgent,
      vibe: int,
      resource: int,
      target: int,
      inventory: int,
      vibeGetResource: int,
      vibeAction: int,
      extractorTag: int,
      name: string
    ): bool {.measure.} =
      # Check the chest.
      var closeChest = agent.cfg.getNearby(agent.location, agent.map, agent.cfg.tags.chest)
      if closeChest.isSome():
        # Does it have the resources we need?
        let chestInventory = agent.cfg.getInventory(agent.map, resource, closeChest.get())
        if chestInventory > 0:
          # Vibe the right resource to take from the chest.
          if vibe != vibeGetResource:
            doAction(vibeAction.int32)
            log "vibing " & name & " to take from chest"
            return true
          measurePush("chest nearby to take " & name)
          let action = agent.cfg.aStar(agent.location, closeChest.get(), agent.map)
          measurePop()
          if action.isSome():
            doAction(action.get().int32)
            log "going to chest to take " & name & " from chest"
            return true

      # Check the carbon extractor.
      let extractorNearby = agent.cfg.getNearbyExtractor(agent.location, agent.map, extractorTag)
      if extractorNearby.isSome():
        measurePush("extractor nearby to take " & name)
        let action = agent.cfg.aStar(agent.location, extractorNearby.get(), agent.map)
        measurePop()
        if action.isSome():
          doAction(action.get().int32)
          log "going to " & name & ", need: " & $target & " have: " & $inventory
          return true

    # Is there carbon nearby?
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

    # Is there silicon nearby?
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

    # Is there oxygen nearby?
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

    # Is there germanium nearby?
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
        log "no unseen location found"
        # agent.cfg.drawMap(agent.map, visited)
      measurePop()

    measurePush("explore locations")
    log "explore locations: " & $agent.exploreLocations
    if agent.exploreLocations.len > 0:
      for location in agent.exploreLocations:
        if location notin agent.seen:
          let action = agent.cfg.aStar(agent.location, location, agent.map)
          if action.isSome():
            doAction(action.get().int32)
            log "going to explore location: " & $location
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
    log "taking random action " & $action
    doAction(action.int32)

  except:
    echo getCurrentException().getStackTrace()
    echo getCurrentExceptionMsg()
    quit()

proc newThinkyPolicy*(environmentConfig: string): ThinkyPolicy =
  let cfg = parseConfig(environmentConfig)
  var agents: seq[ThinkyAgent] = @[]
  for id in 0 ..< cfg.config.numAgents:
    agents.add(newThinkyAgent(id, environmentConfig))
  return ThinkyPolicy(agents: agents)

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
