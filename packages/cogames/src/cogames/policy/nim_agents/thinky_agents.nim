import
  std/[strformat, tables, random, sets, options, json],
  common

const
  MaxEnergy = 100
  MaxResourceInventory = 100
  MaxToolInventory = 100
  PutInventoryAmount = 10

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

  ThinkyPolicy* = ref object
    agents*: seq[ThinkyAgent]

proc log(message: string) =
  when defined(debug):
    echo message

const the8Offsets = [
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

proc getActiveRecipe(agent: ThinkyAgent): RecipeInfo =
  ## Get the recipes form the assembler protocol inputs.
  let assemblerLocation = agent.cfg.getNearby(agent.location, agent.map, agent.cfg.tags.assembler)
  if assemblerLocation.isSome():
    let location = assemblerLocation.get()
    # Get the vibe key.
    result.pattern = @[]
    for offsets in the8Offsets:
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

proc newThinkyAgent*(agentId: int, environmentConfig: string): ThinkyAgent =
  ## Create a new thinky agent, the fastest and the smartest agent.

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

  # Apply the offsets and take the best guess about our movement.
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
    log "Looks like we are lost?"
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
    let observations = cast[ptr UncheckedArray[uint8]](rawObservation)

    proc doAction(action: int) =

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

      agent.lastActions.add(action)
      agentAction[] = action.int32

    # Parse the tokens into a vision map.
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
      if location notin map:
        map[location] = @[]
      map[location].add(FeatureValue(featureId: featureId.int, value: value.int))

    #agent.cfg.drawMap(map, initHashSet[Location]())
    updateMap(agent, map)
    #agent.cfg.drawMap(agent.map, agent.seen)

    let vibe = agent.cfg.getVibe(map, Location(x: 0, y: 0))
    log "my vibe: " & $vibe

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

    log &"H:{invHeart} E:{invEnergy} C:{invCarbon} O2:{invOxygen} Ge:{invGermanium} Si:{invSilicon} D:{invDecoder} M:{invModulator} R:{invResonator} S:{invScrambler}"

    let activeRecipe = agent.getActiveRecipe()
    log "active recipe: " & $activeRecipe

    if activeRecipe.pattern.len > 0:
      # Split the cost evenly across the agents.
      agent.carbonTarget = max(agent.carbonTarget, activeRecipe.carbonCost div activeRecipe.pattern.len)
      agent.oxygenTarget = max(agent.oxygenTarget, activeRecipe.oxygenCost div activeRecipe.pattern.len)
      agent.germaniumTarget = max(agent.germaniumTarget, activeRecipe.germaniumCost div activeRecipe.pattern.len)
      agent.siliconTarget = max(agent.siliconTarget, activeRecipe.siliconCost div activeRecipe.pattern.len)
    else:
      agent.carbonTarget = max(agent.carbonTarget, activeRecipe.carbonCost)
      agent.oxygenTarget = max(agent.oxygenTarget, activeRecipe.oxygenCost)
      agent.germaniumTarget = max(agent.germaniumTarget, activeRecipe.germaniumCost)
      agent.siliconTarget = max(agent.siliconTarget, activeRecipe.siliconCost)

    # Are we running low on energy?
    if invEnergy < MaxEnergy div 4:
      let chargerNearby = agent.cfg.getNearby(agent.location, agent.map, agent.cfg.tags.charger)
      if chargerNearby.isSome():
        let action = agent.cfg.aStar(agent.location, chargerNearby.get(), agent.map)
        if action.isSome():
          doAction(action.get().int32)
          log "going to charger"
          return
    # Charge opportunistically.
    if invEnergy < MaxEnergy - 20:
      let chargerNearby = agent.cfg.getNearby(agent.location, agent.map, agent.cfg.tags.charger)
      if chargerNearby.isSome():
        if manhattan(agent.location, chargerNearby.get()) < 2:
          let action = agent.cfg.aStar(agent.location, chargerNearby.get(), agent.map)
          if action.isSome():
            doAction(action.get().int32)
            log "charge nearby might as well charge"
            return

    # Deposit heart into the chest.
    if invHeart > 0:
      # Reset the targets when we deposit hearts.
      log "depositing hearts"
      agent.carbonTarget = 10
      agent.oxygenTarget = 10
      agent.germaniumTarget = 1
      agent.siliconTarget = 20

      let depositAction = agent.cfg.actions.vibeHeartB
      let depositVibe = agent.cfg.vibes.heartB
      if depositAction != 0 and vibe != depositVibe:
        doAction(depositAction.int32)
        return
      let chestNearby = agent.cfg.getNearby(agent.location, agent.map, agent.cfg.tags.chest)
      if chestNearby.isSome():
        let action = agent.cfg.aStar(agent.location, chestNearby.get(), agent.map)
        if action.isSome():
          doAction(action.get().int32)
          log "going to chest"
          return

    if invCarbon >= agent.carbonTarget and invOxygen >= agent.oxygenTarget and invGermanium >= agent.germaniumTarget and invSilicon >= agent.siliconTarget:
      # We have all the resources we need, so we can build a heart.
      log "trying to build a heart"
      var assembleAction = agent.cfg.actions.vibeHeartA
      var assembleVibe = agent.cfg.vibes.heartA
      if assembleAction == 0:
        assembleAction = agent.cfg.actions.vibeHeartB
        assembleVibe = agent.cfg.vibes.heartB
      if assembleAction != 0 and vibe != assembleVibe:
        doAction(assembleAction.int32)
        log "vibing heart for assembler"
        return

      let assemblerNearby = agent.cfg.getNearby(agent.location, agent.map, agent.cfg.tags.assembler)
      if assemblerNearby.isSome():
        let action = agent.cfg.aStar(agent.location, assemblerNearby.get(), agent.map)
        if action.isSome():
          doAction(action.get().int32)
          log "going to assembler to build heart"
          return

    # Dump excess resources.
    let atMaxInventory = invCarbon + invOxygen + invGermanium + invSilicon >= MaxResourceInventory
    let avgResource = (invCarbon + invOxygen + invGermanium + invSilicon) div 4
    if atMaxInventory:
      log "at max inventory"

    if atMaxInventory and invCarbon > avgResource and invCarbon > agent.carbonTarget + PutInventoryAmount:
      let chestNearby = agent.cfg.getNearby(agent.location, agent.map, agent.cfg.tags.chest)
      if chestNearby.isSome():
        if vibe != agent.cfg.vibes.carbonB:
          doAction(agent.cfg.actions.vibeCarbonB.int32)
          log "vibing carbon B to dump excess carbon"
          log "avgResource: " & $avgResource & " carbonTarget: " & $agent.carbonTarget & " putInventoryAmount: " & $PutInventoryAmount
          return
        let action = agent.cfg.aStar(agent.location, chestNearby.get(), agent.map)
        if action.isSome():
          doAction(action.get().int32)
          log "going to chest to dump excess carbon"
          return

    if atMaxInventory and invSilicon > avgResource and invSilicon > agent.siliconTarget + PutInventoryAmount:
      let chestNearby = agent.cfg.getNearby(agent.location, agent.map, agent.cfg.tags.chest)
      if chestNearby.isSome():
        if vibe != agent.cfg.vibes.siliconB:
          doAction(agent.cfg.actions.vibeSiliconB.int32)
          log "vibing silicon B to dump excess silicon"
          log "avgResource: " & $avgResource & " siliconTarget: " & $agent.siliconTarget & " putInventoryAmount: " & $PutInventoryAmount
          return
        let action = agent.cfg.aStar(agent.location, chestNearby.get(), agent.map)
        if action.isSome():
          doAction(action.get().int32)
          log "going to chest to dump excess silicon"
          return

    if atMaxInventory and invOxygen > avgResource and invOxygen > agent.oxygenTarget + PutInventoryAmount:
      let chestNearby = agent.cfg.getNearby(agent.location, agent.map, agent.cfg.tags.chest)
      if chestNearby.isSome():
        if vibe != agent.cfg.vibes.oxygenB:
          doAction(agent.cfg.actions.vibeOxygenB.int32)
          log "vibing oxygen B to dump excess oxygen"
          log "avgResource: " & $avgResource & " oxygenTarget: " & $agent.oxygenTarget & " putInventoryAmount: " & $PutInventoryAmount
          return
        let action = agent.cfg.aStar(agent.location, chestNearby.get(), agent.map)
        if action.isSome():
          doAction(action.get().int32)
          log "going to chest to dump excess oxygen"
          return

    if atMaxInventory and invGermanium > avgResource and invGermanium > agent.germaniumTarget + PutInventoryAmount:
      let chestNearby = agent.cfg.getNearby(agent.location, agent.map, agent.cfg.tags.chest)
      if chestNearby.isSome():
        if vibe != agent.cfg.vibes.germaniumB:
          doAction(agent.cfg.actions.vibeGermaniumB.int32)
          log "vibing germanium B to dump excess germanium"
          log "avgResource: " & $avgResource & " germaniumTarget: " & $agent.germaniumTarget & " putInventoryAmount: " & $PutInventoryAmount
          return
        let action = agent.cfg.aStar(agent.location, chestNearby.get(), agent.map)
        if action.isSome():
          doAction(action.get().int32)
          log "going to chest to dump excess germanium"
          return

    # Is there carbon nearby?
    if agent.carbonTarget > 0 and invCarbon < agent.carbonTarget:
      var closeChest = agent.cfg.getNearby(agent.location, agent.map, agent.cfg.tags.chest)
      if closeChest.isSome():
        # Does it have the resources we need?
        let chestInventory = agent.cfg.getInventory(agent.map, agent.cfg.features.invCarbon, closeChest.get())
        if chestInventory > 0:
          # Vibe the right resource to take from the chest.
          if vibe != agent.cfg.vibes.carbonA:
            doAction(agent.cfg.actions.vibeCarbonA.int32)
            log "vibing carbon A to take from chest"
            return
          let action = agent.cfg.aStar(agent.location, closeChest.get(), agent.map)
          if action.isSome():
            doAction(action.get().int32)
            log "going to chest to take carbon from chest"
        let action = agent.cfg.aStar(agent.location, closeChest.get(), agent.map)
        if action.isSome():
          doAction(action.get().int32)
      let carbonNearby = agent.cfg.getNearby(agent.location, agent.map, agent.cfg.tags.carbonExtractor)
      if carbonNearby.isSome():
        let action = agent.cfg.aStar(agent.location, carbonNearby.get(), agent.map)
        if action.isSome():
          doAction(action.get().int32)
          log "going to carbon, need: " & $agent.carbonTarget & " have: " & $invCarbon
          return

    # Is there silicon nearby?
    if agent.siliconTarget > 0 and invSilicon < agent.siliconTarget:
      var closeChest = agent.cfg.getNearby(agent.location, agent.map, agent.cfg.tags.chest)
      if closeChest.isSome():
        # Does it have the resources we need?
        let chestInventory = agent.cfg.getInventory(agent.map, agent.cfg.features.invSilicon, closeChest.get())
        if chestInventory > 0:
          # Vibe the right resource to take from the chest.
          if vibe != agent.cfg.vibes.siliconA:
            doAction(agent.cfg.actions.vibeSiliconA.int32)
            log "vibing silicon A to take from chest"
            return
          let action = agent.cfg.aStar(agent.location, closeChest.get(), agent.map)
          if action.isSome():
            doAction(action.get().int32)
            log "going to chest to take silicon from chest"
            return
      let siliconNearby = agent.cfg.getNearby(agent.location, agent.map, agent.cfg.tags.siliconExtractor)
      if siliconNearby.isSome():
        let action = agent.cfg.aStar(agent.location, siliconNearby.get(), agent.map)
        if action.isSome():
          doAction(action.get().int32)
          log "going to silicon, need: " & $agent.siliconTarget & " have: " & $invSilicon
          return

    # Is there oxygen nearby?
    if agent.oxygenTarget > 0 and invOxygen < agent.oxygenTarget:
      var closeChest = agent.cfg.getNearby(agent.location, agent.map, agent.cfg.tags.chest)
      if closeChest.isSome():
        # Does it have the resources we need?
        let chestInventory = agent.cfg.getInventory(agent.map, agent.cfg.features.invOxygen, closeChest.get())
        if chestInventory > 0:
          # Vibe the right resource to take from the chest.
          if vibe != agent.cfg.vibes.oxygenA:
            doAction(agent.cfg.actions.vibeOxygenA.int32)
            log "vibing oxygen A to take from chest"
            return
          let action = agent.cfg.aStar(agent.location, closeChest.get(), agent.map)
          if action.isSome():
            doAction(action.get().int32)
            log "going to chest to take oxygen from chest"
            return
      let oxygenNearby = agent.cfg.getNearby(agent.location, agent.map, agent.cfg.tags.oxygenExtractor)
      if oxygenNearby.isSome():
        let action = agent.cfg.aStar(agent.location, oxygenNearby.get(), agent.map)
        if action.isSome():
          doAction(action.get().int32)
          log "going to oxygen, need: " & $agent.oxygenTarget & " have: " & $invOxygen
          return

    # Is there germanium nearby?
    if agent.germaniumTarget > 0 and invGermanium < agent.germaniumTarget:
      var closeChest = agent.cfg.getNearby(agent.location, agent.map, agent.cfg.tags.chest)
      if closeChest.isSome():
        # Does it have the resources we need?
        let chestInventory = agent.cfg.getInventory(agent.map, agent.cfg.features.invGermanium, closeChest.get())
        if chestInventory > 0:
          # Vibe the right resource to take from the chest.
          if vibe != agent.cfg.vibes.germaniumA:
            doAction(agent.cfg.actions.vibeGermaniumA.int32)
            log "vibing germanium A to take from chest"
            return
          let action = agent.cfg.aStar(agent.location, closeChest.get(), agent.map)
          if action.isSome():
            doAction(action.get().int32)
            log "going to chest to take germanium from chest"
            return
      let germaniumNearby = agent.cfg.getNearby(agent.location, agent.map, agent.cfg.tags.germaniumExtractor)
      if germaniumNearby.isSome():
        let action = agent.cfg.aStar(agent.location, germaniumNearby.get(), agent.map)
        if action.isSome():
          doAction(action.get().int32)
          log "going to germanium, need: " & $agent.germaniumTarget & " have: " & $invGermanium
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
            log "going to key location to explore"
            return

    # Try to find a nearby unseen location nearest to the agent.
    let unseenNearby = agent.cfg.getNearbyUnseen(agent.location, agent.map, agent.seen)
    if unseenNearby.isSome():
      let action = agent.cfg.aStar(agent.location, unseenNearby.get(), agent.map)
      if action.isSome():
        doAction(action.get().int32)
        log "going to unseen location nearest to agent"
        return

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
