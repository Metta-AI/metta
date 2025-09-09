import std/[json, times, strformat], zippy, vmath, jsony
import environment, common

type
  TribalItemAmount* = object
    itemId*: int
    count*: int

  TribalEntity* = ref object
    # Common keys
    id*: int
    typeId*: int
    groupId*: int
    agentId*: int
    location*: seq[IVec2]
    orientation*: seq[int]
    inventory*: seq[seq[TribalItemAmount]]
    inventoryMax*: int
    color*: seq[int]

    # Agent specific keys
    actionId*: seq[int]
    actionParameter*: seq[int]
    actionSuccess*: seq[bool]
    currentReward*: seq[float32]
    totalReward*: seq[float32]
    isFrozen*: seq[bool]
    frozenProgress*: seq[int]
    frozenTime*: int

    # Building specific keys (altars, mines, converters, etc.)
    resourcesAvailable*: seq[int]  # For mines - remaining ore
    cooldownProgress*: seq[int]
    cooldownTime*: int
    hearts*: seq[int]  # For altars - number of hearts

    # Computed fields
    gainMap*: seq[seq[TribalItemAmount]]
    isAgent*: bool

  TribalReplay* = ref object
    version*: int
    numAgents*: int
    maxSteps*: int
    mapSize*: (int, int)
    fileName*: string
    typeNames*: seq[string]
    actionNames*: seq[string] 
    itemNames*: seq[string]
    groupNames*: seq[string]
    itemImages*: seq[string]
    traceImages*: seq[string]
    objects*: seq[TribalEntity]
    agents*: seq[TribalEntity]
    tribalConfig*: JsonNode

  TribalReplayRecorder* = ref object
    replay*: TribalReplay
    environment*: Environment
    currentStep*: int
    recording*: bool

# Item type mappings for tribal environment
const TribalItemTypes = [
  "ore",
  "battery", 
  "water",
  "wheat",
  "wood",
  "spear",
  "hat",
  "armor"
]

const TribalThingTypes = [
  "Agent",
  "Wall", 
  "Mine",
  "Converter",
  "Altar",
  "Spawner",
  "Clippy",
  "Armory",
  "Forge",
  "ClayOven",
  "WeavingLoom"
]

const TribalActionTypes = [
  "noop",
  "move",
  "attack", 
  "get",
  "swap",
  "put"
]

proc expand[T](data: auto, numSteps: int, defaultValue: T): seq[T] =
  ## Expand data arrays for replay format
  if data == nil:
    return @[defaultValue]
  elif data.kind == JArray:
    if data.len == 0:
      return @[defaultValue]
    elif data[0].kind != JArray:
      return @[data.to(T)]
    else:
      # Expand sequence of step-value pairs
      var i = 0
      var j = 0
      var v: T = defaultValue
      for i in 0 ..< numSteps:
        if j < data.len and data[j].kind == JArray and data[j][0].kind == JInt and data[j][0].getInt == i:
          v = data[j][1].to(T)
          j += 1
        result.add(v)
  else:
    return @[data.to(T)]

proc newTribalReplayRecorder*(env: Environment, fileName: string = ""): TribalReplayRecorder =
  ## Create a new replay recorder for the tribal environment
  result = TribalReplayRecorder(
    environment: env,
    currentStep: 0,
    recording: false
  )
  
  result.replay = TribalReplay(
    version: 2,
    numAgents: MapAgents,
    maxSteps: 0,  # Will be set when recording stops
    mapSize: (MapWidth, MapHeight),
    fileName: fileName,
    typeNames: @TribalThingTypes,
    actionNames: @TribalActionTypes,
    itemNames: @TribalItemTypes,
    groupNames: @[],  # Could be used for tribe/village grouping
    itemImages: @[],
    traceImages: @[],
    objects: @[],
    agents: @[]
  )

proc startRecording*(recorder: TribalReplayRecorder) =
  ## Start recording a replay
  recorder.recording = true
  recorder.currentStep = 0
  recorder.replay.objects.setLen(0)
  recorder.replay.agents.setLen(0)
  
  # Initialize entities from environment
  var entityId = 0
  for thing in recorder.environment.things:
    let entity = TribalEntity(
      id: entityId,
      typeId: ord(thing.kind),
      location: @[thing.pos],
      orientation: @[ord(thing.orientation)],
      inventory: @[@[]],  # Will be populated for agents
      color: @[0]  # Could be mapped from village colors
    )
    
    if thing.kind == Agent:
      entity.isAgent = true
      entity.agentId = thing.agentId
      entity.groupId = 0  # Could be village ID
      entity.actionId = @[0]
      entity.actionParameter = @[0] 
      entity.actionSuccess = @[false]
      entity.currentReward = @[thing.reward]
      entity.totalReward = @[thing.reward]
      entity.isFrozen = @[thing.frozen > 0]
      entity.frozenProgress = @[thing.frozen]
      
      # Initialize inventory
      var inventoryItems: seq[TribalItemAmount] = @[]
      if thing.inventoryOre > 0:
        inventoryItems.add(TribalItemAmount(itemId: 0, count: thing.inventoryOre))
      if thing.inventoryBattery > 0:
        inventoryItems.add(TribalItemAmount(itemId: 1, count: thing.inventoryBattery))
      if thing.inventoryWater > 0:
        inventoryItems.add(TribalItemAmount(itemId: 2, count: thing.inventoryWater))
      if thing.inventoryWheat > 0:
        inventoryItems.add(TribalItemAmount(itemId: 3, count: thing.inventoryWheat))
      if thing.inventoryWood > 0:
        inventoryItems.add(TribalItemAmount(itemId: 4, count: thing.inventoryWood))
      if thing.inventorySpear > 0:
        inventoryItems.add(TribalItemAmount(itemId: 5, count: thing.inventorySpear))
      if thing.inventoryHat > 0:
        inventoryItems.add(TribalItemAmount(itemId: 6, count: thing.inventoryHat))
      if thing.inventoryArmor > 0:
        inventoryItems.add(TribalItemAmount(itemId: 7, count: thing.inventoryArmor))
      
      entity.inventory = @[inventoryItems]
      recorder.replay.agents.add(entity)
      
    elif thing.kind == Mine:
      entity.resourcesAvailable = @[thing.resources]
      entity.cooldownProgress = @[thing.cooldown]
      entity.cooldownTime = thing.cooldown
      
    elif thing.kind == Altar:
      entity.hearts = @[thing.hearts]
      entity.cooldownProgress = @[thing.cooldown]
      entity.cooldownTime = thing.cooldown
      
    elif thing.kind in {Converter, Spawner, Armory, Forge, ClayOven, WeavingLoom}:
      entity.cooldownProgress = @[thing.cooldown]
      entity.cooldownTime = thing.cooldown
    
    recorder.replay.objects.add(entity)
    entityId += 1

proc recordStep*(recorder: TribalReplayRecorder, actions: ptr array[MapAgents, array[2, uint8]]) =
  ## Record a single step of the environment
  if not recorder.recording:
    return
    
  recorder.currentStep += 1
  
  # Update entity states
  for i, thing in recorder.environment.things:
    if i >= recorder.replay.objects.len:
      continue
      
    let entity = recorder.replay.objects[i]
    
    # Record position if changed
    if entity.location.len == 0 or entity.location[^1] != thing.pos:
      entity.location.add(thing.pos)
    
    # Record orientation if changed
    let currentOrientation = ord(thing.orientation)
    if entity.orientation.len == 0 or entity.orientation[^1] != currentOrientation:
      entity.orientation.add(currentOrientation)
    
    if thing.kind == Agent:
      let agentId = thing.agentId
      if agentId < MapAgents:
        # Record action
        entity.actionId.add(actions[agentId][0].int)
        entity.actionParameter.add(actions[agentId][1].int)
        entity.actionSuccess.add(true)  # Could track success/failure
        
        # Record rewards
        entity.currentReward.add(thing.reward)
        if entity.totalReward.len > 0:
          entity.totalReward.add(entity.totalReward[^1] + thing.reward)
        else:
          entity.totalReward.add(thing.reward)
        
        # Record frozen state
        entity.isFrozen.add(thing.frozen > 0)
        entity.frozenProgress.add(thing.frozen)
        
        # Record inventory changes
        var inventoryItems: seq[TribalItemAmount] = @[]
        if thing.inventoryOre > 0:
          inventoryItems.add(TribalItemAmount(itemId: 0, count: thing.inventoryOre))
        if thing.inventoryBattery > 0:
          inventoryItems.add(TribalItemAmount(itemId: 1, count: thing.inventoryBattery))
        if thing.inventoryWater > 0:
          inventoryItems.add(TribalItemAmount(itemId: 2, count: thing.inventoryWater))
        if thing.inventoryWheat > 0:
          inventoryItems.add(TribalItemAmount(itemId: 3, count: thing.inventoryWheat))
        if thing.inventoryWood > 0:
          inventoryItems.add(TribalItemAmount(itemId: 4, count: thing.inventoryWood))
        if thing.inventorySpear > 0:
          inventoryItems.add(TribalItemAmount(itemId: 5, count: thing.inventorySpear))
        if thing.inventoryHat > 0:
          inventoryItems.add(TribalItemAmount(itemId: 6, count: thing.inventoryHat))
        if thing.inventoryArmor > 0:
          inventoryItems.add(TribalItemAmount(itemId: 7, count: thing.inventoryArmor))
          
        entity.inventory.add(inventoryItems)
    
    elif thing.kind == Mine:
      entity.resourcesAvailable.add(thing.resources)
      entity.cooldownProgress.add(thing.cooldown)
      
    elif thing.kind == Altar:
      entity.hearts.add(thing.hearts)
      entity.cooldownProgress.add(thing.cooldown)
      
    elif thing.kind in {Converter, Spawner, Armory, Forge, ClayOven, WeavingLoom}:
      entity.cooldownProgress.add(thing.cooldown)

proc stopRecording*(recorder: TribalReplayRecorder): string =
  ## Stop recording and return compressed replay data
  if not recorder.recording:
    return ""
    
  recorder.recording = false
  recorder.replay.maxSteps = recorder.currentStep
  
  # Compute gain maps for agents (inventory changes over time)
  for agent in recorder.replay.agents:
    agent.gainMap = newSeq[seq[TribalItemAmount]](recorder.replay.maxSteps)
    
    if agent.inventory.len > 0:
      # Set initial gain map (first step inventory)
      agent.gainMap[0] = agent.inventory[0]
      
      # Calculate gains/losses for subsequent steps
      for step in 1 ..< min(agent.inventory.len, recorder.replay.maxSteps):
        let currentInv = agent.inventory[step]
        let prevInv = agent.inventory[step - 1]
        
        var gainItems: seq[TribalItemAmount] = @[]
        
        # Create maps for easier comparison
        var currentCounts: array[8, int]
        var prevCounts: array[8, int]
        
        for item in currentInv:
          if item.itemId < 8:
            currentCounts[item.itemId] = item.count
            
        for item in prevInv:
          if item.itemId < 8:
            prevCounts[item.itemId] = item.count
        
        # Record differences
        for itemId in 0 ..< 8:
          let diff = currentCounts[itemId] - prevCounts[itemId]
          if diff != 0:
            gainItems.add(TribalItemAmount(itemId: itemId, count: diff))
        
        agent.gainMap[step] = gainItems
  
  # Convert to JSON manually to handle tuple types
  var jsonReplay = newJObject()
  jsonReplay["version"] = %recorder.replay.version
  jsonReplay["numAgents"] = %recorder.replay.numAgents
  jsonReplay["maxSteps"] = %recorder.replay.maxSteps
  jsonReplay["mapSize"] = %[recorder.replay.mapSize[0], recorder.replay.mapSize[1]]
  jsonReplay["fileName"] = %recorder.replay.fileName
  jsonReplay["typeNames"] = %recorder.replay.typeNames
  jsonReplay["actionNames"] = %recorder.replay.actionNames
  jsonReplay["itemNames"] = %recorder.replay.itemNames
  jsonReplay["groupNames"] = %recorder.replay.groupNames
  # Convert objects manually to handle IVec2 serialization
  var objectsArray = newJArray()
  for obj in recorder.replay.objects:
    var objJson = newJObject()
    objJson["id"] = %obj.id
    objJson["typeId"] = %obj.typeId
    
    # Convert location sequence to array of coordinate arrays
    var locationArray = newJArray()
    for loc in obj.location:
      locationArray.add(%[loc.x.int, loc.y.int])
    objJson["location"] = locationArray
    
    objJson["orientation"] = %obj.orientation
    objJson["inventory"] = %obj.inventory
    objJson["color"] = %obj.color
    
    if obj.isAgent:
      objJson["agentId"] = %obj.agentId
      objJson["groupId"] = %obj.groupId
      objJson["actionId"] = %obj.actionId
      objJson["actionParameter"] = %obj.actionParameter
      objJson["actionSuccess"] = %obj.actionSuccess
      objJson["currentReward"] = %obj.currentReward
      objJson["totalReward"] = %obj.totalReward
      objJson["isFrozen"] = %obj.isFrozen
      objJson["frozenProgress"] = %obj.frozenProgress
    
    if obj.resourcesAvailable.len > 0:
      objJson["resourcesAvailable"] = %obj.resourcesAvailable
    if obj.cooldownProgress.len > 0:
      objJson["cooldownProgress"] = %obj.cooldownProgress
    if obj.hearts.len > 0:
      objJson["hearts"] = %obj.hearts
      
    objectsArray.add(objJson)
  
  jsonReplay["objects"] = objectsArray
  
  # Convert agents manually to handle IVec2 serialization
  var agentsArray = newJArray()
  for agent in recorder.replay.agents:
    var agentJson = newJObject()
    agentJson["id"] = %agent.id
    agentJson["typeId"] = %agent.typeId
    agentJson["agentId"] = %agent.agentId
    agentJson["groupId"] = %agent.groupId
    
    # Convert location sequence to array of coordinate arrays
    var locationArray = newJArray()
    for loc in agent.location:
      locationArray.add(%[loc.x.int, loc.y.int])
    agentJson["location"] = locationArray
    
    agentJson["orientation"] = %agent.orientation
    agentJson["inventory"] = %agent.inventory
    agentJson["color"] = %agent.color
    agentJson["actionId"] = %agent.actionId
    agentJson["actionParameter"] = %agent.actionParameter
    agentJson["actionSuccess"] = %agent.actionSuccess
    agentJson["currentReward"] = %agent.currentReward
    agentJson["totalReward"] = %agent.totalReward
    agentJson["isFrozen"] = %agent.isFrozen
    agentJson["frozenProgress"] = %agent.frozenProgress
    
    agentsArray.add(agentJson)
  
  jsonReplay["agents"] = agentsArray
  if not isNil(recorder.replay.tribalConfig):
    jsonReplay["tribalConfig"] = recorder.replay.tribalConfig
  
  let jsonStr = $jsonReplay
  
  # Compress with zippy
  let compressedData = zippy.compress(jsonStr)
  return compressedData

proc saveReplay*(recorder: TribalReplayRecorder, fileName: string) =
  ## Save the recorded replay to a file
  let compressedData = recorder.stopRecording()
  if compressedData.len > 0:
    writeFile(fileName, compressedData)

proc loadTribalReplay*(data: string, fileName: string): TribalReplay =
  ## Load a tribal replay from compressed data
  let jsonData = zippy.uncompress(data)
  let jsonObj = fromJson(jsonData)
  
  doAssert jsonObj["version"].getInt == 2
  
  let replay = TribalReplay(
    version: jsonObj["version"].getInt,
    actionNames: jsonObj["actionNames"].to(seq[string]),
    itemNames: jsonObj["itemNames"].to(seq[string]),
    typeNames: jsonObj["typeNames"].to(seq[string]),
    numAgents: jsonObj["numAgents"].getInt,
    maxSteps: jsonObj["maxSteps"].getInt,
    mapSize: (jsonObj["mapSize"][0].getInt, jsonObj["mapSize"][1].getInt),
    fileName: fileName
  )
  
  if "tribalConfig" in jsonObj:
    replay.tribalConfig = jsonObj["tribalConfig"]
  
  # Load objects and agents
  for obj in jsonObj["objects"]:
    let entity = TribalEntity(
      id: obj["id"].getInt,
      typeId: obj["typeId"].getInt,
      isAgent: obj.hasKey("agentId")
    )
    
    # Expand location data - handle both array and individual coordinate formats
    if obj.hasKey("location"):
      if obj["location"].kind == JArray and obj["location"].len > 0:
        if obj["location"][0].kind == JObject:
          # Handle IVec2 objects directly
          var location: seq[IVec2]
          for locObj in obj["location"]:
            if locObj.hasKey("x") and locObj.hasKey("y"):
              location.add(ivec2(locObj["x"].getInt, locObj["y"].getInt))
            else:
              location.add(ivec2(0, 0))
          entity.location = location
        else:
          # Handle array of coordinate arrays [[x,y], [x,y], ...]
          var location: seq[IVec2]
          for locArray in obj["location"]:
            if locArray.kind == JArray and locArray.len >= 2:
              location.add(ivec2(locArray[0].getInt, locArray[1].getInt))
            else:
              location.add(ivec2(0, 0))
          entity.location = location
      else:
        entity.location = @[ivec2(0, 0)]
    else:
      entity.location = @[ivec2(0, 0)]
    
    # Load other fields directly from arrays
    if obj.hasKey("orientation") and obj["orientation"].kind == JArray:
      for orientVal in obj["orientation"]:
        entity.orientation.add(orientVal.getInt)
    else:
      entity.orientation = @[0]
    
    if entity.isAgent:
      entity.agentId = obj["agentId"].getInt
      if obj.hasKey("groupId"):
        entity.groupId = obj["groupId"].getInt
      # Load agent-specific arrays
      if obj.hasKey("actionId") and obj["actionId"].kind == JArray:
        for val in obj["actionId"]:
          entity.actionId.add(val.getInt)
      
      if obj.hasKey("actionParameter") and obj["actionParameter"].kind == JArray:
        for val in obj["actionParameter"]:
          entity.actionParameter.add(val.getInt)
      
      if obj.hasKey("actionSuccess") and obj["actionSuccess"].kind == JArray:
        for val in obj["actionSuccess"]:
          entity.actionSuccess.add(val.getBool)
      
      if obj.hasKey("currentReward") and obj["currentReward"].kind == JArray:
        for val in obj["currentReward"]:
          entity.currentReward.add(val.getFloat.float32)
      
      if obj.hasKey("totalReward") and obj["totalReward"].kind == JArray:
        for val in obj["totalReward"]:
          entity.totalReward.add(val.getFloat.float32)
      
      if obj.hasKey("isFrozen") and obj["isFrozen"].kind == JArray:
        for val in obj["isFrozen"]:
          entity.isFrozen.add(val.getBool)
      
      if obj.hasKey("frozenProgress") and obj["frozenProgress"].kind == JArray:
        for val in obj["frozenProgress"]:
          entity.frozenProgress.add(val.getInt)
      
      # Load inventory data
      if obj.hasKey("inventory") and obj["inventory"].kind == JArray:
        for invStep in obj["inventory"]:
          if invStep.kind == JArray:
            var itemAmounts: seq[TribalItemAmount]
            for invItem in invStep:
              if invItem.kind == JArray and invItem.len >= 2:
                itemAmounts.add(TribalItemAmount(
                  itemId: invItem[0].getInt, 
                  count: invItem[1].getInt
                ))
            entity.inventory.add(itemAmounts)
      
      replay.agents.add(entity)
    
    # Load building-specific data
    if obj.hasKey("resourcesAvailable") and obj["resourcesAvailable"].kind == JArray:
      for val in obj["resourcesAvailable"]:
        entity.resourcesAvailable.add(val.getInt)
    
    if obj.hasKey("cooldownProgress") and obj["cooldownProgress"].kind == JArray:
      for val in obj["cooldownProgress"]:
        entity.cooldownProgress.add(val.getInt)
    
    if obj.hasKey("hearts") and obj["hearts"].kind == JArray:
      for val in obj["hearts"]:
        entity.hearts.add(val.getInt)
    
    replay.objects.add(entity)
  
  return replay

proc loadTribalReplay*(fileName: string): TribalReplay =
  ## Load a tribal replay from file
  let data = readFile(fileName)
  return loadTribalReplay(data, fileName)

proc getEntityAtStep*(replay: TribalReplay, entityId: int, step: int): TribalEntity =
  ## Get entity state at a specific step (for playback)
  if entityId >= replay.objects.len or step < 0:
    return nil
    
  let entity = replay.objects[entityId]
  result = TribalEntity(
    id: entity.id,
    typeId: entity.typeId,
    isAgent: entity.isAgent
  )
  
  # Get location at step
  if step < entity.location.len:
    result.location = @[entity.location[step]]
  elif entity.location.len > 0:
    result.location = @[entity.location[^1]]
  else:
    result.location = @[ivec2(0, 0)]
  
  # Get orientation at step
  if step < entity.orientation.len:
    result.orientation = @[entity.orientation[step]]
  elif entity.orientation.len > 0:
    result.orientation = @[entity.orientation[^1]]
  else:
    result.orientation = @[0]
  
  if entity.isAgent:
    result.agentId = entity.agentId
    result.groupId = entity.groupId
    
    # Get agent-specific data at step
    if step < entity.actionId.len:
      result.actionId = @[entity.actionId[step]]
    if step < entity.currentReward.len:
      result.currentReward = @[entity.currentReward[step]]
    if step < entity.totalReward.len:
      result.totalReward = @[entity.totalReward[step]]
    if step < entity.isFrozen.len:
      result.isFrozen = @[entity.isFrozen[step]]
    if step < entity.inventory.len:
      result.inventory = @[entity.inventory[step]]
  
  # Get building-specific data at step
  if entity.resourcesAvailable.len > 0:
    if step < entity.resourcesAvailable.len:
      result.resourcesAvailable = @[entity.resourcesAvailable[step]]
    else:
      result.resourcesAvailable = @[entity.resourcesAvailable[^1]]
      
  if entity.cooldownProgress.len > 0:
    if step < entity.cooldownProgress.len:
      result.cooldownProgress = @[entity.cooldownProgress[step]]
    else:
      result.cooldownProgress = @[entity.cooldownProgress[^1]]
      
  if entity.hearts.len > 0:
    if step < entity.hearts.len:
      result.hearts = @[entity.hearts[step]]
    else:
      result.hearts = @[entity.hearts[^1]]

proc renderReplayStep*(replay: TribalReplay, step: int): string =
  ## Render a text representation of a replay step (for debugging)
  result = fmt"=== Step {step} ===\n"
  
  for entityId, entity in replay.objects:
    let stepEntity = replay.getEntityAtStep(entityId, step)
    if stepEntity == nil:
      continue
      
    let thingType = if stepEntity.typeId < replay.typeNames.len:
      replay.typeNames[stepEntity.typeId]
    else:
      "Unknown"
    
    if stepEntity.location.len > 0:
      let pos = stepEntity.location[0]
      result.add fmt"{thingType}[{stepEntity.id}] at ({pos.x}, {pos.y})"
      
      if stepEntity.isAgent:
        result.add fmt" Agent[{stepEntity.agentId}]"
        if stepEntity.currentReward.len > 0:
          result.add fmt" Reward: {stepEntity.currentReward[0]:.3f}"
        if stepEntity.isFrozen.len > 0 and stepEntity.isFrozen[0]:
          result.add " [FROZEN]"
      
      result.add "\n"