import
  std/[json, os, osproc, strformat, strutils, sequtils, times],
  zippy

# Required top-level keys for replay version 2.
const RequiredKeys = ["version", "num_agents", "max_steps", "map_size", "action_names",
  "item_names", "type_names", "objects"]

# Optional new top-level keys for replay version 2.
const OptionalKeys = ["file_name", "group_names", "reward_sharing_matrix", "mg_config"]

proc requireFields(obj: JsonNode, fields: openArray[string], objName: string) =
  ## Assert that all required fields are present.
  var missing: seq[string]
  for field in fields:
    if field notin obj:
      missing.add(field)
  if missing.len > 0:
    raise newException(ValueError, &"{objName} missing required fields: {missing}")

proc validateType(value: JsonNode, expectedType: string, fieldName: string) =
  ## Validate that value has the expected type.
  let actualType = case value.kind
    of JInt: "int"
    of JFloat: "float"
    of JString: "string"
    of JBool: "bool"
    of JArray: "array"
    of JObject: "object"
    of JNull: "null"

  if actualType != expectedType:
    raise newException(ValueError, &"'{fieldName}' must be {expectedType}, got {actualType}")

proc validatePositiveInt(value: JsonNode, fieldName: string) =
  ## Validate that value is a positive integer.
  validateType(value, "int", fieldName)
  let val = value.getInt()
  if val <= 0:
    raise newException(ValueError, &"'{fieldName}' must be positive, got {val}")

proc validateNonNegativeNumber(value: JsonNode, fieldName: string) =
  ## Validate that value is a non-negative number.
  case value.kind
  of JInt:
    let val = value.getInt()
    if val < 0:
      raise newException(ValueError, &"'{fieldName}' must be non-negative, got {val}")
  of JFloat:
    let val = value.getFloat()
    if val < 0:
      raise newException(ValueError, &"'{fieldName}' must be non-negative, got {val}")
  else:
    raise newException(ValueError, &"'{fieldName}' must be a number, got {value.kind}")

proc validateStringList(lst: JsonNode, fieldName: string, allowEmptyStrings: bool = false) =
  ## Validate that value is a list of strings.
  validateType(lst, "array", fieldName)
  if lst.len == 0:
    raise newException(ValueError, &"'{fieldName}' must not be empty")

  var invalidEntries: seq[string]
  for i, value in lst.getElems():
    if value.kind != JString:
      let valStr = $value
      invalidEntries.add(&"index {i}: expected string, got {value.kind} ({valStr})")
    elif not allowEmptyStrings and value.getStr().len == 0:
      invalidEntries.add(&"index {i}: empty string")

  if invalidEntries.len > 0:
    let requirement = if allowEmptyStrings: "strings" else: "non-empty strings"
    let joinedEntries = invalidEntries.join(", ")
    raise newException(AssertionError, &"'{fieldName}' must contain {requirement}; invalid entries: {joinedEntries}")

proc validateStaticValue(value: JsonNode, expectedType: string, fieldName: string) =
  ## Validate that value is a static value of the expected type (never a time series).
  validateType(value, expectedType, fieldName)

proc validateTimeSeries(data: JsonNode, fieldName: string, expectedType: string) =
  ## Validate time series values: either single values (never changed) or arrays of [step, value] pairs.

  # Check if it's a single value (field never changes)
  case expectedType
  of "int":
    if data.kind == JInt:
      return
  of "float":
    if data.kind == JFloat or data.kind == JInt:
      return
  of "bool":
    if data.kind == JBool:
      return
  else:
    discard

  # Check if it's a time series array (field changes on specific steps)
  if data.kind == JArray:
    if data.len == 0:
      return
    # Validate time series of [step, value] pairs
    for item in data.getElems():
      if item.kind != JArray or item.len != 2:
        raise newException(AssertionError, &"'{fieldName}' time series items must be [step, value] pairs")

      let step = item[0]
      let value = item[1]

      if step.kind != JInt or step.getInt() < 0:
        raise newException(ValueError, &"'{fieldName}' time series step must be non-negative")

      case expectedType
      of "int":
        if value.kind != JInt:
          raise newException(ValueError, &"'{fieldName}' time series value must be int")
      of "float":
        if value.kind notin {JInt, JFloat}:
          raise newException(ValueError, &"'{fieldName}' time series value must be number")
      of "bool":
        if value.kind != JBool:
          raise newException(ValueError, &"'{fieldName}' time series value must be bool")
      else:
        discard

    # First entry should be step 0
    if data.len > 0 and data[0][0].getInt() != 0:
      raise newException(ValueError, &"'{fieldName}' time series must start with step 0")
    return

  # Neither single value nor valid time series
  raise newException(AssertionError, &"'{fieldName}' must be {expectedType} or time series of [step, {expectedType}] pairs")

proc validateInventoryList(inventoryList: JsonNode, fieldName: string) =
  ## Validate a single inventory list: list of [item_id, amount] pairs.
  validateType(inventoryList, "array", fieldName)

  for pair in inventoryList.getElems():
    if pair.kind != JArray or pair.len != 2:
      raise newException(ValueError, &"'{fieldName}' must contain [item_id, amount] pairs")

    let itemId = pair[0]
    let amount = pair[1]

    if itemId.kind != JInt or itemId.getInt() < 0:
      raise newException(ValueError, &"'{fieldName}' item_id must be non-negative integer")

    if amount.kind notin {JInt, JFloat} or amount.getFloat() < 0:
      raise newException(ValueError, &"'{fieldName}' amount must be non-negative number")

proc validateInventoryFormat(inventory: JsonNode, fieldName: string) =
  ## Validate inventory format: single inventory list or time series of [step, inventory_list] pairs.
  if inventory.kind == JNull:
    return

  validateType(inventory, "array", fieldName)
  if inventory.len == 0:
    return

  # Check if it's a single inventory list (never changed during replay)
  # Single inventory format: [[item_id, amount], [item_id, amount], ...]
  var isSingleInventory = true
  for item in inventory.getElems():
    if item.kind != JArray or item.len != 2:
      isSingleInventory = false
      break
    let itemId = item[0]
    let amount = item[1]
    if itemId.kind != JInt or (amount.kind notin {JInt, JFloat}):
      isSingleInventory = false
      break

  if isSingleInventory:
    validateInventoryList(inventory, fieldName)
    return

  # Check if it's a time series format: [[step, inventory_list], ...]
  for item in inventory.getElems():
    if item.kind != JArray or item.len != 2:
      raise newException(AssertionError, &"'{fieldName}' time series items must be [step, inventory_list] pairs")

    let step = item[0]
    let inventoryList = item[1]

    if step.kind != JInt or step.getInt() < 0:
      raise newException(ValueError, &"'{fieldName}' time series step must be non-negative")

    validateInventoryList(inventoryList, fieldName)

proc validateLocation(location: JsonNode, objName: string) =
  ## Validate location field format: single [x, y] or time series of [step, [x, y]] pairs.
  let fieldName = &"{objName}.location"

  # Check if it's a single location (never changed during replay)
  # also make sure it's not a time series array with only 2 elements
  if location.kind == JArray and location.len == 2:
    var allNumbers = true
    for coord in location.getElems():
      if coord.kind notin {JInt, JFloat}:
        allNumbers = false
        break
    if allNumbers:
      return

  # Check if it's a time series array (location changed during replay)
  validateType(location, "array", fieldName)
  if location.len == 0:
    raise newException(ValueError, &"{fieldName} must have at least one entry")

  # Validate time series of [step, [x, y]] pairs
  for stepData in location.getElems():
    if stepData.kind != JArray or stepData.len != 2:
      raise newException(AssertionError, &"{fieldName} items must be [step, [x, y]] pairs")

    let step = stepData[0]
    let coords = stepData[1]

    if step.kind != JInt or step.getInt() < 0:
      raise newException(ValueError, &"{fieldName} step must be non-negative")

    if coords.kind != JArray or coords.len != 2:
      raise newException(ValueError, &"{fieldName} coordinates must be [x, y]")

    for i, coord in coords.getElems():
      if coord.kind notin {JInt, JFloat}:
        raise newException(ValueError, &"{fieldName} coord[{i}] must be a number")

  # Must start with step 0
  if location[0][0].getInt() != 0:
    raise newException(ValueError, &"{fieldName} must start with step 0")

proc validateActionIdRange(actionIds: JsonNode, objName: string, actionNames: seq[string]) =
  ## Validate that action_id values are within the valid range.
  # Handle single value case
  if actionIds.kind == JInt:
    let actionId = actionIds.getInt()
    if actionId < 0 or actionId >= actionNames.len:
      raise newException(ValueError, &"{objName}.action_id {actionId} out of range")
    return

  # Handle time series case
  if actionIds.kind == JArray:
    for stepData in actionIds.getElems():
      if stepData.kind == JArray and stepData.len == 2:
        let actionId = stepData[1].getInt()
        if actionId < 0 or actionId >= actionNames.len:
          raise newException(ValueError, &"{objName}.action_id {actionId} out of range")

proc validateAgentFields(obj: JsonNode, objName: string, replayData: JsonNode) =
  ## Validate all agent-specific fields.
  let agentFields = [
    "agent_id", "is_agent", "vision_size", "action_id", "action_param", "action_success",
    "current_reward", "total_reward", "freeze_remaining", "is_frozen", "freeze_duration", "group_id"
  ]
  requireFields(obj, agentFields, objName)

  # Validate static agent fields.
  let agentId = obj["agent_id"].getInt()
  validateStaticValue(obj["agent_id"], "int", &"{objName}.agent_id")
  validateNonNegativeNumber(obj["agent_id"], &"{objName}.agent_id")
  if agentId >= replayData["num_agents"].getInt():
    raise newException(ValueError, &"{objName}.agent_id {agentId} out of range")

  validateStaticValue(obj["is_agent"], "bool", &"{objName}.is_agent")
  if not obj["is_agent"].getBool():
    raise newException(ValueError, &"{objName}.is_agent must be True")

  validateStaticValue(obj["vision_size"], "int", &"{objName}.vision_size")
  validatePositiveInt(obj["vision_size"], &"{objName}.vision_size")

  validateStaticValue(obj["group_id"], "int", &"{objName}.group_id")
  validateNonNegativeNumber(obj["group_id"], &"{objName}.group_id")

  # Validate dynamic agent fields (always time series).
  validateTimeSeries(obj["action_id"], &"{objName}.action_id", "int")
  validateTimeSeries(obj["action_param"], &"{objName}.action_param", "int")
  validateTimeSeries(obj["action_success"], &"{objName}.action_success", "bool")
  validateTimeSeries(obj["current_reward"], &"{objName}.current_reward", "float")
  validateTimeSeries(obj["total_reward"], &"{objName}.total_reward", "float")
  validateTimeSeries(obj["freeze_remaining"], &"{objName}.freeze_remaining", "float")
  validateTimeSeries(obj["is_frozen"], &"{objName}.is_frozen", "bool")
  validateTimeSeries(obj["freeze_duration"], &"{objName}.freeze_duration", "float")

  # Validate action_id values are in range.
  validateActionIdRange(obj["action_id"], objName, replayData["action_names"].to(seq[string]))

proc validateBuildingFields(obj: JsonNode, objName: string) =
  ## Validate all building-specific fields.
  let buildingFields = [
    "input_resources", "output_resources", "output_limit", "conversion_remaining",
    "is_converting", "conversion_duration", "cooldown_remaining", "is_cooling_down",
    "cooldown_duration"
  ]
  requireFields(obj, buildingFields, objName)

  # Validate static building fields.
  validateStaticValue(obj["output_limit"], "float", &"{objName}.output_limit")
  validateNonNegativeNumber(obj["output_limit"], &"{objName}.output_limit")

  validateStaticValue(obj["conversion_duration"], "float", &"{objName}.conversion_duration")
  validateNonNegativeNumber(obj["conversion_duration"], &"{objName}.conversion_duration")

  validateStaticValue(obj["cooldown_duration"], "float", &"{objName}.cooldown_duration")
  validateNonNegativeNumber(obj["cooldown_duration"], &"{objName}.cooldown_duration")

  # Validate dynamic building fields (always time series).
  validateInventoryFormat(obj["input_resources"], &"{objName}.input_resources")
  validateInventoryFormat(obj["output_resources"], &"{objName}.output_resources")
  validateTimeSeries(obj["conversion_remaining"], &"{objName}.conversion_remaining", "float")
  validateTimeSeries(obj["is_converting"], &"{objName}.is_converting", "bool")
  validateTimeSeries(obj["cooldown_remaining"], &"{objName}.cooldown_remaining", "float")
  validateTimeSeries(obj["is_cooling_down"], &"{objName}.is_cooling_down", "bool")

proc validateObject(obj: JsonNode, objIndex: int, replayData: JsonNode) =
  ## Validate a single object in the replay.
  let objName = &"Object {objIndex}"

  # All objects have these required fields.
  let requiredFields = [
    "id", "type_name", "location", "orientation", "inventory", "inventory_max", "color"
  ]
  requireFields(obj, requiredFields, objName)

  # Validate static fields.
  validateStaticValue(obj["id"], "int", &"{objName}.id")
  validatePositiveInt(obj["id"], &"{objName}.id")

  let typeName = obj["type_name"].getStr()
  validateStaticValue(obj["type_name"], "string", &"{objName}.type_name")
  let typeNames = replayData["type_names"].to(seq[string])
  if typeName notin typeNames:
    raise newException(ValueError, &"{objName}.type_name '{typeName}' not in type_names list")

  # Validate dynamic fields (always time series).
  validateLocation(obj["location"], objName)
  validateTimeSeries(obj["orientation"], &"{objName}.orientation", "int")
  validateInventoryFormat(obj["inventory"], &"{objName}.inventory")
  validateTimeSeries(obj["inventory_max"], &"{objName}.inventory_max", "int")
  validateTimeSeries(obj["color"], &"{objName}.color", "int")

  # Validate specific object types.
  if obj.getOrDefault("is_agent").getBool() or "agent_id" in obj:
    validateAgentFields(obj, objName, replayData)
  elif "input_resources" in obj:
    validateBuildingFields(obj, objName)

proc validateReplaySchema(data: JsonNode) =
  ## Validate that replay data matches the version 2 schema specification.
  # Check required keys and absence of unexpected keys.
  let dataKeys = toSeq(keys(data))
  var missing: seq[string]
  for key in RequiredKeys:
    if key notin dataKeys:
      missing.add(key)

  var allowedKeys: seq[string]
  allowedKeys.add(RequiredKeys)
  allowedKeys.add(OptionalKeys)

  var unexpected: seq[string]
  for key in dataKeys:
    if key notin allowedKeys:
      unexpected.add(key)

  if missing.len > 0:
    let joinedMissing = missing.join(", ")
    raise newException(ValueError, &"Missing required keys: {joinedMissing}")
  if unexpected.len > 0:
    let joinedUnexpected = unexpected.join(", ")
    raise newException(ValueError, &"Unexpected keys present: {joinedUnexpected}")

  # Top-level field validation.
  let version = data["version"].getInt()
  if version != 2:
    raise newException(ValueError, &"'version' must equal 2, got {version}")

  validatePositiveInt(data["num_agents"], "num_agents")
  validateNonNegativeNumber(data["max_steps"], "max_steps")

  # Validate map_size.
  let mapSize = data["map_size"]
  validateType(mapSize, "array", "map_size")
  if mapSize.len != 2:
    raise newException(ValueError, "'map_size' must have exactly 2 dimensions")
  for i in 0..<mapSize.len:
    validatePositiveInt(mapSize[i], &"map_size[{i}]")

  # Required string lists.
  for field in ["action_names", "item_names", "type_names"]:
    validateStringList(data[field], field, allowEmptyStrings = true)

  # Optional file_name validation.
  if "file_name" in data:
    let fileName = data["file_name"]
    validateType(fileName, "string", "file_name")
    if fileName.getStr().len == 0:
      raise newException(ValueError, "'file_name' must be non-empty")

  # Optional string lists.
  if "group_names" in data:
    validateStringList(data["group_names"], "group_names", allowEmptyStrings = true)

  # Optional reward sharing matrix.
  if "reward_sharing_matrix" in data:
    let matrix = data["reward_sharing_matrix"]
    validateType(matrix, "array", "reward_sharing_matrix")
    let numAgents = data["num_agents"].getInt()
    if matrix.len != numAgents:
      raise newException(ValueError, &"'reward_sharing_matrix' must have {numAgents} rows")
    for i, row in matrix.getElems():
      validateType(row, "array", &"reward_sharing_matrix[{i}]")
      if row.len != numAgents:
        raise newException(ValueError, &"'reward_sharing_matrix[{i}]' must have {numAgents} columns")
      for v in row.getElems():
        if v.kind notin {JInt, JFloat}:
          raise newException(ValueError, &"'reward_sharing_matrix[{i}]' must contain numbers")

  # Objects validation.
  let objects = data["objects"]
  validateType(objects, "array", "objects")
  if objects.len == 0:
    raise newException(ValueError, "'objects' must not be empty")
  for obj in objects.getElems():
    if obj.kind != JObject:
      raise newException(ValueError, "'objects' must contain objects")

  # Validate each object and count agents.
  var agentCount = 0
  for i, obj in objects.getElems():
    validateObject(obj, i, data)
    if obj.getOrDefault("is_agent").getBool() or "agent_id" in obj:
      agentCount += 1

  let expectedAgents = data["num_agents"].getInt()
  if agentCount != expectedAgents:
    raise newException(ValueError, &"Expected {expectedAgents} agents, found {agentCount}")

proc loadReplay(path: string): JsonNode =
  ## Load and decompress a .json.z replay file.
  if not path.endsWith(".json.z"):
    raise newException(ValueError, "Replay file name must end with '.json.z'")

  let compressedData = readFile(path)
  var decompressed: string
  try:
    decompressed = zippy.uncompress(compressedData)
  except:
    raise newException(ValueError, "Failed to decompress replay file")

  try:
    result = parseJson(decompressed)
  except:
    raise newException(ValueError, "Invalid JSON in replay file")

proc makeValidReplay(fileName: string = "sample.json.z"): JsonNode =
  ## Create a minimal valid replay dict per the spec.
  result = %*{
    "version": 2,
    "num_agents": 2,
    "max_steps": 100,
    "map_size": [10, 10],
    "file_name": "test replay file format",
    "type_names": ["agent", "resource"],
    "action_names": ["move", "collect"],
    "item_names": ["wood", "stone"],
    "group_names": ["group1", "group2"],
    "reward_sharing_matrix": [[1, 0], [0, 1]],
    "objects": [
      {
        "id": 1,
        "type_name": "agent",
        "agent_id": 0,
        "is_agent": true,
        "vision_size": 11,
        "group_id": 0,
        # Time series fields (some single values, some arrays for testing)
        "location": [[0, [5, 5]], [1, [6, 5]], [2, [7, 5]]],
        "action_id": 0,
        "action_param": 0,
        "action_success": true,
        "current_reward": 0.0,
        "total_reward": 0.0,
        "freeze_remaining": 0,
        "is_frozen": false,
        "freeze_duration": 0,
        "orientation": 0,
        "inventory": [],
        "inventory_max": 10,
        "color": 0,
      },
      {
        "id": 2,
        "type_name": "agent",
        "agent_id": 1,
        "is_agent": true,
        "vision_size": 11,
        "group_id": 0,
        # Time series fields (mix of single values and arrays for testing)
        "location": [[0, [3, 3]], [5, [4, 3]]],
        "action_id": [[0, 1], [10, 0]],
        "action_param": 0,
        "action_success": [[0, false], [10, true]],
        "current_reward": 1.5,
        "total_reward": [[0, 0.0], [10, 1.5]],
        "freeze_remaining": 0,
        "is_frozen": false,
        "freeze_duration": 0,
        "orientation": 1,
        "inventory": [[0, []], [20, [[0, 2], [1, 1]]]],
        "inventory_max": 10,
        "color": 1,
      },
    ]
  }

# Test cases - validate replay schema and generated replays

block schema_validation:
  block valid_replay:
    let replay = makeValidReplay()
    validateReplaySchema(replay)
    echo "✓ Valid replay schema passes validation"

  block invalid_version:
    var replay = makeValidReplay()
    replay["version"] = %*1
    try:
      validateReplaySchema(replay)
      doAssert false, "Should have failed validation"
    except ValueError as e:
      doAssert "'version' must equal 2" in e.msg, &"Unexpected error: {e.msg}"
    echo "✓ Invalid version properly rejected"

  block invalid_num_agents:
    var replay = makeValidReplay()
    replay["num_agents"] = %*(-1)
    try:
      validateReplaySchema(replay)
      doAssert false, "Should have failed validation"
    except ValueError as e:
      doAssert "'num_agents' must be positive" in e.msg, &"Unexpected error: {e.msg}"
    echo "✓ Invalid num_agents properly rejected"

  block invalid_map_size:
    var replay = makeValidReplay()
    replay["map_size"] = %*[0, 5]
    try:
      validateReplaySchema(replay)
      doAssert false, "Should have failed validation"
    except ValueError as e:
      doAssert "'map_size[0]' must be positive" in e.msg, &"Unexpected error: {e.msg}"
    echo "✓ Invalid map_size properly rejected"

block generated_replay_test:
  # Generate a replay using the CI setup and validate it against the strict schema.

  # Create temporary directory
  let tmpDir = getTempDir() / "metta_replay_test_" & $getTime().toUnix()
  createDir(tmpDir)
  defer: removeDir(tmpDir)

  # Generate a replay using the CI configuration
  let projectRoot = parentDir(parentDir(parentDir(parentDir(parentDir(parentDir(currentSourcePath()))))))
  let cmd = &"cd {projectRoot} && uv run --no-sync tools/run.py ci.replay_null replay_dir={tmpDir}"
  echo &"Running replay generation: {cmd}"

  let exitCode = execCmd(cmd)
  if exitCode != 0:
    raise newException(AssertionError, &"Replay generation failed with exit code {exitCode}")

  # Find generated replay files
  var replayFiles: seq[string]
  for file in walkDirRec(tmpDir):
    if file.endsWith(".json.z"):
      replayFiles.add(file)

  if replayFiles.len == 0:
    raise newException(AssertionError, &"No replay files were generated in {tmpDir}")

  # Should have exactly one replay file
  if replayFiles.len != 1:
    raise newException(AssertionError, &"Expected exactly 1 replay file, found {replayFiles.len}: {replayFiles}")

  # Validate the replay file
  let replayPath = replayFiles[0]
  let loadedReplay = loadReplay(replayPath)
  validateReplaySchema(loadedReplay)

  echo &"✓ Successfully generated and validated replay: {extractFilename(replayPath)}"
