import
  std/[json, strformat, strutils, sequtils]

type
  ValidationIssue* = object
    message*: string
    field*: string

# Important: Never use obj["key"] unless we previously checked that the key exists.
# validator functions must not throw exceptions, they must append issues.

# Required top-level keys for replay version 3.
const RequiredKeys = ["version", "num_agents", "max_steps", "map_size", "action_names",
  "item_names", "type_names", "objects"]

# Optional new top-level keys for replay version 3.
const OptionalKeys = ["file_name", "group_names", "reward_sharing_matrix", "mg_config"]

proc requireFields*(obj: JsonNode, fields: openArray[string], objName: string, issues: var seq[ValidationIssue]) =
  ## Assert that all required fields are present.
  var missing: seq[string]
  for field in fields:
    if field notin obj:
      missing.add(field)
  if missing.len > 0:
    let missingStr = missing.join(", ")
    issues.add(ValidationIssue(
      message: &"{objName} missing required fields: {missingStr}",
      field: objName
    ))

proc validateType*(obj: JsonNode, key: string, expectedType: string, fieldName: string, issues: var seq[ValidationIssue]) =
  ## Validate that obj[key] has the expected type, adding an issue if the key is missing.
  if key notin obj:
    issues.add(ValidationIssue(
      message: &"'{fieldName}' is missing (required)",
      field: fieldName
    ))
    return

  let value = obj[key]
  let actualType = case value.kind
    of JInt: "int"
    of JFloat: "float"
    of JString: "string"
    of JBool: "bool"
    of JArray: "array"
    of JObject: "object"
    of JNull: "null"

  if actualType != expectedType:
    issues.add(ValidationIssue(
      message: &"'{fieldName}' must be {expectedType}, got {actualType}",
      field: fieldName
    ))

proc validateTypeValue*(value: JsonNode, expectedType: string, fieldName: string, issues: var seq[ValidationIssue]) =
  ## Validate that value has the expected type (for direct JsonNode values, no key checking).
  let actualType = case value.kind
    of JInt: "int"
    of JFloat: "float"
    of JString: "string"
    of JBool: "bool"
    of JArray: "array"
    of JObject: "object"
    of JNull: "null"

  if actualType != expectedType:
    issues.add(ValidationIssue(
      message: &"'{fieldName}' must be {expectedType}, got {actualType}",
      field: fieldName
    ))

proc validatePositiveInt*(obj: JsonNode, key: string, fieldName: string, issues: var seq[ValidationIssue]) =
  ## Validate that obj[key] is a positive integer, adding an issue if the key is missing.
  if key notin obj:
    issues.add(ValidationIssue(
      message: &"'{fieldName}' is missing (required)",
      field: fieldName
    ))
    return

  let value = obj[key]
  validateTypeValue(value, "int", fieldName, issues)
  if value.kind == JInt:
    let val = value.getInt()
    if val <= 0:
      issues.add(ValidationIssue(
        message: &"'{fieldName}' must be positive, got {val}",
        field: fieldName
      ))

proc validatePositiveIntValue*(value: JsonNode, fieldName: string, issues: var seq[ValidationIssue]) =
  ## Validate that value is a positive integer (for direct JsonNode values).
  validateTypeValue(value, "int", fieldName, issues)
  if value.kind == JInt:
    let val = value.getInt()
    if val <= 0:
      issues.add(ValidationIssue(
        message: &"'{fieldName}' must be positive, got {val}",
        field: fieldName
      ))

proc validateNonNegativeNumber*(obj: JsonNode, key: string, fieldName: string, issues: var seq[ValidationIssue]) =
  ## Validate that obj[key] is a non-negative number, adding an issue if the key is missing.
  if key notin obj:
    issues.add(ValidationIssue(
      message: &"'{fieldName}' is missing (required)",
      field: fieldName
    ))
    return

  let value = obj[key]
  case value.kind
  of JInt:
    let val = value.getInt()
    if val < 0:
      issues.add(ValidationIssue(
        message: &"'{fieldName}' must be non-negative, got {val}",
        field: fieldName
      ))
  of JFloat:
    let val = value.getFloat()
    if val < 0:
      issues.add(ValidationIssue(
        message: &"'{fieldName}' must be non-negative, got {val}",
        field: fieldName
      ))
  else:
    issues.add(ValidationIssue(
      message: &"'{fieldName}' must be a number, got {value.kind}",
      field: fieldName
    ))

proc validateStringList*(obj: JsonNode, key: string, fieldName: string, issues: var seq[ValidationIssue], allowEmptyStrings: bool = false) =
  ## Validate that obj[key] is a list of strings, adding an issue if the key is missing.
  if key notin obj:
    issues.add(ValidationIssue(
      message: &"'{fieldName}' is missing (required)",
      field: fieldName
    ))
    return

  let lst = obj[key]
  validateTypeValue(lst, "array", fieldName, issues)
  if lst.kind == JArray:
    if lst.len == 0:
      issues.add(ValidationIssue(
        message: &"'{fieldName}' must not be empty",
        field: fieldName
      ))

    var invalidEntries: seq[string]
    for i, value in lst.getElems():
      if value.kind != JString:
        let valStr = $value
        invalidEntries.add(&"index {i}: expected string, got {value.kind} ({valStr})")
      elif not allowEmptyStrings and value.getStr().len == 0:
        invalidEntries.add(&"index {i}: empty string")

    if invalidEntries.len > 0:
      let requirement = if allowEmptyStrings: "strings" else: "non-empty strings"
      let invalidEntriesStr = invalidEntries.join(", ")
      issues.add(ValidationIssue(
        message: &"'{fieldName}' must contain {requirement}; invalid entries: {invalidEntriesStr}",
        field: fieldName
      ))

proc validateStaticValue*(obj: JsonNode, key: string, expectedType: string, fieldName: string, issues: var seq[ValidationIssue]) =
  ## Validate that obj[key] has the expected static type, adding an issue if the key is missing.
  if key notin obj:
    issues.add(ValidationIssue(
      message: &"'{fieldName}' is missing (required)",
      field: fieldName
    ))
    return

  let value = obj[key]
  if value.kind == JArray:
    issues.add(ValidationIssue(
      message: &"'{fieldName}' must be a single value, not an array",
      field: fieldName
    ))
    return
  validateType(obj, key, expectedType, fieldName, issues)

proc validateTimeSeries*(obj: JsonNode, key: string, fieldName: string, expectedType: string, issues: var seq[ValidationIssue]) =
  ## Validate time series values: either single values (never changed) or arrays of [step, value] pairs.
  if key notin obj:
    issues.add(ValidationIssue(
      message: &"'{fieldName}' is missing (required)",
      field: fieldName
    ))
    return

  let data = obj[key]

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
        issues.add(ValidationIssue(
          message: &"'{fieldName}' time series items must be [step, value] pairs",
          field: fieldName
        ))
        return

      let step = item[0]
      let value = item[1]

      if step.kind != JInt or step.getInt() < 0:
        issues.add(ValidationIssue(
          message: &"'{fieldName}' time series step must be non-negative",
          field: fieldName
        ))

      case expectedType
      of "int":
        if value.kind != JInt:
          issues.add(ValidationIssue(
            message: &"'{fieldName}' time series value must be int",
            field: fieldName
          ))
      of "float":
        if value.kind notin {JInt, JFloat}:
          issues.add(ValidationIssue(
            message: &"'{fieldName}' time series value must be number",
            field: fieldName
          ))
      of "bool":
        if value.kind != JBool:
          issues.add(ValidationIssue(
            message: &"'{fieldName}' time series value must be bool",
            field: fieldName
          ))
      else:
        discard

    # First entry should be step 0
    if data.len > 0 and data[0][0].getInt() != 0:
      issues.add(ValidationIssue(
        message: &"'{fieldName}' time series must start with step 0",
        field: fieldName
      ))
    return

  # Neither single value nor valid time series
  issues.add(ValidationIssue(
    message: &"'{fieldName}' must be {expectedType} or time series of [step, {expectedType}] pairs",
    field: fieldName
  ))

proc validateInventoryFormat*(obj: JsonNode, key: string, fieldName: string, issues: var seq[ValidationIssue]) =
  ## Validate inventory: array of [itemId, count] pairs, or time series of [step, inventory_array] pairs.
  if key notin obj:
    issues.add(ValidationIssue(
      message: &"'{fieldName}' is missing (required)",
      field: fieldName
    ))
    return

  let inventory = obj[key]
  if inventory.kind == JNull:
    return

  validateTypeValue(inventory, "array", fieldName, issues)
  if inventory.kind != JArray:
    return

  if inventory.len == 0:
    return

  # Check if this is a time series format [[step, inventory_array], ...]
  # Time series: first element is [step, inventory_array] where step is int and inventory_array is array
  let firstItem = inventory[0]
  if firstItem.kind == JArray and firstItem.len == 2:
    let step = firstItem[0]
    let inventoryArray = firstItem[1]
    if step.kind == JInt and step.getInt() >= 0 and inventoryArray.kind == JArray:
      # This is time series format: [[step, [[itemId, count], [itemId, count], ...]], ...]
      for item in inventory.getElems():
        if item.kind != JArray or item.len != 2:
          issues.add(ValidationIssue(
            message: &"'{fieldName}' time series items must be [step, inventory_array] pairs",
            field: fieldName
          ))
          continue

        let tsStep = item[0]
        let tsInventory = item[1]

        if tsStep.kind != JInt or tsStep.getInt() < 0:
          issues.add(ValidationIssue(
            message: &"'{fieldName}' time series step must be non-negative integer",
            field: fieldName
          ))

        if tsInventory.kind != JArray:
          issues.add(ValidationIssue(
            message: &"'{fieldName}' inventory must be array of item amounts, got {tsInventory.kind}",
            field: fieldName
          ))
          continue

        # Validate the inventory array contents (compressed [itemId, count] format)
        for itemAmount in tsInventory.getElems():
          if itemAmount.kind != JArray or itemAmount.len != 2:
            issues.add(ValidationIssue(
              message: &"'{fieldName}' item amounts must be [itemId, count] pairs, got {itemAmount}",
              field: fieldName
            ))
            continue

          let itemId = itemAmount[0]
          let count = itemAmount[1]

          if itemId.kind != JInt or itemId.getInt() < 0:
            issues.add(ValidationIssue(
              message: &"'{fieldName}' item IDs must be non-negative integers, got {itemId}",
              field: fieldName
            ))

          if count.kind != JInt or count.getInt() < 0:
            issues.add(ValidationIssue(
              message: &"'{fieldName}' item counts must be non-negative integers, got {count}",
              field: fieldName
            ))
      return

  # Otherwise, treat as single inventory array that does not change over time: [[itemId, count], [itemId, count], ...]
  for itemAmount in inventory.getElems():
    if itemAmount.kind != JArray or itemAmount.len != 2:
      issues.add(ValidationIssue(
        message: &"'{fieldName}' item amounts must be [itemId, count] pairs, got {itemAmount}",
        field: fieldName
      ))
      continue

    let itemId = itemAmount[0]
    let count = itemAmount[1]

    if itemId.kind != JInt or itemId.getInt() < 0:
      issues.add(ValidationIssue(
        message: &"'{fieldName}' item IDs must be non-negative integers, got {itemId}",
        field: fieldName
      ))

    if count.kind != JInt or count.getInt() < 0:
      issues.add(ValidationIssue(
        message: &"'{fieldName}' item counts must be non-negative integers, got {count}",
        field: fieldName
      ))

proc validateLocation*(obj: JsonNode, key: string, fieldName: string, issues: var seq[ValidationIssue]) =
  ## Validate location field format: single [x, y] or time series of [step, [x, y]] pairs.
  if key notin obj:
    issues.add(ValidationIssue(
      message: &"'{fieldName}' is missing (required)",
      field: fieldName
    ))
    return

  let location = obj[key]

  # Check if it's a single location (never changed during replay)
  # also make sure it's not a time series array
  if location.kind == JArray and (location.len == 2 or location.len == 3):
    var allNumbers = true
    for coord in location.getElems():
      if coord.kind notin {JInt, JFloat}:
        allNumbers = false
        break
    if allNumbers:
      return

  # Check if it's a time series array (location changed during replay)
  validateTypeValue(location, "array", fieldName, issues)
  if location.kind == JArray and location.len == 0:
    issues.add(ValidationIssue(
      message: &"{fieldName} must have at least one entry",
      field: fieldName
    ))

  # Validate time series of [step, [x, y]] pairs
  if location.kind == JArray:
    for stepData in location.getElems():
      if stepData.kind != JArray or stepData.len != 2:
        issues.add(ValidationIssue(
          message: &"{fieldName} items must be [step, [x, y]] pairs",
          field: fieldName
        ))
        continue

      let step = stepData[0]
      let coords = stepData[1]

      if step.kind != JInt or step.getInt() < 0:
        issues.add(ValidationIssue(
          message: &"{fieldName} step must be non-negative",
          field: fieldName
        ))

      # allow vec3 so that older replays are not so noisy
      if coords.kind != JArray or (coords.len != 2 and coords.len != 3):
        issues.add(ValidationIssue(
          message: &"{fieldName} coordinates must be [x, y] array, got {coords.kind}",
          field: fieldName
        ))
        continue  # Skip coordinate validation if coords is not an array

      if coords.kind == JArray:
        for i, coord in coords.getElems():
          if coord.kind notin {JInt, JFloat}:
            issues.add(ValidationIssue(
              message: &"{fieldName} coord[{i}] must be a number",
              field: fieldName
            ))

    # Must start with step 0 (only check if we have valid time series data)
    if location.len > 0 and location[0].kind == JArray and location[0].len >= 1 and location[0][0].kind == JInt:
      if location[0][0].getInt() != 0:
        issues.add(ValidationIssue(
          message: &"{fieldName} must start with step 0",
          field: fieldName
        ))

proc validateActionIdRange*(obj: JsonNode, key: string, objName: string, actionNames: seq[string], issues: var seq[ValidationIssue]) =
  ## Validate that action_id values are within the valid range.
  if key notin obj:
    issues.add(ValidationIssue(
      message: &"'{objName}.{key}' is missing (required)",
      field: &"{objName}.{key}"
    ))
    return

  let actionIds = obj[key]
  # Handle single value case
  if actionIds.kind == JInt:
    let actionId = actionIds.getInt()
    if actionId < 0 or actionId >= actionNames.len:
      issues.add(ValidationIssue(
        message: &"{objName}.action_id {actionId} out of range",
        field: &"{objName}.action_id"
      ))
    return

  # Handle time series case
  if actionIds.kind == JArray:
    for stepData in actionIds.getElems():
      if stepData.kind == JArray and stepData.len == 2:
        let actionId = stepData[1].getInt()
        if actionId < 0 or actionId >= actionNames.len:
          issues.add(ValidationIssue(
            message: &"{objName}.action_id {actionId} out of range",
            field: &"{objName}.action_id"
          ))

proc validateAgentFields*(obj: JsonNode, objName: string, replayData: JsonNode, issues: var seq[ValidationIssue]) =
  ## Validate all agent-specific fields.

  let requiredAgentFields = [
    "agent_id", "action_id", "action_success",
    "total_reward", "current_reward", "group_id"
  ]
  requireFields(obj, requiredAgentFields, objName, issues)

  # let optionalAgentFields = [
  #   "frozen", "frozen_time", "frozen_progress", "action_parameter"
  # ]

  # Validate static agent fields.
  validateStaticValue(obj, "agent_id", "int", objName & ".agent_id", issues)
  validateNonNegativeNumber(obj, "agent_id", objName & ".agent_id", issues)
  if "num_agents" in replayData:
    let agentId = obj["agent_id"].getInt()
    if agentId >= replayData["num_agents"].getInt():
      issues.add(ValidationIssue(
        message: &"{objName}.agent_id {agentId} out of range",
        field: objName & ".agent_id"
      ))

  validateStaticValue(obj, "group_id", "int", objName & ".group_id", issues)
  validateNonNegativeNumber(obj, "group_id", objName & ".group_id", issues)

  # Validate dynamic agent fields (always time series).
  validateTimeSeries(obj, "action_id", objName & ".action_id", "int", issues)
  validateTimeSeries(obj, "action_success", objName & ".action_success", "bool", issues)
  validateTimeSeries(obj, "current_reward", objName & ".current_reward", "float", issues)
  validateTimeSeries(obj, "total_reward", objName & ".total_reward", "float", issues)
  # validate optional agent fields
  if "action_param" in obj:
    validateTimeSeries(obj, "action_param", objName & ".action_param", "int", issues)
  if "action_parameter" in obj:
    validateTimeSeries(obj, "action_parameter", objName & ".action_parameter", "int", issues)
  if "frozen" in obj:
    validateTimeSeries(obj, "frozen", objName & ".frozen", "bool", issues)
  if "is_frozen" in obj:
    validateTimeSeries(obj, "is_frozen", objName & ".is_frozen", "bool", issues)
  if "frozen_progress" in obj:
    validateTimeSeries(obj, "frozen_progress", objName & ".frozen_progress", "int", issues)
  if "frozen_time" in obj:
    validateTimeSeries(obj, "frozen_time", objName & ".frozen_time", "int", issues)

  # Validate action_id values are in range.
  if "action_names" in replayData:
    validateActionIdRange(obj, "action_id", objName, replayData["action_names"].to(seq[string]), issues)

proc validateProtocol*(protocol: JsonNode, protocolIndex: int, objName: string, issues: var seq[ValidationIssue]) =
  ## Validate a single protocol within an assembler.
  let protocolName = &"{objName}.protocols[{protocolIndex}]"

  # Protocol must be an object
  validateTypeValue(protocol, "object", protocolName, issues)

  if protocol.kind != JObject:
    return

  # Check required fields
  let requiredFields = ["minAgents", "vibes", "inputs", "outputs", "cooldown"]
  requireFields(protocol, requiredFields, protocolName, issues)

  # Validate field types
  validateType(protocol, "minAgents", "int", protocolName & ".minAgents", issues)
  validateType(protocol, "vibes", "array", protocolName & ".vibes", issues)
  validateType(protocol, "inputs", "array", protocolName & ".inputs", issues)
  validateType(protocol, "outputs", "array", protocolName & ".outputs", issues)
  validateType(protocol, "cooldown", "int", protocolName & ".cooldown", issues)

  # Validate non-negative values
  if "minAgents" in protocol and protocol["minAgents"].kind == JInt and protocol["minAgents"].getInt() < 0:
    issues.add(ValidationIssue(
      message: &"{protocolName}.minAgents must be non-negative",
      field: protocolName & ".minAgents"
    ))

  if "cooldown" in protocol and protocol["cooldown"].kind == JInt and protocol["cooldown"].getInt() < 0:
    issues.add(ValidationIssue(
      message: &"{protocolName}.cooldown must be non-negative",
      field: protocolName & ".cooldown"
    ))

  # Validate vibes array contains integers
  if "vibes" in protocol and protocol["vibes"].kind == JArray:
    for i, vibe in protocol["vibes"].getElems():
      validateTypeValue(vibe, "int", &"{protocolName}.vibes[{i}]", issues)

  # Validate inputs and outputs arrays
  if "inputs" in protocol and protocol["inputs"].kind == JArray:
    for i, itemAmount in protocol["inputs"].getElems():
      if itemAmount.kind == JArray and itemAmount.len == 2:
        let itemId = itemAmount[0]
        let count = itemAmount[1]
        validateTypeValue(itemId, "int", &"{protocolName}.inputs[{i}][0]", issues)
        validateTypeValue(count, "int", &"{protocolName}.inputs[{i}][1]", issues)
        if count.kind == JInt and count.getInt() < 0:
          issues.add(ValidationIssue(
            message: &"{protocolName}.inputs[{i}][1] must be non-negative",
            field: &"{protocolName}.inputs[{i}][1]"
          ))
      else:
        issues.add(ValidationIssue(
          message: &"{protocolName}.inputs[{i}] must be [item_id, count] array",
          field: &"{protocolName}.inputs[{i}]"
        ))

  if "outputs" in protocol and protocol["outputs"].kind == JArray:
    for i, itemAmount in protocol["outputs"].getElems():
      if itemAmount.kind == JArray and itemAmount.len == 2:
        let itemId = itemAmount[0]
        let count = itemAmount[1]
        validateTypeValue(itemId, "int", &"{protocolName}.outputs[{i}][0]", issues)
        validateTypeValue(count, "int", &"{protocolName}.outputs[{i}][1]", issues)
        if count.kind == JInt and count.getInt() < 0:
          issues.add(ValidationIssue(
            message: &"{protocolName}.outputs[{i}][1] must be non-negative",
            field: &"{protocolName}.outputs[{i}][1]"
          ))
      else:
        issues.add(ValidationIssue(
          message: &"{protocolName}.outputs[{i}] must be [item_id, count] array",
          field: &"{protocolName}.outputs[{i}]"
        ))

proc validateAssemblerFields*(obj: JsonNode, objName: string, issues: var seq[ValidationIssue]) =
  ## Validate all assembler-specific fields.
  let assemblerFields = [
    "protocols", "is_clipped", "is_clip_immune", "uses_count", "max_uses", "allow_partial_usage"
  ]
  requireFields(obj, assemblerFields, objName, issues)

  # Validate static assembler fields.
  validateStaticValue(obj, "max_uses", "int", objName & ".max_uses", issues)
  validateNonNegativeNumber(obj, "max_uses", objName & ".max_uses", issues)

  validateStaticValue(obj, "max_uses", "int", objName & ".max_uses", issues)
  validateNonNegativeNumber(obj, "max_uses", objName & ".max_uses", issues)

  validateStaticValue(obj, "allow_partial_usage", "bool", objName & ".allow_partial_usage", issues)

  # Validate protocols array
  validateType(obj, "protocols", "array", objName & ".protocols", issues)
  if "protocols" in obj and obj["protocols"].kind == JArray:
    let protocols = obj["protocols"]
    for i in 0 ..< protocols.len:
      let protocol = protocols[i]
      if protocol.kind == JObject:
        validateProtocol(protocol, i, objName, issues)

  # Validate dynamic assembler fields (time series).
  validateTimeSeries(obj, "is_clipped", objName & ".is_clipped", "bool", issues)
  validateTimeSeries(obj, "is_clip_immune", objName & ".is_clip_immune", "bool", issues)
  validateTimeSeries(obj, "uses_count", objName & ".uses_count", "int", issues)

proc validateBuildingFields*(obj: JsonNode, objName: string, issues: var seq[ValidationIssue]) =
  ## Validate all building-specific fields (legacy buildings).
  let buildingFields = [
    "input_resources", "output_resources", "conversion_remaining",
    "is_converting"
  ]
  requireFields(obj, buildingFields, objName, issues)

  # Validate dynamic building fields (always time series).
  validateInventoryFormat(obj, "input_resources", objName & ".input_resources", issues)
  validateInventoryFormat(obj, "output_resources", objName & ".output_resources", issues)
  validateTimeSeries(obj, "conversion_remaining", objName & ".conversion_remaining", "float", issues)
  validateTimeSeries(obj, "is_converting", objName & ".is_converting", "bool", issues)

proc validateObject*(obj: JsonNode, objIndex: int, replayData: JsonNode, issues: var seq[ValidationIssue]) =
  ## Validate a single object in the replay.
  let objName = "Object " & $objIndex

  # All objects have these required fields.
  # type_name is optional if type_id is present (loader can resolve it).
  let requiredFields = [
    "id", "location", "orientation", "inventory", "inventory_max", "color"
  ]
  requireFields(obj, requiredFields, objName, issues)

  # Validate static fields.
  validateStaticValue(obj, "id", "int", objName & ".id", issues)
  validatePositiveInt(obj, "id", objName & ".id", issues)

  # type_name is required unless type_id is present (which allows fallback resolution).
  if "type_name" notin obj and "type_id" notin obj:
    issues.add(ValidationIssue(
      message: &"{objName} must have either 'type_name' or 'type_id'",
      field: objName & ".type_name"
    ))
  elif "type_name" in obj:
    validateStaticValue(obj, "type_name", "string", objName & ".type_name", issues)
    if "type_names" in replayData:
      let typeName = obj["type_name"].getStr()
      let typeNames = replayData["type_names"].to(seq[string])
      if typeName notin typeNames:
        issues.add(ValidationIssue(
          message: &"{objName}.type_name '{typeName}' not in type_names list",
          field: objName & ".type_name"
        ))
  elif "type_id" in obj:
    # Validate type_id is in range if present.
    validateStaticValue(obj, "type_id", "int", objName & ".type_id", issues)
    if "type_names" in replayData:
      let typeId = obj["type_id"].getInt()
      let typeNames = replayData["type_names"].to(seq[string])
      if typeId < 0 or typeId >= typeNames.len:
        issues.add(ValidationIssue(
          message: &"{objName}.type_id {typeId} out of range (0..{typeNames.len - 1})",
          field: objName & ".type_id"
        ))

  # Validate dynamic fields (always time series).
  validateLocation(obj, "location", objName & ".location", issues)
  validateTimeSeries(obj, "orientation", objName & ".orientation", "int", issues)
  validateInventoryFormat(obj, "inventory", objName & ".inventory", issues)
  validateTimeSeries(obj, "inventory_max", objName & ".inventory_max", "int", issues)
  validateTimeSeries(obj, "color", objName & ".color", "int", issues)

  # Validate specific object types.
  if obj.getOrDefault("is_agent").getBool() or "agent_id" in obj:
    validateAgentFields(obj, objName, replayData, issues)
  elif "protocols" in obj:
    validateAssemblerFields(obj, objName, issues)
  elif "input_resources" in obj:
    validateBuildingFields(obj, objName, issues)

proc validateReplaySchema*(data: JsonNode, issues: var seq[ValidationIssue]) =
  ## Validate that replay data matches the version 3 schema specification.
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
    issues.add(ValidationIssue(
      message: &"Missing required keys: {joinedMissing}",
      field: ""
    ))
  if unexpected.len > 0:
    let joinedUnexpected = unexpected.join(", ")
    issues.add(ValidationIssue(
      message: &"Unexpected keys present: {joinedUnexpected}",
      field: ""
    ))

  # Top-level field validation.
  if "version" in data:
    let version = data["version"].getInt()
    if version != 3:
      issues.add(ValidationIssue(
        message: &"'version' must equal 3, got {version}",
        field: "version"
      ))

  validateNonNegativeNumber(data, "num_agents", "num_agents", issues)
  validateNonNegativeNumber(data, "max_steps", "max_steps", issues)

  # Validate map_size.
  if "map_size" in data:
    let mapSize = data["map_size"]
    validateTypeValue(mapSize, "array", "map_size", issues)
    if mapSize.kind == JArray and mapSize.len != 2:
      issues.add(ValidationIssue(
        message: "'map_size' must have exactly 2 dimensions",
        field: "map_size"
      ))
    if mapSize.kind == JArray:
      for i in 0..<mapSize.len:
        validatePositiveIntValue(mapSize[i], "map_size[" & $i & "]", issues)

  # Required string lists.
  for field in ["action_names", "item_names", "type_names"]:
    validateStringList(data, field, field, issues, allowEmptyStrings = true)

  # Optional file_name validation.
  if "file_name" in data:
    validateType(data, "file_name", "string", "file_name", issues)
    if "file_name" in data and data["file_name"].kind == JString and data["file_name"].getStr().len == 0:
      issues.add(ValidationIssue(
        message: "'file_name' must be non-empty",
        field: "file_name"
      ))

  # Optional string lists.
  if "group_names" in data:
    validateStringList(data, "group_names", "group_names", issues, allowEmptyStrings = true)

  # Optional reward sharing matrix.
  if "reward_sharing_matrix" in data and "num_agents" in data:
    let matrix = data["reward_sharing_matrix"]
    validateTypeValue(matrix, "array", "reward_sharing_matrix", issues)
    let numAgents = data["num_agents"].getInt()
    if matrix.kind == JArray and matrix.len != numAgents:
      issues.add(ValidationIssue(
        message: &"'reward_sharing_matrix' must have {numAgents} rows",
        field: "reward_sharing_matrix"
      ))
    if matrix.kind == JArray:
      for i, row in matrix.getElems():
        validateTypeValue(row, "array", "reward_sharing_matrix[" & $i & "]", issues)
        if row.kind == JArray and row.len != numAgents:
          issues.add(ValidationIssue(
            message: &"'reward_sharing_matrix[{i}]' must have {numAgents} columns",
            field: "reward_sharing_matrix[" & $i & "]"
          ))
        if row.kind == JArray:
          for v in row.getElems():
            if v.kind notin {JInt, JFloat}:
              issues.add(ValidationIssue(
                message: &"'reward_sharing_matrix[{i}]' must contain numbers",
                field: "reward_sharing_matrix[" & $i & "]"
              ))

  # Objects validation.
  var agentCount = 0
  if "objects" in data:
    let objects = data["objects"]
    validateTypeValue(objects, "array", "objects", issues)
    if objects.kind == JArray:
      for obj in objects.getElems():
        if obj.kind != JObject:
          issues.add(ValidationIssue(
            message: "'objects' must contain objects",
            field: "objects"
          ))

      # Validate each object and count agents.
      for i, obj in objects.getElems():
        validateObject(obj, i, data, issues)
        if obj.getOrDefault("is_agent").getBool() or "agent_id" in obj:
          agentCount += 1

  if "num_agents" in data:
    let expectedAgents = data["num_agents"].getInt()
    if agentCount != expectedAgents:
      issues.add(ValidationIssue(
        message: &"Expected {expectedAgents} agents, found {agentCount}",
        field: "objects"
      ))

proc validateReplay*(data: JsonNode): seq[ValidationIssue] =
  ## Validate that replay data matches the version 3 schema specification.
  ## Returns a sequence of validation issues found. Empty sequence means valid.
  var issues: seq[ValidationIssue]
  validateReplaySchema(data, issues)
  return issues

proc `prettyPrint`*(issues: seq[ValidationIssue]) =
  ## Display validation issues in a readable format.
  if issues.len == 0:
    echo "✓ No validation issues found."
    return

  echo &"✗ Found {issues.len} validation issue(s):"
  for i, issue in issues:
    let fieldInfo = if issue.field.len > 0: &" (field: {issue.field})" else: ""
    echo &"  {i+1}. {issue.message}{fieldInfo}"

