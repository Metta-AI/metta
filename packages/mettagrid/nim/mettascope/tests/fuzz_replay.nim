import
  std/[json, random, strformat],
  zippy,
  mettascope/replays, test_replay

randomize()

const iterations = 10000

proc fuzzJsonField(obj: JsonNode, fieldPath: seq[string], depth: int = 0): JsonNode =
  ## Recursively fuzz JSON fields by modifying values and structure
  if depth > 3 or rand(5) == 0: return obj  # 20% chance to leave unchanged

  result = obj.copy()

  case obj.kind
  of JInt:
    case rand(6)
    of 0: result = %(-obj.getInt())  # Negative
    of 1: result = %(obj.getInt() + rand(1000))  # Slightly larger
    of 2: result = %(obj.getInt() * rand(10))  # Multiplied
    of 3: result = %obj.getFloat()  # Change to float (same value)
    of 4: result = %($obj.getInt())  # Change to string
    of 5: result = %[obj.getInt()]  # Change to single-element array
    else: discard

  of JFloat:
    case rand(5)
    of 0: result = %(-obj.getFloat())  # Negative
    of 1: result = %(obj.getFloat() + rand(1000).float)  # Slightly larger
    of 2: result = %(obj.getFloat() * rand(10).float)  # Multiplied
    of 3: result = %obj.getInt()  # Change to int (truncated)
    of 4: result = %($obj.getFloat())  # Change to string
    else: discard

  of JString:
    case rand(4)
    of 0: result = %(obj.getStr() & $rand(100))  # Append random string
    of 1: result = %""  # Empty string
    of 2: result = %rand(100)  # Change to int
    of 3: result = %[obj.getStr()]  # Change to single-element array
    else: discard

  of JBool:
    case rand(3)
    of 0: result = %(not obj.getBool())  # Flip bool
    of 1: result = %rand(2)  # Change to 0 or 1 int
    of 2: result = %($obj.getBool())  # Change to string
    else: discard

  of JArray:
    let arr = obj.getElems()
    if arr.len > 0:
      case rand(5)
      of 0: # Modify random element
        var newArr = arr
        let idx = rand(arr.high)
        newArr[idx] = fuzzJsonField(newArr[idx], fieldPath, depth + 1)
        result = %newArr
      of 1: # Duplicate last element
        var newArr = arr
        if newArr.len > 0:
          newArr.add(newArr[^1])
        result = %newArr
      of 2: # Remove random element (if more than 1)
        var newArr = arr
        if newArr.len > 1:
          newArr.delete(rand(newArr.high))
        result = %newArr
      of 3: # Add random number
        var newArr = arr
        newArr.add(%rand(1000))
        result = %newArr
      of 4: # Change to non-array
        result = %rand(100)
      else: discard

  of JObject:
    var newObj = newJObject()
    for k, v in obj:
      if rand(20) == 0:  # 5% chance to remove field
        continue
      if rand(10) == 0:  # 10% chance to fuzz this field
        var newPath = fieldPath & k
        newObj[k] = fuzzJsonField(v, newPath, depth + 1)
      else:
        newObj[k] = v
    # Sometimes add an extra invalid field
    if rand(10) == 0:
      newObj["invalid_field_" & $rand(100)] = %rand(1000)
    result = newObj

  of JNull:
    case rand(4)
    of 0: result = %rand(100)
    of 1: result = %"null"
    of 2: result = %false
    of 3: result = %[]  # Empty array
    else: discard

proc fuzzReplay(replay: JsonNode): JsonNode =
  ## Apply random fuzzing mutations to replay JSON
  result = fuzzJsonField(replay, @["root"])

echo "Starting replay fuzzing with field-level mutations..."

for i in 0 ..< iterations:
  # Generate fresh valid replay each iteration
  let baseReplay = makeValidReplay("fuzz_test_replay.json.z")
  let fuzzedReplay = fuzzReplay(baseReplay)

  echo &"{i}: Fuzzing replay structure"

  # Test 1: Try to validate schema on fuzzed JSON
  try:
    validateReplaySchema(fuzzedReplay)
  except CatchableError:
    discard

  # Test 2: Try to load fuzzed replay
  try:
    let jsonStr = $fuzzedReplay
    let replay = loadReplayString(jsonStr, "fuzz_test.json.z")
    doAssert replay != nil
  except CatchableError:
    discard

  # Test 3: Also test with compressed version
  try:
    let jsonStr = $fuzzedReplay
    let compressed = zippy.compress(jsonStr)
    let replay = loadReplay(compressed, "fuzz_test.json.z")
    doAssert replay != nil
  except CatchableError:
    discard

echo "Replay field-level fuzzing completed successfully"