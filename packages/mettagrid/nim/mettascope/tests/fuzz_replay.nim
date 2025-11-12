import
  std/[json, random, strformat, math, strutils, os, osproc, times],
  zippy,
  boxy, windy, opengl,
  fidget2/[loader, hybridrender],
  mettascope/[replays, worldmap], test_replay

const
  iterations = 1000

randomize()

proc initHeadlessFidget2*() =
  ## Initialize fidget2 in headless mode for testing.
  ## Sets up minimal state without windows or rendering.

  # Create an invisible fidget2 window
  let window = newWindow("Headless", ivec2(1, 1), visible = false)
  makeContextCurrent(window)
  loadExtensions()

  # Initialize boxy with the atlas, so we can fully test that assets are found for replays.
  bxy = newBoxy()
  buildAtlas()


proc fuzzJsonField(obj: JsonNode, fieldPath: seq[string], depth: int = 0, baseReplay: JsonNode): JsonNode =
  ## Spec-aware fuzzing - respects replay format constraints
  if rand(100) != 0: return obj  # 99% chance to leave unchanged

  result = obj.copy()

  # Get spec constraints from base replay
  let mapSize = baseReplay["map_size"]
  let maxX = mapSize[0].getInt() - 1
  let maxY = mapSize[1].getInt() - 1
  let numActions = baseReplay["action_names"].len
  let numItems = baseReplay["item_names"].len

  case obj.kind
  of JInt:
    # Context-aware integer fuzzing based on field path
    if fieldPath.len >= 2 and fieldPath[0] == "objects":
      let fieldName = fieldPath[^1]  # Last element is the field name
      case fieldName
      of "action_id":
        # action_id must be valid index into action_names
        if numActions > 0:
          result = %rand(numActions - 1)
      of "group_id":
        # group_id should be non-negative
        result = %max(0, obj.getInt() + rand(3) - 1)
      of "color":
        # color is typically 0-255
        result = %max(0, min(255, obj.getInt() + rand(3) - 1))
      of "inventory_max":
        # inventory_max should be reasonable
        result = %max(0, obj.getInt() + rand(5) - 2)
      of "vision_size":
        # vision_size should be positive
        result = %max(1, obj.getInt() + rand(3) - 1)
      else:
        # Generic small change for other integers
        result = %(obj.getInt() + rand(3) - 1)
    else:
      # Top-level integers
      result = %(max(0, obj.getInt() + rand(3) - 1))

  of JFloat:
    # Float values should stay reasonable
    result = %(max(0.0, obj.getFloat() + (rand(3).float - 1.0)))

  of JBool:
    if rand(20) == 0:  # 5% chance to flip boolean
      result = %(not obj.getBool())

  of JArray:
    if rand(200) == 0:  # Very rare array modifications
      let arr = obj.getElems()
      if arr.len > 0:
        var newArr = arr

        # Context-aware array fuzzing
        if fieldPath.len >= 2 and fieldPath[^1] == "location":
          # Location arrays: [x, y] - keep within map bounds
          if arr.len == 2 and arr[0].kind == JInt and arr[1].kind == JInt:
            let x = max(0, min(maxX, arr[0].getInt() + rand(3) - 1))
            let y = max(0, min(maxY, arr[1].getInt() + rand(3) - 1))
            result = %[x, y]
            return

        elif fieldPath.len >= 2 and fieldPath[^1] == "inventory":
          # Inventory arrays: [[item_id, count], ...] - keep valid
          if arr.len > 0 and rand(10) == 0:  # Occasionally remove one item
            newArr.delete(rand(min(arr.high, 2)))  # Don't remove too many
          elif rand(20) == 0 and arr.len < 5:  # Occasionally add valid item
            let itemId = rand(numItems)
            let count = rand(10) + 1
            newArr.add(%[itemId, count])
          else:
            # Sometimes fuzz existing inventory items
            for j in 0..<newArr.len:
              if newArr[j].kind == JArray and newArr[j].len == 2:
                var itemId = fuzzJsonField(newArr[j][0], fieldPath & $j & "item_id", depth + 1, baseReplay)
                var count = fuzzJsonField(newArr[j][1], fieldPath & $j & "count", depth + 1, baseReplay)
                newArr[j] = %[itemId, count]

        else:
          # Generic array fuzzing - sometimes fuzz elements
          for j in 0..<newArr.len:
            if rand(50) == 0:  # Occasionally fuzz array elements
              newArr[j] = fuzzJsonField(newArr[j], fieldPath & $j, depth + 1, baseReplay)

        result = %newArr

  else:
    discard  # Leave other types unchanged

proc fuzzReplay(replay: JsonNode): (JsonNode, seq[string]) =
  ## Apply random fuzzing mutations to replay JSON.
  ## Returns (fuzzedReplay, changeLog) where changeLog describes what was changed.
  result[0] = replay.copy()
  result[1] = @[]

  # Only fuzz the objects array, and only specific properties within objects
  if "objects" in result[0] and result[0]["objects"].kind == JArray:
    var newObjects = newJArray()
    for i, obj in result[0]["objects"].getElems():
      if obj.kind == JObject:
        var newObj = newJObject()
        for k, v in obj:
          if k in ["location", "orientation", "inventory", "inventory_max", "color",
                   "current_reward", "total_reward", "action_id", "action_param", "action_success",
                   "freeze_remaining", "is_frozen", "freeze_duration", "vision_size", "group_id"]:
            # Only fuzz these data fields that don't affect image loading
            let originalValue = v
            let fuzzedValue = fuzzJsonField(v, @["objects", $i, k], 0, result[0])
            newObj[k] = fuzzedValue

            # Record the change if it actually changed
            if $fuzzedValue != $originalValue:
              result[1].add(&"Object {i}.{k}: {$originalValue} -> {$fuzzedValue}")
          else:
            # Keep critical fields unchanged
            newObj[k] = v
        newObjects.add(newObj)
      else:
        newObjects.add(obj)
    result[0]["objects"] = newObjects

initHeadlessFidget2()
let baseReplay = generateReplayForTesting()

type
  FailureType = enum
    ValidationFailure
    LoadingFailure

  TestResult = object
    iteration: int
    failureType: FailureType
    err: ref Exception
    changeLog: seq[string]

var
  passedCount = 0
  failures: seq[TestResult]

for i in 0 ..< iterations:
  let (fuzzedReplay, changeLog) = fuzzReplay(baseReplay)
  let jsonStr = $fuzzedReplay

  # Test validation (collect failures but don't stop)
  var validationPassed = true
  try:
    validateReplaySchema(fuzzedReplay)
  except Exception as e:
    let failure = TestResult(
      iteration: i,
      failureType: ValidationFailure,
      err: e,
      changeLog: changeLog,
    )
    failures.add(failure)
    validationPassed = false

  # Test loading regardless of validation result
  try:
    let replay = loadReplayString(jsonStr, "fuzz_test.json.z")
    doAssert replay != nil
    # Only count as passed if both validation AND loading succeeded
    if validationPassed:
      passedCount += 1
  except Exception as e:
    let failure = TestResult(
      iteration: i,
      failureType: LoadingFailure,
      err: e,
      changeLog: changeLog,
    )
    failures.add(failure)

# Report results
echo ""
echo "=== FUZZING RESULTS ==="
echo &"Total iterations: {iterations}"
echo &"Passed: {passedCount} ({passedCount.float32 / iterations.float32 * 100:.1f}%) - both validation and loading succeeded"
echo &"Issues found: {failures.len} ({failures.len.float32 / iterations.float32 * 100:.1f}%) - validation or loading problems"

if failures.len > 0:
  echo ""
  echo "=== FAILURE DETAILS ==="

  # Group failures by type and check for segfaults
  var validationFailures: seq[TestResult]
  var loadingFailures: seq[TestResult]
  var segfaultFailures: seq[TestResult]

  for failure in failures:
    # Check if this failure is actually a segfault, regardless of original classification
    let isSegfault = strutils.contains(failure.err.msg, "SIGSEGV") or
                    strutils.contains(failure.err.msg, "Illegal storage access") or
                    strutils.contains(failure.err.msg, "Access violation") or
                    strutils.contains(failure.err.msg, "Segmentation fault")

    if isSegfault:
      segfaultFailures.add(failure)
    else:
      case failure.failureType
      of ValidationFailure: validationFailures.add(failure)
      of LoadingFailure: loadingFailures.add(failure)

  # Report segfaults first (most critical)
  if segfaultFailures.len > 0:
    echo ""
    echo &"🚨 CRITICAL: {segfaultFailures.len} SEGMENTATION FAULTS DETECTED!"
    for failure in segfaultFailures:
      echo &"  Iteration {failure.iteration}: {failure.err.msg}"
      echo &"  Stack trace: {failure.err.getStackTrace()}"
      if failure.changeLog.len > 0:
        echo &"  Fuzzing applied:"
        for change in failure.changeLog:
          echo &"    {change}"
      else:
        echo &"  Fuzzing applied: none (no changes made)"
      echo ""

  # Report validation failures
  if validationFailures.len > 0:
    echo ""
    echo &"⚠️  VALIDATION FAILURES: {validationFailures.len}"
    for failure in validationFailures:
      if failure.changeLog.len > 0:
        echo &"  Fuzzing applied:"
        for change in failure.changeLog:
          echo &"    {change}"
      else:
        echo &"  Fuzzing applied: none (no changes made)"
      echo &"  Iteration {failure.iteration}: {failure.err.msg}"
      echo &"  Stack trace: {failure.err.getStackTrace()}"

  # Report loading failures
  if loadingFailures.len > 0:
    echo ""
    echo &"⚠️  LOADING FAILURES: {loadingFailures.len}"
    for failure in loadingFailures:
      if failure.changeLog.len > 0:
        echo &"  Fuzzing applied:"
        for change in failure.changeLog:
          echo &"    {change}"
      else:
        echo &"  Fuzzing applied: none (no changes made)"
      echo &"  Iteration {failure.iteration}: {failure.err.msg}"
      echo &"  Stack trace: {failure.err.getStackTrace()}"

  echo ""
  echo "=== SUMMARY ==="
  if segfaultFailures.len > 0:
    echo &"🚨 {segfaultFailures.len} critical segfaults"
  if validationFailures.len > 0:
    echo &"⚠️  {validationFailures.len} validation failures"
  if loadingFailures.len > 0:
    echo &"⚠️  {loadingFailures.len} loading failures - potential bugs in replay parsing"
else:
  echo ""
  echo "🎉 No failures detected - all tests passed!"

echo ""
echo "Fuzzing completed successfully"
