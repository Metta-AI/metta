import
  std/[json, strformat, strutils, httpclient],
  zippy,
  boxy, windy, opengl,
  fidget2/[loader, hybridrender],
  mettascope/[validation, replays, worldmap]

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


proc testReplayFromUrl(url: string) =
  ## Test loading and validating a replay from a URL.
  echo &"Testing replay loading from URL: {url}"

  # Fetch the replay data from URL
  let client = newHttpClient()
  defer: client.close()

  let response = client.get(url)
  if response.code != Http200:
    raise newException(ValueError, &"Failed to fetch replay from URL: HTTP {response.code}")

  let compressedData = response.body
  echo &"Fetched {compressedData.len} bytes of compressed data"

  # Decompress the data
  var decompressed: string
  try:
    decompressed = zippy.uncompress(compressedData)
  except:
    raise newException(ValueError, "Failed to decompress replay data from URL")

  echo &"Decompressed to {decompressed.len} bytes"

  # Validate the replay first
  let jsonData = parseJson(decompressed)
  let issues = validateReplay(jsonData)
  echo &"Found {issues.len} validation issues"
  if issues.len > 0:
    echo "Validation issues found:"
    issues.prettyPrint()
    raise newException(AssertionError, &"Validation failed for replay {url}: {issues[0].message}")
  else:
    echo "No validation issues found"

  # Load the replay using loadReplayString (which does full parsing and object construction)
  let fileName = url.split("/")[^1]  # Extract filename from URL
  let replay = loadReplayString(decompressed, fileName)
  echo &"✓ Successfully loaded replay: {replay.numAgents} agents, {replay.maxSteps} steps, map size {replay.mapSize[0]}x{replay.mapSize[1]}"

block url_replay_test:
  # Initialize headless graphics context for replay loading
  initHeadlessFidget2()

  # Good replays that should load successfully
  let goodReplayUrls = [
    "https://softmax-public.s3.amazonaws.com/datasets/replays/a9723dd1-ea98-4b57-a3ef-e17ea3ffb7d4/ae5f6fe0-950d-4917-b11e-e9dae451bd7f.json.z"
  ]

  # Bad replays that should fail validation with clear errors
  let badReplayUrls = [
    "https://softmax-public.s3.amazonaws.com/datasets/replays/a9723dd1-ea98-4b57-a3ef-e17ea3ffb7d4/e12d08b8-32da-43ab-ac84-b29dbebedb1d.json.z"
  ]

  # Test good replays - these should load successfully
  echo "Testing good replays..."
  for url in goodReplayUrls:
    testReplayFromUrl(url)

  # Test bad replays - these should fail validation
  echo "\nTesting bad replays (should fail validation)..."
  for url in badReplayUrls:
    try:
      testReplayFromUrl(url)
      # If we get here, the test failed - bad replay loaded when it shouldn't have
      raise newException(AssertionError, &"Bad replay {url} loaded successfully when it should have failed validation")
    except AssertionError as e:
      # This is expected - validation should fail for bad replays
      echo &"✓ Bad replay {url} correctly failed validation: {e.msg}"
    except:
      # Other errors (like network issues) should be re-raised
      let err = getCurrentException()
      echo &"Unexpected error for bad replay {url}: {err.msg}"
      raise err
