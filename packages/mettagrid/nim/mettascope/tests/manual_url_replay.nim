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

  # Load the replay using loadReplayString (which does full parsing and object construction)
  let fileName = url.split("/")[^1]  # Extract filename from URL
  let replay = loadReplayString(decompressed, fileName)
  echo &"Loaded replay: {replay.numAgents} agents, {replay.maxSteps} steps, map size {replay.mapSize[0]}x{replay.mapSize[1]}"

  # Validate the replay
  let issues = validateReplay(parseJson(decompressed))
  if issues.len > 0:
    echo "Validation issues found:"
    issues.prettyPrint()
    raise newException(AssertionError, "Validation issues found in URL replay")

  echo "✓ Successfully loaded and validated replay from URL"

block url_replay_test:
  # Initialize headless graphics context for replay loading
  initHeadlessFidget2()

  # Array of replay URLs to test
  let replayUrls = [
    "https://softmax-public.s3.amazonaws.com/datasets/replays/a9723dd1-ea98-4b57-a3ef-e17ea3ffb7d4/ae5f6fe0-950d-4917-b11e-e9dae451bd7f.json.z",
    "https://softmax-public.s3.amazonaws.com/datasets/replays/a9723dd1-ea98-4b57-a3ef-e17ea3ffb7d4/e12d08b8-32da-43ab-ac84-b29dbebedb1d.json.z"
  ]

  # Test each URL
  for url in replayUrls:
    testReplayFromUrl(url)
