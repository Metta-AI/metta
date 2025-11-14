import
  std/[json, os, osproc, sequtils, strformat, strutils, times],
  ../src/mettascope/validation

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
