import
  std/[json, os, osproc, strformat, strutils, times],
  zippy,
  mettascope/[validation, replays]

proc loadReplay(path: string): JsonNode =
  ## Load and decompress a .json.z replay file.
  if not (path.endsWith(".json.gz") or path.endsWith(".json.z")):
    raise newException(ValueError, "Replay file name must end with '.json.gz' or '.json.z'")

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
    if file.endsWith(".json.gz") or file.endsWith(".json.z"):
      replayFiles.add(file)

  if replayFiles.len == 0:
    raise newException(AssertionError, &"No replay files were generated in {tmpDir}")

  # Should have exactly one replay file
  if replayFiles.len != 1:
    raise newException(AssertionError, &"Expected exactly 1 replay file, found {replayFiles.len}: {replayFiles}")

  # Validate the replay file
  let replayPath = replayFiles[0]
  let loadedReplay = loadReplay(replayPath)
  let issues = validateReplay(loadedReplay)
  if issues.len > 0:
    issues.prettyPrint()
    raise newException(AssertionError, &"Validation issues found in replay")

  echo &"✓ Successfully generated and validated replay: {extractFilename(replayPath)}"
