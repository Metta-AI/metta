import
  std/[os, strutils, tables, sets],
  vmath,
  ../src/mettascope/[replays, common]

proc drawAsciiMap(replay: Replay): string =
  ## Draw the map in ASCII and save to file.

  # Determine bounds
  var minX = 0
  var maxX = replay.mapSize[0] - 1
  var minY = 0
  var maxY = replay.mapSize[1] - 1


  var map = initTable[(int, int), string]()

  # Populate map
  for obj in replay.objects:
    if obj.location.len > 0:
      let loc = obj.location[0]
      var cell = "  "

      # Logic to determine cell string based on typeName
      case obj.typeName:
      of "agent": cell = "@@"
      of "assembler": cell = "As"
      of "carbon_extractor": cell = "Ca"
      of "charger": cell = "En"
      of "chest": cell = "Ch"
      of "germanium_extractor": cell = "Ge"
      of "oxygen_extractor": cell = "O2"
      of "silicon_extractor": cell = "Si"
      of "wall": cell = "##"
      else:
         if obj.typeName.len >= 2:
            cell = obj.typeName[0..1]
         elif obj.typeName.len == 1:
            cell = obj.typeName & " "
         else:
            cell = "??"

      map[(loc.x.int, loc.y.int)] = cell

  var output = ""
  var line = "+"
  for x in minX .. maxX: line.add "--"
  line.add "+"
  output.add line & "\n"

  for y in minY .. maxY:
    line = "|"
    for x in minX .. maxX:
      var cell = "  "
      if map.hasKey((x, y)):
        cell = map[(x, y)]
      else:
        cell = "  "
      line.add cell
    line.add "|"
    output.add line & "\n"

  line = "+"
  for x in minX .. maxX: line.add "--"
  line.add "+"
  output.add line & "\n"
  return output

let replayDir = "tests" / "data" / "replays"
var count = 0
for file in walkDirRec(replayDir):
  if file.endsWith(".json.gz") or file.endsWith(".json.z") or file.endsWith(".json"):
    echo "Loading ", file.extractFilename

    let r = loadReplay(file)
    doAssert r != nil
    doAssert r.version == 3

    # Draw ASCII map for the first step
    let mapFile = file.parentDir / (file.extractFilename & ".map.txt")
    let map = drawAsciiMap(r)

    when defined(writeMaps):
      writeFile(mapFile, map)
      echo "Saved map to ", mapFile
    else:
      if not fileExists(mapFile):
        doAssert false, "Map file does not exist: " & mapFile
      else:
        let expectedMap = readFile(mapFile)
        doAssert map == expectedMap, "Map does not match expected map: " & mapFile

    count += 1

echo "Loaded ", count, " replays successfully."
