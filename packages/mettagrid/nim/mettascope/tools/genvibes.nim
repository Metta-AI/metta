# Parse the python vibes.py file and generate the nim vibes.nim file.

import std/[os, strutils, json, unicode, strformat]

let pythonVibes = "../../../cogames/src/cogames/cogs_vs_clips/vibes.py"
let data = readFile(pythonVibes)
let lines = data.splitLines()
for line in lines:
  if line.strip().startsWith("Vibe("):
    let vibe = line.strip().split("Vibe(")[1].split(")")[0]
    let vibeName = vibe.split(",")[1].strip().replace("\"", "")
    let vibeSymbol = vibe.split(",")[0].strip().replace("\"", "")

    # format: emoji_u2764_200d_1fa79.png
    var runeNums: seq[string] = @[]
    for rune in vibeSymbol.toRunes():
      let num = cast[uint32](rune)
      let hexNum = &"{num:04x}"
      if hexNum != "fe0f":
        runeNums.add(hexNum)
    let runeStr = runeNums.join("_")

    let vibePngFile = "../../../../../noto-emoji/png/32/emoji_u" & runeStr & ".png"
    if existsFile(vibePngFile):
      copyFile(vibePngFile, "data/vibe/" & vibeName & ".png")
    else:
      echo "vibeName: ", vibeName
      echo "vibeSymbol: ", vibeSymbol
      echo "runeStr: ", runeStr
      echo "vibePngFile not found: ", vibePngFile
