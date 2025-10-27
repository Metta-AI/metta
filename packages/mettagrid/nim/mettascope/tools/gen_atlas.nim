import ../src/mettascope/pixelatlas

generatePixelAtlas(
  size = 1024,
  margin = 4,
  dirsToScan = @[
    "data/agents",
    "data/objects",
    "data/view"
  ],
  outputImagePath = "data/atlas.png",
  outputJsonPath = "data/atlas.json"
)
