import ../src/mettascope/pixelator

generatePixelAtlas(
  size = 2048,
  margin = 4,
  dirsToScan = @[
    "data/agents",
    "data/objects",
    "data/view",
    "data/minimap"
  ],
  outputImagePath = "data/atlas.png",
  outputJsonPath = "data/atlas.json"
)
