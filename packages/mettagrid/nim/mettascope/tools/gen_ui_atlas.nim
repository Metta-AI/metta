import std/[os]
import boxy

let rootDir = "packages/mettagrid/nim/mettascope/"
let distDir = rootDir / "dist"

if not dirExists(distDir):
  createDir(distDir)

var builder = newAtlasBuilder(1024, 4)
builder.addDir(rootDir / "data/theme/", rootDir / "data/theme/")
builder.addDir(rootDir / "data/ui/", rootDir / "data/")
builder.addDir(rootDir / "data/vibe/", rootDir / "data/")
builder.addDir(rootDir / "data/resources/", rootDir / "data/")
builder.addFont(rootDir / "data/fonts/Inter-Regular.ttf", "H1", 32.0)
builder.addFont(rootDir / "data/fonts/Inter-Regular.ttf", "Default", 18.0)
builder.write(distDir / "atlas.png", distDir / "atlas.json")
