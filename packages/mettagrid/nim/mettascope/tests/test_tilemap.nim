import
  std/[strformat],
  boxy, windy, pixie,
  ../src/mettascope/[tilemap], perlin

import boxy, opengl, pixie, windy, random, boxy/shaders

let window = newWindow("Tilemap", ivec2(1280, 800))
makeContextCurrent(window)
loadExtensions()

let bxy = newBoxy()

proc generateTileMap(width: int, height: int, atlasPath: string): TileMap =
# Generate a 1024x1024 texture where each pixel is a byte index into the 16x16 tile map

  var terrainMap = newTileMap(
    width = width,
    height = height,
    tileSize = 64,
    atlasPath = "tools/blob7x7.png"
  )

  let mb = width.float32 * height.float32/1024/1024
  echo &"map size: {mb:.2f} MB"
  var asteroidMap: seq[bool] = newSeq[bool](width * height)

  # Fill with random tile indices (0-255 for 16x16 atlas) or load from file
  randomize()
  let p1 = initPerlin2D(1337'u32)
  let p2 = initPerlin2D(837'u32)
  for y in 0 ..< height:
    for x in 0 ..< width:
      let v = 0 +
        p1.noise(x.float32 * 0.2, y.float32 * 0.2) +
        p2.noise(x.float32 * 0.02, y.float32 * 0.02)
      if v > 0:
        asteroidMap[y * width + x] = true

  # Generate random tilemap if file doesn't exist or is invalid
  echo "Generating random tilemap... this will take a few seconds..."
  let patternToTile = @[
    18, 17, 4, 4, 12, 22, 4, 4, 30, 13, 41, 41, 30, 13, 41, 41, 19, 23, 5, 5, 37,
    9, 5, 5, 30, 13, 41, 41, 30, 13, 41, 41, 24, 43, 39, 39, 44, 45, 39, 39, 48,
    32, 46, 46, 48, 32, 46, 46, 24, 43, 39, 39, 44, 45, 39, 39, 48, 32, 46, 46,
    48, 32, 46, 46, 36, 10, 3, 3, 16, 40, 3, 3, 20, 27, 6, 6, 20, 27, 6, 6, 25,
    15, 2, 2, 26, 38, 2, 2, 20, 27, 6, 6, 20, 27, 6, 6, 24, 43, 39, 39, 44, 45,
    39, 39, 48, 32, 46, 46, 48, 32, 46, 46, 24, 43, 39, 39, 44, 45, 39, 39, 48,
    32, 46, 46, 48, 32, 46, 46, 28, 28, 8, 8, 21, 21, 8, 8, 33, 33, 7, 7, 33, 33,
    7, 7, 35, 35, 31, 31, 14, 14, 31, 31, 33, 33, 7, 7, 33, 33, 7, 7, 47, 47, 1,
    1, 42, 42, 1, 1, 34, 34, 29, 29, 34, 34, 29, 29, 47, 47, 1, 1, 42, 42, 1, 1,
    34, 34, 29, 29, 34, 34, 29, 29, 28, 28, 8, 8, 21, 21, 8, 8, 33, 33, 7, 7, 33,
    33, 7, 7, 35, 35, 31, 31, 14, 14, 31, 31, 33, 33, 7, 7, 33, 33, 7, 7, 47, 47,
    1, 1, 42, 42, 1, 1, 34, 34, 29, 29, 34, 34, 29, 29, 47, 47, 1, 1, 42, 42, 1,
    1, 34, 34, 29, 29, 34, 34, 29, 29
  ]
  for i in 0 ..< terrainMap.indexData.len:
    # Create some patterns for more interesting visuals
    let x = i mod width
    let y = i div width

    proc get(map: seq[bool], x: int, y: int): int =
      if x < 0 or y < 0 or x >= width or y >= height:
        return 0
      if map[y * width + x]:
        return 1
      return 0

    # On off
    var tile: uint8 = 0
    if asteroidMap[y * width + x]:
      tile = 0
    else:
      #tile = 29

      let
        pattern = (
          1 * asteroidMap.get(x-1, y+1) + # NW
          2 * asteroidMap.get(x, y+1) + # N
          4 * asteroidMap.get(x+1, y+1) + # NE
          8 * asteroidMap.get(x+1, y) + # E
          16 * asteroidMap.get(x+1, y-1) + # SE
          32 * asteroidMap.get(x, y-1) + # S
          64 * asteroidMap.get(x-1, y-1) + # SW
          128 * asteroidMap.get(x-1, y) # W
        )
      tile = patternToTile[pattern].uint8
    terrainMap.indexData[i] = tile

  terrainMap.setupGPU()
  echo "Done generating tile map"

  return terrainMap

var
  vel: Vec2
  pos: Vec2
  zoom: float32 = 1
  zoomVel: float32
  frame: int

let terrainMap = generateTileMap(1024, 1024, "tools/blob7x7.png")

# Called when it is time to draw a new frame.
window.onFrame = proc() =
  # Clear the screen and begin a new frame.
  bxy.beginFrame(window.size)

  glClearColor(0.0, 0.0, 0.0, 1.0)
  glClear(GL_COLOR_BUFFER_BIT)

  # Handle input for panning and zooming like hex example
  # Left mouse button: drag to pan
  # Mouse wheel: zoom in/out
  if window.buttonDown[MouseMiddle]:
    vel = window.mouseDelta.vec2 + vel * 0.1
  else:
    vel *= 0.99

  pos += vel

  if window.scrollDelta.y != 0:
    zoomVel = window.scrollDelta.y * 0.005
  else:
    zoomVel *= 0.95

  var zoomPow2 = zoom * zoom
  let oldMat = translate(vec2(pos.x, pos.y)) * scale(vec2(zoomPow2, zoomPow2))
  zoom += zoomVel
  zoom = clamp(zoom, 0.1, 100.0)
  zoomPow2 = zoom * zoom
  let newMat = translate(vec2(pos.x, pos.y)) * scale(vec2(zoomPow2, zoomPow2))
  let newAt = newMat.inverse() * window.mousePos.vec2
  let oldAt = oldMat.inverse() * window.mousePos.vec2
  pos -= (oldAt - newAt).xy * (zoomPow2)


  # Create MVP matrix
  let projection = ortho(0.0f, window.size.x.float32, window.size.y.float32, 0.0f, -1.0f, 1.0f)
  let view = translate(vec3(pos.x, pos.y, 0.0f)) *
             scale(vec3(zoomPow2 * terrainMap.width.float32/2, zoomPow2 * terrainMap.height.float32/2, 1.0f))
  let mvp = projection * view

  # Alter the terrain tiles by clicking on them.
  if window.buttonDown[MouseLeft]:
    let screenPos = vec2(
      window.mousePos.x.float32,
      window.mousePos.y.float32
    )
    let tilePos = view.inverse() * vec3(screenPos.x, screenPos.y, 0.0f)
    let tileX = ((tilePos.x / 2 - 1) * terrainMap.width.float32).int
    let tileY = ((tilePos.y / 2 - 1) * terrainMap.height.float32).int

    let mouseTile = terrainMap.getTile(
      tileX,
      tileY
    )
    echo "mouse tile: ", tileX, ", ", tileY, " -> ", mouseTile

  terrainMap.draw(mvp, zoom, 1.25f)

  # End this frame, flushing the draw commands.
  bxy.endFrame()

  # Swap buffers displaying the new Boxy frame.
  window.swapBuffers()
  inc frame

while not window.closeRequested:
  pollEvents()
