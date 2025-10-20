
# Tilemap Example - Renders a huge tilemap using OpenGL 4.1 shaders
#
# Features:
# - 1024x1024 tilemap (over 1 million tiles!)
# - Uses a 16x16 tile atlas (256 unique tiles)
# - Custom OpenGL 4.1 shaders for efficient rendering
# - Pan with left mouse button, zoom with mouse wheel
# - Generates random pattern with interesting tile variations
#
# The whole tilemap is rendered as a single quad using:
# - Index texture: 1024x1024 R8 texture where each pixel is a tile index (0-255)
# - Atlas texture: The tile atlas loaded from testTexture.png
# - Custom OpenGL 4.1 fragment shader that samples the index, calculates atlas coordinates, and renders

import boxy, opengl, pixie, windy, random, boxy/shaders, perlin

let window = newWindow("Tilemap", ivec2(1280, 800))
makeContextCurrent(window)
loadExtensions()

let bxy = newBoxy()

type
  TileMap = ref object
    width: int
    height: int
    tileSize: int
    tileAtlas: Image
    indexData: seq[uint8]

    # OpenGL textures
    indexTexture: GLuint
    tileAtlasTexture: GLuint
    shader: Shader
    VAO: GLuint
    VBO: GLuint
    EBO: GLuint
    quadVertices: seq[float32]
    quadIndices: seq[uint32]

proc newTileMap(
  width: int,
  height: int,
  tileSize: int,
  atlasPath: string
): TileMap =
  result = TileMap(
    width: width,
    height: height,
    tileSize: tileSize,
    tileAtlas: readImage(atlasPath),
    indexData: newSeq[uint8](width * height)
  )

var vel: Vec2
var pos: Vec2
var zoom: float32 = 1
var zoomVel: float32
var frame: int

# Generate a 1024x1024 texture where each pixel is a byte index into the 16x16 tile map
const MAP_SIZE = 1024
var terrainMap = newTileMap(
  width = MAP_SIZE,
  height = MAP_SIZE,
  tileSize = 64,
  atlasPath = "tools/blob7x7.png"
)

echo "map size: ", MAP_SIZE * MAP_SIZE, " bytes"
var asteroidMap: seq[bool] = newSeq[bool](MAP_SIZE * MAP_SIZE)

# Fill with random tile indices (0-255 for 16x16 atlas) or load from file
randomize()
let p1 = initPerlin2D(1337'u32)
let p2 = initPerlin2D(837'u32)
for y in 0 ..< MAP_SIZE:
  for x in 0 ..< MAP_SIZE:
    let v = 0 +
      p1.noise(x.float32 * 0.2, y.float32 * 0.2) +
      p2.noise(x.float32 * 0.02, y.float32 * 0.02)
    if v > 0:
      asteroidMap[y * MAP_SIZE + x] = true

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
  let x = i mod MAP_SIZE
  let y = i div MAP_SIZE

  proc get(map: seq[bool], x: int, y: int): int =
    if x < 0 or y < 0 or x >= MAP_SIZE or y >= MAP_SIZE:
      return 0
    if map[y * MAP_SIZE + x]:
      return 1
    return 0

  # On off
  var tile: uint8 = 0
  if asteroidMap[y * MAP_SIZE + x]:
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

echo "Done generating tile map"

proc setupGPU(tileMap: TileMap) =
  # Create OpenGL texture for tile indices
  glGenTextures(1, tileMap.indexTexture.addr)
  glBindTexture(GL_TEXTURE_2D, tileMap.indexTexture)
  glTexImage2D(
    GL_TEXTURE_2D,
    0,
    GL_R8.GLint,
    tileMap.width.GLint,
    tileMap.height.GLint,
    0,
    GL_RED,
    GL_UNSIGNED_BYTE,
    tileMap.indexData[0].addr
  )
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST.GLint)
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST.GLint)

  # Create tile atlas texture, mipmapped.
  glGenTextures(1, tileMap.tileAtlasTexture.addr)
  glBindTexture(GL_TEXTURE_2D, tileMap.tileAtlasTexture)
  glTexImage2D(
    GL_TEXTURE_2D,
    0,
    GL_RGBA.GLint,
    tileMap.tileAtlas.width.GLint,
    tileMap.tileAtlas.height.GLint,
    0,
    GL_RGBA,
    GL_UNSIGNED_BYTE,
    tileMap.tileAtlas.data[0].addr
  )
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR.GLint)
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR.GLint)
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE.GLint)
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE.GLint)
  glGenerateMipmap(GL_TEXTURE_2D)

  # glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST.GLint)
  # glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST.GLint)

  # Vertex shader source (OpenGL 4.1)
  let vertexShaderSource = """
  #version 410 core

  layout (location = 0) in vec2 aPos;
  layout (location = 1) in vec2 aTexCoord;

  out vec2 TexCoord;

  uniform mat4 uMVP;

  void main() {
      gl_Position = uMVP * vec4(aPos, 0.0, 1.0);
      TexCoord = aTexCoord;
  }
  """

  # Fragment shader source (OpenGL 4.1)
  let fragmentShaderSource = """
#version 410 core

in vec2 TexCoord;
out vec4 FragColor;

uniform sampler2D uIndexTexture;  // set sampler to NEAREST, no mipmaps
uniform sampler2D uTileAtlas;     // mipmaps recommended
uniform vec2  uMapSize;           // map size in tiles, e.g. (256, 256)
uniform vec2  uAtlasSize;         // atlas size in texels, e.g. (2048, 2048)
uniform float uTileSize;          // tile size in texels, e.g. 16

// Set to 1.0 if each tile in the atlas has a 1px padded border copied from its edges.
// Set to 0.0 if there is no padding (may see some seams/minification artifacts).
const float GUARD = 1.0;

void main() {
    // --- 1) Which map cell (tile) are we rendering? ---
    vec2  mapPos    = TexCoord * uMapSize;        // [0..mapW/H) in tile units
    ivec2 mapTexel  = ivec2(floor(mapPos));       // integer tile coords

    // Read index without filtering/mips. uIndexTexture must be NEAREST, no mipmaps.
    // If you can, store indices as R8UI (usampler2D) and read integers directly.
    float idxNorm   = texelFetch(uIndexTexture, mapTexel, 0).r;
    int   tileIndex = int(idxNorm * 255.0 + 0.5);

    // --- 2) Tile's base position in atlas (texel space) ---
    int tilesPerRow = int(uAtlasSize.x / uTileSize);
    int tileX       = tileIndex % tilesPerRow;
    int tileY       = tileIndex / tilesPerRow;
    vec2 tileBase   = vec2(float(tileX), float(tileY)) * uTileSize; // texel coords

    // --- 3) Local position inside the tile ---
    vec2 tilePos01   = fract(mapPos);             // [0..1) inside current map cell
    vec2 localTexel  = tilePos01 * uTileSize;     // [0..tileSize) in texels (continuous)

    // --- 4) AA "pixel snap" in local tile texel space ---
    // Use fwidth to adaptively blend toward the nearest texel center.
    vec2 fw       = max(fwidth(localTexel), vec2(1e-5));
    vec2 fracPart = fract(localTexel);
    vec2 blend    = clamp(fracPart / fw, 0.0, 1.0);
    vec2 localAA  = floor(localTexel) + blend + 0.5; // center of target texel

    // --- 5) Stay inside the tile to avoid bleeding into neighbors ---
    vec2 minPix = vec2(0.5 + GUARD);
    vec2 maxPix = vec2(uTileSize - 0.5 - GUARD);
    vec2 localClamped = clamp(localAA, minPix, maxPix);

    // Tiles in your original shader were addressed with a flipped Y (1.0 - tilePos.y)
    // so flip the local texel Y here in texel space.
    float flippedY  = (uTileSize - localClamped.y);
    vec2  atlasTexel = tileBase + vec2(localClamped.x, flippedY);

    // --- 6) Compute stable gradients for LOD from the *continuous* pre-snap coords ---
    // Derivatives from localTexel (not localAA) keep LOD smooth while the sample is snapped.
    vec2 dLocaldx = dFdx(localTexel);
    vec2 dLocaldy = dFdy(localTexel);

    // Convert texel gradients to atlas UV gradients.
    // Note Y flip: texel +y becomes UV -y.
    vec2 texelToUV = 1.0 / uAtlasSize;
    vec2 gradX = dLocaldx * texelToUV * vec2(1.0, -1.0);
    vec2 gradY = dLocaldy * texelToUV * vec2(1.0, -1.0);

    // --- 7) Sample the atlas ---
    vec2 atlasUV = atlasTexel / uAtlasSize;
    FragColor = textureGrad(uTileAtlas, atlasUV, gradX, gradY);
}
  """

  # Compile shader
  tileMap.shader = newShader(
    ("vertex", vertexShaderSource),
    ("fragment", fragmentShaderSource)
  )

  # Quad vertices (position + texture coordinates)
  tileMap.quadVertices = @[
    # positions    # texture coords
    -1.0f,  1.0f,   0.0f, 0.0f,  # top left
    -1.0f, -1.0f,   0.0f, 1.0f,  # bottom left
    1.0f, -1.0f,   1.0f, 1.0f,  # bottom right
    1.0f,  1.0f,   1.0f, 0.0f   # top right
  ]

  tileMap.quadIndices = @[
    0u32, 1u32, 2u32,  # first triangle
    0u32, 2u32, 3u32   # second triangle
  ]

  # Create VAO, VBO, EBO
  glGenVertexArrays(1, tileMap.VAO.addr)
  glGenBuffers(1, tileMap.VBO.addr)
  glGenBuffers(1, tileMap.EBO.addr)

  glBindVertexArray(tileMap.VAO)

  glBindBuffer(GL_ARRAY_BUFFER, tileMap.VBO)
  glBufferData(GL_ARRAY_BUFFER, tileMap.quadVertices.len * sizeof(float32), tileMap.quadVertices[0].addr, GL_STATIC_DRAW)

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, tileMap.EBO)
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, tileMap.quadIndices.len * sizeof(uint32), tileMap.quadIndices[0].addr, GL_STATIC_DRAW)

  # Position attribute
  glVertexAttribPointer(
    0,
    2,
    cGL_FLOAT,
    GL_FALSE,
    4 * sizeof(float32),
    cast[pointer](0)
  )
  glEnableVertexAttribArray(0)

  # Texture coordinate attribute
  glVertexAttribPointer(
    1,
    2,
    cGL_FLOAT,
    GL_FALSE,
    4 * sizeof(float32),
    cast[pointer](2 * sizeof(float32))
  )
  glEnableVertexAttribArray(1)

  glBindVertexArray(0)


proc draw(tileMap: TileMap, mvp: Mat4) =
  # Use our custom shader
  glUseProgram(tileMap.shader.programId)

  # Set uniforms
  tileMap.shader.setUniform("uMVP", mvp)
  tileMap.shader.setUniform("uMapSize", vec2(MAP_SIZE.float32, MAP_SIZE.float32))
  tileMap.shader.setUniform("uAtlasSize", vec2(tileMap.tileAtlas.width.float32, tileMap.tileAtlas.height.float32))
  tileMap.shader.setUniform("uTileSize", 64.0f)  # Tile size in pixels.

  tileMap.shader.bindUniforms()

  # Bind textures
  glActiveTexture(GL_TEXTURE0)
  glBindTexture(GL_TEXTURE_2D, tileMap.indexTexture)
  tileMap.shader.setUniform("uIndexTexture", 0)

  glActiveTexture(GL_TEXTURE1)
  glBindTexture(GL_TEXTURE_2D, tileMap.tileAtlasTexture)
  tileMap.shader.setUniform("uTileAtlas", 1)

  tileMap.shader.bindUniforms()

  # Draw the quad
  glBindVertexArray(tileMap.VAO)
  glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nil)
  glBindVertexArray(0)


terrainMap.setupGPU()

# Called when it is time to draw a new frame.
window.onFrame = proc() =
  # Clear the screen and begin a new frame.
  bxy.beginFrame(window.size)

  glClearColor(0.0, 0.0, 0.0, 1.0)
  glClear(GL_COLOR_BUFFER_BIT)

  # Handle input for panning and zooming like hex example
  # Left mouse button: drag to pan
  # Mouse wheel: zoom in/out
  if window.buttonDown[MouseLeft]:
    vel = window.mouseDelta.vec2 + vel * 0.1
  else:
    vel *= 0.99

  pos += vel

  if window.scrollDelta.y != 0:
    zoomVel = window.scrollDelta.y * 0.005
    echo "pos: ", pos, " zoom: ", zoom
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
             scale(vec3(zoomPow2 * MAP_SIZE.float32/2, zoomPow2 * MAP_SIZE.float32/2, 1.0f))
  let mvp = projection * view


  terrainMap.draw(mvp)

  # End this frame, flushing the draw commands.
  bxy.endFrame()

  # Swap buffers displaying the new Boxy frame.
  window.swapBuffers()
  inc frame

while not window.closeRequested:
  pollEvents()
