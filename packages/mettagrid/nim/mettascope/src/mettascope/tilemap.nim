import pixie, opengl, boxy/shaders

type
  TileMap* = ref object
    width*: int
    height*: int
    tileSize*: int
    tileAtlas*: Image
    indexData*: seq[uint8]

    # OpenGL textures
    indexTexture: GLuint
    tileAtlasTextureArray: GLuint
    overworldTexture: GLuint
    shader: Shader
    VAO: GLuint
    VBO: GLuint
    EBO: GLuint
    quadVertices: seq[float32]
    quadIndices: seq[uint32]

proc newTileMap*(
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

proc getTile*(tileMap: TileMap, x: int, y: int): int =
  if x < 0 or y < 0 or x >= tileMap.width or y >= tileMap.height:
    return -1
  return tileMap.indexData[y * tileMap.width + x].int

proc averageColor(img: Image): ColorRGBX =
  ## Returns the average color of the image.
  var
    r, g, b, a: float32 = 0
  let m = img.width.float32 * img.height.float32
  for y in 0 ..< img.height:
    for x in 0 ..< img.width:
      r += img.unsafe[x, y].r.float32 / m
      g += img.unsafe[x, y].g.float32 / m
      b += img.unsafe[x, y].b.float32 / m
      a += img.unsafe[x, y].a.float32 / m
  return rgbx(
    floor(r).uint8,
    floor(g).uint8,
    floor(b).uint8,
    floor(a).uint8,
  )

proc setupGPU*(tileMap: TileMap) =
  ## Setup the GPU for the tile map.

  var maxLayers: GLint
  glGetIntegerv(GL_MAX_ARRAY_TEXTURE_LAYERS, maxLayers.addr)
  doAssert maxLayers >= 256, "Layer count must be at least 256 for tile atlas."

  # Create OpenGL texture for tile indices
  glGenTextures(1, tileMap.indexTexture.addr)
  glBindTexture(GL_TEXTURE_2D, tileMap.indexTexture)
  glTexImage2D(
    GL_TEXTURE_2D,
    0,
    GL_R8UI.GLint,
    tileMap.width.GLint,
    tileMap.height.GLint,
    0,
    GL_RED_INTEGER,
    GL_UNSIGNED_BYTE,
    tileMap.indexData[0].addr
  )
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST.GLint)
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST.GLint)

  # Create tile atlas texture array, mipmapped.
  glGenTextures(1, tileMap.tileAtlasTextureArray.addr)
  glBindTexture(GL_TEXTURE_2D_ARRAY, tileMap.tileAtlasTextureArray)
  # Determine tile array geometry and mip levels
  let tilesPerRow = tileMap.tileAtlas.width div tileMap.tileSize
  let tilesPerCol = tileMap.tileAtlas.height div tileMap.tileSize
  let layerCount = tilesPerRow * tilesPerCol
  var mipLevels = 1
  var mw = tileMap.tileSize
  var mh = tileMap.tileSize
  while mw > 1 or mh > 1:
    if mw > 1: mw = mw div 2
    if mh > 1: mh = mh div 2
    inc mipLevels

  glTexStorage3D(
    target = GL_TEXTURE_2D_ARRAY,
    levels = mipLevels.GLsizei,
    internalFormat = GL_RGBA8,
    width = tileMap.tileSize.GLsizei,
    height = tileMap.tileSize.GLsizei,
    depth = layerCount.GLsizei,
  )
  glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR.GLint)
  glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR.GLint)
  glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE.GLint)
  glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE.GLint)
  glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE.GLint)
  glGenerateMipmap(GL_TEXTURE_2D_ARRAY)

  # Create overworld texture (RGBA8, mipmapped), same size as index texture.
  glGenTextures(1, tileMap.overworldTexture.addr)
  glBindTexture(GL_TEXTURE_2D, tileMap.overworldTexture)
  glTexImage2D(
    GL_TEXTURE_2D,
    0,
    GL_RGBA8.GLint,
    tileMap.width.GLint,
    tileMap.height.GLint,
    0,
    GL_RGBA,
    GL_UNSIGNED_BYTE,
    nil
  )
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR.GLint)
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR.GLint)
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE.GLint)
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE.GLint)

  # Fill the texture array with the tile atlas.
  var layer = 0
  var avgColors: seq[ColorRGBX]
  for ty in 0 ..< tilesPerCol:
    for tx in 0 ..< tilesPerRow:
      let sx = tx * tileMap.tileSize
      let sy = ty * tileMap.tileSize
      let subImg = subImage(tileMap.tileAtlas, sx, sy, tileMap.tileSize, tileMap.tileSize)
      glTexSubImage3D(
        GL_TEXTURE_2D_ARRAY,
        0.GLint,                       # level
        0.GLint, 0.GLint, layer.GLint, # xoffset, yoffset, zoffset (layer)
        tileMap.tileSize.GLsizei,
        tileMap.tileSize.GLsizei,
        1.GLsizei,                     # depth (one layer)
        GL_RGBA,
        GL_UNSIGNED_BYTE,
        cast[pointer](subImg.data[0].addr)
      )
      avgColors.add(subImg.averageColor())
      inc layer
  glGenerateMipmap(GL_TEXTURE_2D_ARRAY)

  # Fill overworld texture with average color per tile.
  var owData = newImage(tileMap.width, tileMap.height)
  for y in 0 ..< tileMap.height:
    for x in 0 ..< tileMap.width:
      let tileIndex = tileMap.indexData[y * tileMap.width + x].int
      owData.unsafe[x, y] = avgColors[tileIndex]

  glTexSubImage2D(
    GL_TEXTURE_2D,
    0,
    0,
    0,
    tileMap.width.GLsizei,
    tileMap.height.GLsizei,
    GL_RGBA,
    GL_UNSIGNED_BYTE,
    owData.data[0].addr
  )
  glGenerateMipmap(GL_TEXTURE_2D)

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

in vec2 TexCoord;    // 0..1 over the whole tilemap (screen mapping)
out vec4 FragColor;

// --- Textures ---
// Use an integer index map if you can (R8UI or R16UI). No filtering, no mipmaps.
uniform usampler2D    uIndexTexture;  // tile ids [0 .. layerCount-1]

// Each layer is ONE tile image. All layers must be the same dimensions (uTileSize x uTileSize).
uniform sampler2DArray uTileArray;
uniform sampler2D      uOverworld;

// --- Map / tile geometry ---
uniform vec2  uMapSize;   // map size in tiles, e.g. (256, 256)
uniform float uTileSize;  // tile size in texels, e.g. 16
uniform float uZoom;      // current zoom (screen->world scale)
uniform float uZoomThreshold; // switch to overworld when zoom below this

void main()
{
    // 0) Switch to overworld if zoomed out enough
    if (uZoom < uZoomThreshold) {
        // Sample overworld at map texel resolution
        vec2 mapUV = TexCoord; // TexCoord already spans 0..1 across map
        FragColor = texture(uOverworld, mapUV);
        return;
    }

    // 1) Which map cell are we in?
    vec2  mapPos   = TexCoord * uMapSize;        // [0..mapW/H) in tile units
    ivec2 mapTexel = ivec2(floor(mapPos));       // integer tile coords

    // 2) Read tile index EXACTLY (no filtering, no mips)
    uint tileIndexU = texelFetch(uIndexTexture, mapTexel, 0).r;
    int  tileIndex  = int(tileIndexU);

    // 3) Local coordinates inside this tile (continuous + fractional)
    vec2 tilePos01   = fract(mapPos);                 // [0..1) within the tile cell
    vec2 localTexel  = tilePos01 * uTileSize;         // [0..tileSize) in texels (continuous)
    vec2 contTexel   = TexCoord * (uMapSize * uTileSize); // continuous (no fract) for stable derivatives

    // 4) Anti-aliased "nearest" snap in tile texel space
    vec2 fw       = max(fwidth(contTexel), vec2(1e-5)); // stable across tile seams
    vec2 fracPart = fract(localTexel);
    vec2 blend    = clamp(fracPart / fw, 0.0, 1.0);
    vec2 localAA  = floor(localTexel) + blend + 0.5;    // center of target texel

    // 5) Clamp to valid texel centers
    vec2 localClamped = clamp(localAA, vec2(0.5), vec2(uTileSize - 0.5));

    // 6) Convert to per-layer UVs in [0,1], flipping Y if your source images are top-left origin
    // If your tiles are already bottom-left origin, drop the flip and just do localClamped / uTileSize.
    vec2 layerUV = vec2(
        localClamped.x / uTileSize,
        (uTileSize - localClamped.y) / uTileSize   // flip Y
    );

    // 7) Stable LOD: gradients from the continuous coords, scaled to layer UV space
    vec2 dUVdx = (dFdx(contTexel) / uTileSize) * vec2(1.0, -1.0);
    vec2 dUVdy = (dFdy(contTexel) / uTileSize) * vec2(1.0, -1.0);

    // 8) Sample the tile layer
    FragColor = textureGrad(uTileArray, vec3(layerUV, float(tileIndex)), dUVdx, dUVdy);
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

proc draw*(
  tileMap: TileMap,
  mvp: Mat4,
  zoom: float32,
  zoomThreshold = 1.25f
) =
  # Use our custom shaderf
  glUseProgram(tileMap.shader.programId)

  # Set uniforms
  tileMap.shader.setUniform("uMVP", mvp)
  tileMap.shader.setUniform("uMapSize", vec2(tileMap.width.float32, tileMap.height.float32))
  tileMap.shader.setUniform("uTileSize", 64.0f)  # Tile size in pixels.
  tileMap.shader.setUniform("uZoom", zoom)
  tileMap.shader.setUniform("uZoomThreshold", zoomThreshold)

  tileMap.shader.bindUniforms()

  # Bind textures
  glActiveTexture(GL_TEXTURE0)
  glBindTexture(GL_TEXTURE_2D, tileMap.indexTexture)
  tileMap.shader.setUniform("uIndexTexture", 0)

  glActiveTexture(GL_TEXTURE1)
  glBindTexture(GL_TEXTURE_2D_ARRAY, tileMap.tileAtlasTextureArray)
  tileMap.shader.setUniform("uTileArray", 1)

  glActiveTexture(GL_TEXTURE2)
  glBindTexture(GL_TEXTURE_2D, tileMap.overworldTexture)
  tileMap.shader.setUniform("uOverworld", 2)

  tileMap.shader.bindUniforms()

  # Draw the quad
  glBindVertexArray(tileMap.VAO)
  glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nil)
  glBindVertexArray(0)
