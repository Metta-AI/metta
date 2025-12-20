import
  std/[strutils],
  pixie, opengl, boxy/shaders, shady, vmath

type
  TileMap* = ref object
    width*: int
    height*: int
    tileSize*: int
    tileAtlas*: Image
    indexData*: seq[uint8]

    # Computed average colors for each tile.
    avgColors*: seq[ColorRGBX]
    overworldImage*: Image

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

var
  mvp: Uniform[Mat4]
  indexTexture: Uniform[USampler2D]
  tileArray: Uniform[Sampler2DArray]
  overworld: Uniform[Sampler2D]
  mapSize: Uniform[Vec2]
  tileSize: Uniform[float32]
  zoom: Uniform[float32]
  zoomThreshold: Uniform[float32]
  tint: Uniform[Vec4]

proc tileMapVert*(aPos: Vec2, vertexUv: Vec2, fragmentUv: var Vec2) =
  gl_Position = mvp * vec4(
    aPos.x * mapSize.x - 0.5,
    aPos.y * mapSize.y - 0.5,
    0.0,
    1.0
  )
  fragmentUv = vertexUv

proc tileMapFrag*(fragmentUv: Vec2, fragColor: var Vec4) =
  if zoom < zoomThreshold:
    # Use the overworld texture with mipmapping for higher zoom levels
    let mapUV = fragmentUv
    fragColor = texture(overworld, vec2(mapUV.x, 1.0 - mapUV.y))
  else:
    let
      # Compute the map cell coordinates.
      mapPos = fragmentUv * mapSize
      mapTexel0 = ivec2(mapPos.x.floor.int32, mapPos.y.floor.int32)
      mapTexel = ivec2(mapTexel0.x, mapSize.y.int32 - mapTexel0.y - 1)

      # Read the tile index from the index texture.
      tileIndexU = texelFetch(indexTexture, mapTexel, 0).x
      tileIndex = tileIndexU.int32

      # Local coordinates inside the tile, continuous and fractional.
      localTexel = fract(mapPos) * (tileSize) - 1.0
      contTexel = fragmentUv * (mapSize * tileSize)

      # Anti-aliasing the nearest snap in the tile space.
      fw = max(fwidth(contTexel), vec2(1e-5))
      fracPart = fract(localTexel)
      blend = clamp(fracPart / fw, 0.0, 1.0)
      localAA = floor(localTexel) + blend + 0.5

      # Clamp to the valid texel range.
      localClamped = clamp(localAA, 0.5, tileSize - 0.5)

      # Convert to per layer UVs. Flip Y coordinate to match texture array layout.
      layerUV = vec2(localClamped.x / tileSize, (tileSize - localClamped.y) / tileSize)

      # Stable LOD: gradient from the continuous coordinates, matched to UV space.
      dUVdx = (dFdx(contTexel) / tileSize) * vec2(1.0, -1.0)
      dUVdy = (dFdy(contTexel) / tileSize) * vec2(1.0, -1.0)

    fragColor = textureGrad(
      tileArray,
      vec3(layerUV.x, layerUV.y, float(tileIndex)),
      dUVdx,
      dUVdy
    ) * tint

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

  # Create OpenGL texture for tile indices.
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
  # Determine tile array geometry and mip levels.
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
  tileMap.avgColors = newSeq[ColorRGBX](layerCount)
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
      tileMap.avgColors.add(subImg.averageColor())
      inc layer
  glGenerateMipmap(GL_TEXTURE_2D_ARRAY)

  # Fill overworld texture with average color per tile.
  tileMap.overworldImage = newImage(tileMap.width, tileMap.height)
  for y in 0 ..< tileMap.height:
    for x in 0 ..< tileMap.width:
      let tileIndex = tileMap.indexData[y * tileMap.width + x].int
      tileMap.overworldImage.unsafe[x, y] = tileMap.avgColors[tileIndex]

  glTexSubImage2D(
    GL_TEXTURE_2D,
    0,
    0,
    0,
    tileMap.width.GLsizei,
    tileMap.height.GLsizei,
    GL_RGBA,
    GL_UNSIGNED_BYTE,
    tileMap.overworldImage.data[0].addr
  )
  glGenerateMipmap(GL_TEXTURE_2D)

  # Compile shader via shady.
  when defined(emscripten):
    tileMap.shader = newShader(
      (
        "tileMapVert",
        toGLSL(tileMapVert, "300 es", "precision highp float;\n")
      ),
      (
        "tileMapFrag",
        toGLSL(tileMapFrag, "300 es", "precision highp float;\n")
          .replace("uniform usampler2D", "uniform highp usampler2D")
          .replace("uniform sampler2DArray", "uniform highp sampler2DArray")
      )
    )
  else:
    tileMap.shader = newShader(
      ("tileMapVert", toGLSL(tileMapVert, "410", "")),
      ("tileMapFrag", toGLSL(tileMapFrag, "410", ""))
    )

  # Quad vertices (position + texture coordinates).
  tileMap.quadVertices = @[
    # Positions    # Texture coords
    0.0f,  1.0f,   0.0f, 0.0f,  # Top left.
    0.0f,  0.0f,   0.0f, 1.0f,  # Bottom left.
    1.0f,  0.0f,   1.0f, 1.0f,  # Bottom right.
    1.0f,  1.0f,   1.0f, 0.0f   # Top right.
  ]

  tileMap.quadIndices = @[
    0u32, 1u32, 2u32,  # First triangle.
    0u32, 2u32, 3u32   # Second triangle.
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

  # Position attribute.
  glVertexAttribPointer(
    0,
    2,
    cGL_FLOAT,
    GL_FALSE,
    4 * sizeof(float32),
    cast[pointer](0)
  )
  glEnableVertexAttribArray(0)

  # Texture coordinate attribute.
  glVertexAttribPointer(
    1,
    2,
    cGL_FLOAT,
    GL_FALSE,
    4 * sizeof(float32),
    cast[pointer](2 * sizeof(float32))
  )
  glEnableVertexAttribArray(1)

proc updateGPU*(tileMap: TileMap) =
  ## Update the GPU for the tile map.

  # Update the index texture.
  glBindTexture(GL_TEXTURE_2D, tileMap.indexTexture)
  glTexSubImage2D(
    GL_TEXTURE_2D,
    0,
    0,
    0,
    tileMap.width.GLsizei,
    tileMap.height.GLsizei,
    GL_RED_INTEGER,
    GL_UNSIGNED_BYTE,
    tileMap.indexData[0].addr
  )

  # Update the overworld texture.
  # Fill overworld texture with average color per tile.
  for y in 0 ..< tileMap.height:
    for x in 0 ..< tileMap.width:
      let tileIndex = tileMap.indexData[y * tileMap.width + x].int
      tileMap.overworldImage.unsafe[x, y] = tileMap.avgColors[tileIndex]
  glBindTexture(GL_TEXTURE_2D, tileMap.overworldTexture)
  glTexSubImage2D(
    GL_TEXTURE_2D,
    0,
    0,
    0,
    tileMap.width.GLsizei,
    tileMap.height.GLsizei,
    GL_RGBA,
    GL_UNSIGNED_BYTE,
    tileMap.overworldImage.data[0].addr
  )
  glGenerateMipmap(GL_TEXTURE_2D)

proc draw*(
  tileMap: TileMap,
  mvp: Mat4,
  zoom: float32,
  zoomThreshold = 1.25f,
  tint: Color = color(1, 1, 1, 1)
) =

  # Do not clear here; Boxy manages the target/FBO.
  # Use our custom shader.
  glUseProgram(tileMap.shader.programId)

  # Set uniforms
  tileMap.shader.setUniform("mvp", mvp)
  tileMap.shader.setUniform("mapSize", vec2(tileMap.width.float32, tileMap.height.float32))
  tileMap.shader.setUniform("tileSize", tileMap.tileSize.float32)  # Tile size in pixels.
  tileMap.shader.setUniform("zoom", zoom)
  tileMap.shader.setUniform("zoomThreshold", zoomThreshold)
  tileMap.shader.setUniform("tint", vec4(tint.r, tint.g, tint.b, tint.a))
  tileMap.shader.bindUniforms()

  # Bind textures.
  glActiveTexture(GL_TEXTURE0)
  glBindTexture(GL_TEXTURE_2D, tileMap.indexTexture)
  tileMap.shader.setUniform("indexTexture", 0)

  glActiveTexture(GL_TEXTURE1)
  glBindTexture(GL_TEXTURE_2D_ARRAY, tileMap.tileAtlasTextureArray)
  tileMap.shader.setUniform("tileArray", 1)

  glActiveTexture(GL_TEXTURE2)
  glBindTexture(GL_TEXTURE_2D, tileMap.overworldTexture)
  tileMap.shader.setUniform("overworld", 2)

  tileMap.shader.bindUniforms()

  # Bind our VAO (encapsulates attrib pointers and EBO).
  glBindVertexArray(tileMap.VAO)

  # Draw the quad.
  glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nil)

  # Restore minimal GL state so Boxy continues to work after exitRawOpenGLMode.
  # Unbind textures in reverse order
  glActiveTexture(GL_TEXTURE2)
  glBindTexture(GL_TEXTURE_2D, 0)
  glActiveTexture(GL_TEXTURE1)
  glBindTexture(GL_TEXTURE_2D_ARRAY, 0)
  glActiveTexture(GL_TEXTURE0)
  glBindTexture(GL_TEXTURE_2D, 0)

  # Unbind VAO (Boxy will restore its own in exitRawOpenGLMode).
  glBindVertexArray(0)

  # Unbind our shader program (Boxy will bind its own when needed).
  glUseProgram(0)
