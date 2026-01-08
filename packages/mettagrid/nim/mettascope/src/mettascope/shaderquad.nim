import
  std/[math, strutils],
  opengl, boxy/[shaders], shady, vmath, pixie

## Drawer for a single quad covering the map with texture and shader.

type
  ShaderQuad* = ref object
    shader: Shader
    vao: GLuint
    texture: GLuint
    image: Image
    tilesPerImage: Vec2

var
  uMvp: Uniform[Mat4]
  uMapSize: Uniform[Vec2]
  uTileSize: Uniform[Vec2]
  uGridColor: Uniform[Vec4]
  # Texture uniforms (match pixelator naming for sampling helpers)
  atlasSize: Uniform[Vec2]
  atlas: Uniform[Sampler2D]
  uTilesPerImage: Uniform[Vec2]

proc gridVert*(fragmentWorldPos: var Vec2) =
  ## Generate a full-rect in world units from (0,0) to (mapSize.x, mapSize.y).
  let corner = ivec2(gl_VertexID mod 2, gl_VertexID div 2)
  let worldPos = vec2(float(corner.x) * uMapSize.x, float(corner.y) * uMapSize.y)
  fragmentWorldPos = worldPos
  gl_Position = uMvp * vec4(worldPos.x, worldPos.y, 0.0f, 1.0f)

proc gridFrag*(fragmentWorldPos: Vec2, FragColor: var Vec4) =
  ## Texture-based grid sampling with AA similar to pixelator.
  # Compute tile-space coordinates of the world position.
  let gridCoord = fragmentWorldPos / uTileSize
  # Repeat the texture every uTilesPerImage tiles.
  let texPhase = fract(gridCoord / uTilesPerImage)
  # Convert to pixel coordinates in the texture.
  let pixCoord = texPhase * atlasSize
  # Pixel-accurate AA in texture space (works with mipmaps too).
  let pixAA = floor(pixCoord) + min(fract(pixCoord) / fwidth(pixCoord), 1.0f) - 0.5f
  # Sample and tint.
  FragColor = texture(atlas, pixAA / atlasSize) * uGridColor * uGridColor.a

proc newGridQuad*(imagePath: string, tilesX, tilesY: int): ShaderQuad =
  ## Creates a new GridQuad.
  result = ShaderQuad()
  result.image = readImage(imagePath)
  result.tilesPerImage = vec2(tilesX.float32, tilesY.float32)

  when defined(emscripten):
    result.shader = newShader(
      (
        "gridVert",
        toGLSL(gridVert, "300 es", "precision highp float;\n")
          .replace("uint(2)", "2")
          .replace("mod(gl_VertexID, 2)", "gl_VertexID % 2")
      ),
      (
        "gridFrag",
        toGLSL(gridFrag, "300 es", "precision highp float;\n")
      )
    )
  else:
    result.shader = newShader(
      ("gridVert", toGLSL(gridVert, "410", "")),
      ("gridFrag", toGLSL(gridFrag, "410", ""))
    )

  # Upload texture to GL and generate mipmaps.
  glGenTextures(1, result.texture.addr)
  glActiveTexture(GL_TEXTURE0)
  glBindTexture(GL_TEXTURE_2D, result.texture)
  glTexImage2D(
    GL_TEXTURE_2D,
    0,
    GL_RGBA8.GLint,
    result.image.width.GLint,
    result.image.height.GLint,
    0,
    GL_RGBA,
    GL_UNSIGNED_BYTE,
    cast[pointer](result.image.data[0].addr)
  )
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR.GLint)
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR.GLint)
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT.GLint)
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT.GLint)
  glGenerateMipmap(GL_TEXTURE_2D)
  glBindTexture(GL_TEXTURE_2D, 0)

  # Create an empty VAO; vertices are generated from gl_VertexID.
  glGenVertexArrays(1, result.vao.addr)
  glBindVertexArray(result.vao)
  glBindVertexArray(0)

proc draw*(
  sq: ShaderQuad,
  mvp: Mat4,
  mapSize: Vec2,
  tileSize: Vec2,
  gridColor: Vec4,
  lineWidthPx: float32 = 1.0f
) =
  ## Draw the grid quad.
  if sq.isNil:
    return

  glUseProgram(sq.shader.programId)
  sq.shader.setUniform("uMvp", mvp)
  sq.shader.setUniform("uMapSize", mapSize)
  sq.shader.setUniform("uTileSize", tileSize)
  sq.shader.setUniform("uGridColor", gridColor)
  sq.shader.setUniform("atlasSize", vec2(sq.image.width.float32, sq.image.height.float32))
  sq.shader.setUniform("uTilesPerImage", sq.tilesPerImage)
  glActiveTexture(GL_TEXTURE0)
  glBindTexture(GL_TEXTURE_2D, sq.texture)
  sq.shader.setUniform("atlas", 0)
  sq.shader.bindUniforms()

  glBindVertexArray(sq.vao)
  glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
  glBindVertexArray(0)
  glUseProgram(0)
