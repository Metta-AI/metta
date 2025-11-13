import
  std/[math],
  opengl, boxy/[shaders], shady, vmath

# Draw a single quad covering the map and render the grid inside a shader.

type
  ShaderQuad* = ref object
    shader: Shader
    vao: GLuint

var
  uMvp: Uniform[Mat4]
  uMapSize: Uniform[Vec2]
  uTileSize: Uniform[Vec2]
  uGridColor: Uniform[Vec4]

proc gridVert*(fragmentWorldPos: var Vec2) =
  # Generate a full-rect in world units from (0,0) to (mapSize.x, mapSize.y).
  let corner = uvec2(gl_VertexID mod 2, gl_VertexID div 2)
  let worldPos = vec2(float(corner.x) * uMapSize.x, float(corner.y) * uMapSize.y)
  fragmentWorldPos = worldPos
  gl_Position = uMvp * vec4(worldPos.x, worldPos.y, 0.0f, 1.0f)

proc gridFrag*(fragmentWorldPos: Vec2, FragColor: var Vec4) =
  # Compute grid intensity using AA in screen space via fwidth.
  # Convert to tile-space so lines repeat every tile.
  let gridCoord = fragmentWorldPos / uTileSize
  let gx = gridCoord.x
  let gy = gridCoord.y

  if gx.fract < 0.01f or gy.fract < 0.01f:
    FragColor = uGridColor
  else:
    FragColor = vec4(0.0f, 0.0f, 0.0f, 0.0f)


  # # Distance to nearest vertical and horizontal grid line in tile-space.
  # let distX = min(fract(gx), 1.0f - fract(gx))
  # let distY = min(fract(gy), 1.0f - fract(gy))

  # # Screen-space derivative of tile coordinates (tiles per pixel).
  # let scale = fwidth(gridCoord)
  # let scaleX = scale.x
  # let scaleY = scale.y

  # # Convert desired pixel line width to tile-space width using derivatives.
  # # Use smoothstep for antialiased hard lines.
  # let lineX = 1.0f - smoothstep(0.0f, 1 * scaleX, distX)
  # let lineY = 1.0f - smoothstep(0.0f, 1 * scaleY, distY)
  # let alpha = max(lineX, lineY)

  # FragColor = vec4(uGridColor.rgb, uGridColor.a * alpha)

proc newGridQuad*(): ShaderQuad =
  ## Creates a new ShaderQuad.
  result = ShaderQuad()

  when defined(emscripten):
    result.shader = newShader(
      (
        "gridVert",
        toGLSL(gridVert, "300 es", "precision highp float;\n")
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
  sq.shader.bindUniforms()

  glBindVertexArray(sq.vao)
  glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
  glBindVertexArray(0)
  glUseProgram(0)
