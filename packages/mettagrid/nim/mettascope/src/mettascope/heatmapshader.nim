import
  std/[math, strutils],
  opengl, boxy/[shaders], shady, vmath, pixie,
  heatmap, common

var
  uMvp: Uniform[Mat4]
  uMapSize: Uniform[Vec2]
  uMaxHeat: Uniform[float32]
  uMinOpacity: Uniform[float32]
  uMaxOpacity: Uniform[float32]
  heatmapTexture: Uniform[Sampler2D]

var
  heatmapShader: Shader
  heatmapVao: GLuint
  heatmapTextureId: GLuint

proc heatmapVert*(fragmentWorldPos: var Vec2) =
  ## Generate a full-rect in world units to align heatmap with tile grid.
  let corner = ivec2(gl_VertexID mod 2, gl_VertexID div 2)
  let worldPos = vec2(
    float(corner.x) * uMapSize.x - 0.5,
    float(corner.y) * uMapSize.y - 0.5
  )
  fragmentWorldPos = worldPos
  gl_Position = uMvp * vec4(worldPos.x, worldPos.y, 0.0f, 1.0f)

proc heatmapFrag*(fragmentWorldPos: Vec2, FragColor: var Vec4) =
  ## Sample heatmap texture and convert to thermal colors.
  # Get integer tile coordinates, clamped to valid range.
  let tileX = clamp(floor(fragmentWorldPos.x + 0.5), 0.0, uMapSize.x - 1.0)
  let tileY = clamp(floor(fragmentWorldPos.y + 0.5), 0.0, uMapSize.y - 1.0)
  # Sample at texel center: (tile + 0.5) / size.
  let heatmapCoord = vec2((tileX + 0.5) / uMapSize.x, (tileY + 0.5) / uMapSize.y)
  let heatSample = texture(heatmapTexture, heatmapCoord)
  let heat = heatSample.r * 255.0

  # Default to transparent.
  var r: float32 = 0.0
  var g: float32 = 0.0
  var b: float32 = 0.0
  var opacity: float32 = 0.0

  if heat > 0.0 and uMaxHeat > 0.0:
    let normalizedHeat = heat / uMaxHeat
    let scaledHeat = sqrt(normalizedHeat)

    if scaledHeat < 0.5:
      # Blue to yellow transition.
      let t = scaledHeat * 2.0
      r = max((t - 0.5) * 1.6, 0.0)
      g = t * 0.8
      b = (1.0 - t) * 0.7
    else:
      # Yellow to red transition.
      let t = (scaledHeat - 0.5) * 2.0
      r = 0.8 + t * 0.2
      g = 0.8 * (1.0 - t)
      b = 0.0

    opacity = uMinOpacity * 0.5 + scaledHeat * (uMaxOpacity * 0.6)

  FragColor = vec4(r, g, b, opacity)

proc initHeatmapShader*() =
  ## Initialize the heatmap shader module.
  when defined(emscripten):
    heatmapShader = newShader(
      (
        "heatmapVert",
        toGLSL(heatmapVert, "300 es", "precision highp float;\n")
          .replace("uint(2)", "2")
          .replace("mod(gl_VertexID, 2)", "gl_VertexID % 2")
      ),
      (
        "heatmapFrag",
        toGLSL(heatmapFrag, "300 es", "precision highp float;\n")
      )
    )
  else:
    heatmapShader = newShader(
      ("heatmapVert", toGLSL(heatmapVert, "410", "")),
      ("heatmapFrag", toGLSL(heatmapFrag, "410", ""))
    )

  # Create an empty VAO; vertices are generated from gl_VertexID.
  glGenVertexArrays(1, heatmapVao.addr)
  glBindVertexArray(heatmapVao)
  glBindVertexArray(0)

  # Create texture for heatmap data (single channel).
  glGenTextures(1, heatmapTextureId.addr)
  glActiveTexture(GL_TEXTURE0)
  glBindTexture(GL_TEXTURE_2D, heatmapTextureId)
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST.GLint)
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST.GLint)
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE.GLint)
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE.GLint)
  glBindTexture(GL_TEXTURE_2D, 0)

proc updateTexture*(heatmap: Heatmap, step: int) =
  ## Upload heatmap data for the given step to the texture.
  if step == heatmap.currentTextureStep:
    return # Already up to date

  heatmap.currentTextureStep = step

  # Prepare heatmap data as uint8 array.
  # Store directly without Y flip - texture coordinates will handle mapping.
  var heatmapData: seq[uint8]
  heatmapData.setLen(heatmap.width * heatmap.height)

  for y in 0 ..< heatmap.height:
    for x in 0 ..< heatmap.width:
      let heat = heatmap.getHeat(step, x, y)
      # Clamp heat to 0-255 range for texture storage.
      let clampedHeat = min(heat, 255).uint8
      # Store row-major: row y, column x.
      heatmapData[y * heatmap.width + x] = clampedHeat

  # Upload to texture.
  glActiveTexture(GL_TEXTURE0)
  glBindTexture(GL_TEXTURE_2D, heatmapTextureId)
  # Set alignment to 1 for tightly packed data (no padding between rows)
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
  glTexImage2D(
    GL_TEXTURE_2D,
    0,
    GL_R8.GLint, # Single channel texture
    heatmap.width.GLint,
    heatmap.height.GLint,
    0,
    GL_RED,
    GL_UNSIGNED_BYTE,
    heatmapData[0].addr
  )
  glBindTexture(GL_TEXTURE_2D, 0)

proc draw*(
  heatmap: Heatmap,
  mvp: Mat4,
  mapSize: Vec2,
  maxHeat: float32,
  minOpacity: float32 = 0.4,
  maxOpacity: float32 = 0.8
) =
  ## Draw the heatmap overlay.
  if maxHeat <= 0.0:
    return

  glUseProgram(heatmapShader.programId)
  heatmapShader.setUniform("uMvp", mvp)
  heatmapShader.setUniform("uMapSize", mapSize)
  heatmapShader.setUniform("uMaxHeat", maxHeat)
  heatmapShader.setUniform("uMinOpacity", minOpacity)
  heatmapShader.setUniform("uMaxOpacity", maxOpacity)
  glActiveTexture(GL_TEXTURE0)
  glBindTexture(GL_TEXTURE_2D, heatmapTextureId)
  heatmapShader.setUniform("heatmapTexture", 0)
  heatmapShader.bindUniforms()

  # Enable blending for transparency.
  glEnable(GL_BLEND)
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

  glBindVertexArray(heatmapVao)
  glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
  glBindVertexArray(0)

  glDisable(GL_BLEND)
  glUseProgram(0)
