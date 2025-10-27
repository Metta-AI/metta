import
  std/[os, json, strutils, tables],
  pixie, opengl, boxy/shaders, boxy/allocator, jsony

# This file specifically deals with the pixel atlas texture.
# It supports pixel art style drawing with pixel perfect AA sampling.
# It is used to draw the objects in the mettascope.
# It also supports animations.

type

  Entry* = object
    x*: int
    y*: int
    width*: int
    height*: int
    animationFrames*: int

  PixelAtlas* = ref object
    size*: int
    entries*: Table[string, Entry]

proc generatePixelAtlas*(
  size: int,
  dirsToScan: seq[string],
  outputImagePath: string,
  outputJsonPath: string
) =
  let atlasImage = newImage(size, size)
  let atlas = PixelAtlas(size: size)
  let allocator = newSkylineAllocator(size, 0)

  for dir in dirsToScan:
    for file in walkDir(dir):
      if file.path.endsWith(".png"):
        let image = readImage(file.path)
        let allocation = allocator.allocate(image.width, image.height)
        if allocation.success:
          atlasImage.draw(
            image,
            translate(vec2(allocation.x.float32, allocation.y.float32)),
            OverwriteBlend
          )
        else:
          echo "Failed to allocate space for ", file.path
          echo "You need to increase the size of the atlas"
          quit(1)
        let entry = Entry(
          x: allocation.x,
          y: allocation.y,
          width: image.width,
          height: image.height,
          animationFrames: 1
        )
        atlas.entries[file.path] = entry

  atlasImage.writeFile(outputImagePath)
  writeFile(outputJsonPath, atlas.toJson())


type
  Pixalator* = ref object
    atlas: PixelAtlas
    image: Image
    shader: Shader
    vao: GLuint              ## Vertex array object
    instanceVbo: GLuint      ## Per-instance x,y,w,h buffer (uint16 * 4)
    atlasTexture: GLuint     ## GL texture for the atlas image
    instanceData: seq[uint16]
    instanceCount: int
    currentUvRect: array[4, float32]  ## sx, sy, sw, sh in atlas pixels

var
  # kept to avoid breaking imports if referenced elsewhere
  gl_Position*: Vec4

proc newPixalator*(
  imagePath: string,
  jsonPath: string
): Pixalator =
  result = Pixalator()
  result.image = readImage(imagePath)
  result.atlas = readFile(jsonPath).fromJson(PixelAtlas)
  result.instanceData = @[]
  result.instanceCount = 0
  result.currentUvRect = [0f, 0f, 0f, 0f]

  # GLSL 410 core, integer instanced attribute expands to a quad via gl_VertexID
  let vertexShaderSource = """
#version 410 core

// Per-instance rectangle in pixels: x, y, w, h
layout (location = 0) in uvec4 aRect;

out vec2 vUv;

uniform mat4 uProj;
uniform vec2 uAtlasSize;            // in pixels
uniform vec4 uUvRect;               // sx, sy, sw, sh in pixels

void main() {
    // Corner from gl_VertexID for triangle strip (0..3):
    // 0:(0,0) 1:(1,0) 2:(0,1) 3:(1,1)
    uvec2 corner = uvec2(gl_VertexID & 1, gl_VertexID >> 1);

    // Destination position in pixels
    float dx = float(aRect.x) + float(corner.x) * float(aRect.z);
    float dy = float(aRect.y) + float(corner.y) * float(aRect.w);

    gl_Position = uProj * vec4(dx, dy, 0.0, 1.0);

    // Source UVs from atlas rect, convert to [0,1]
    float sx = uUvRect.x + float(corner.x) * uUvRect.z;
    float sy = uUvRect.y + float(corner.y) * uUvRect.w;
    vUv = vec2(sx, sy) / uAtlasSize;
}
  """

  let fragmentShaderSource = """
#version 410 core

in vec2 vUv;
out vec4 FragColor;

uniform sampler2D uAtlas;

void main() {
    FragColor = texture(uAtlas, vUv);
}
  """

  result.shader = newShader(("vertex", vertexShaderSource), ("fragment", fragmentShaderSource))

  # Upload atlas image to GL texture
  glGenTextures(1, result.atlasTexture.addr)
  glActiveTexture(GL_TEXTURE0)
  glBindTexture(GL_TEXTURE_2D, result.atlasTexture)
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
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST.GLint)
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST.GLint)
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE.GLint)
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE.GLint)

  # Set up VAO and instance buffer (aRect)
  glGenVertexArrays(1, result.vao.addr)
  glBindVertexArray(result.vao)

  glGenBuffers(1, result.instanceVbo.addr)
  glBindBuffer(GL_ARRAY_BUFFER, result.instanceVbo)
  glBufferData(GL_ARRAY_BUFFER, 0, nil, GL_STREAM_DRAW)  # will resize each frame

  # location = 0, 4 components of unsigned short, integer attribute, divisor = 1
  glEnableVertexAttribArray(0)
  glVertexAttribIPointer(0, 4, GL_UNSIGNED_SHORT, 4 * sizeof(uint16), cast[pointer](0))
  glVertexAttribDivisor(0, 1)

  # Unbind
  glBindBuffer(GL_ARRAY_BUFFER, 0)
  glBindVertexArray(0)


proc drawSprite*(
  pixalator: Pixalator,
  name: string,
  x, y: uint16
) =
  if name notin pixalator.atlas.entries:
    raise newException(ValueError, "Sprite not found in atlas: " & name)
  let uv = pixalator.atlas.entries[name]
  pixalator.instanceData.add(x.uint16)
  pixalator.instanceData.add(y.uint16)
  pixalator.instanceData.add(uv.x.uint16)
  pixalator.instanceData.add(uv.y.uint16)
  pixalator.instanceData.add(uv.width.uint16)
  pixalator.instanceData.add(uv.height.uint16)
  inc pixalator.instanceCount

proc clear*(pixalator: Pixalator) =
  ## Clears the current instance queue.
  pixalator.instanceData.setLen(0)
  pixalator.instanceCount = 0

proc draw*(
  pixalator: Pixalator,
  proj: Mat4
) =
  ## Draw all queued instances for the current sprite.
  if pixalator.instanceCount == 0:
    return

  # Upload instance buffer
  glBindBuffer(GL_ARRAY_BUFFER, pixalator.instanceVbo)
  let byteLen = pixalator.instanceData.len * sizeof(uint16)
  glBufferData(GL_ARRAY_BUFFER, byteLen, pixalator.instanceData[0].addr, GL_STREAM_DRAW)

  # Bind state
  glUseProgram(pixalator.shader.programId)
  pixalator.shader.setUniform("uProj", proj)
  pixalator.shader.setUniform("uAtlasSize", vec2(pixalator.image.width.float32, pixalator.image.height.float32))
  pixalator.shader.setUniform("uUvRect", vec4(pixalator.currentUvRect[0], pixalator.currentUvRect[1], pixalator.currentUvRect[2], pixalator.currentUvRect[3]))
  glActiveTexture(GL_TEXTURE0)
  glBindTexture(GL_TEXTURE_2D, pixalator.atlasTexture)
  pixalator.shader.setUniform("uAtlas", 0)
  pixalator.shader.bindUniforms()

  glBindVertexArray(pixalator.vao)

  # Draw 4-vertex triangle strip per instance (expanded in vertex shader)
  glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, pixalator.instanceCount.GLsizei)

  # Unbind minimal state
  glBindVertexArray(0)
  glUseProgram(0)
  glBindTexture(GL_TEXTURE_2D, 0)

  # Reset queue for next frame if desired
  pixalator.clear()
