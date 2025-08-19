import {
  getFont,
  getGlyph,
  getKerning,
  getSpriteBounds,
  getWhiteUV,
  hasSprite,
  loadAtlas,
  type Atlas,
} from './atlas.js'
import { Mesh } from './mesh.js'
import { Mat3f, Vec2f } from './vector_math.js'

const VERTEX_SHADER_SOURCE = `
  attribute vec2 a_position;
  attribute vec2 a_texcoord;
  attribute vec4 a_color;

  uniform vec2 u_canvasSize;

  varying vec2 v_texcoord;
  varying vec4 v_color;

  void main() {
    vec2 zeroToOne = a_position / u_canvasSize;
    vec2 zeroToTwo = zeroToOne * 2.0;
    vec2 clipSpace = zeroToTwo - vec2(1.0, 1.0);
    gl_Position = vec4(clipSpace.x, -clipSpace.y, 0.0, 1.0);

    v_texcoord = a_texcoord;
    v_color = a_color;
  }
`

const FRAGMENT_SHADER_SOURCE = `
  precision mediump float;
  uniform sampler2D u_sampler;
  varying vec2 v_texcoord;
  varying vec4 v_color;
  void main() {
    vec4 texColor = texture2D(u_sampler, v_texcoord);
    // texColor is already premultiplied, just multiply by vertex color
    gl_FragColor = texColor * v_color;
  }
`

/** Clamp a value between a minimum and maximum. */
export function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(value, max))
}

/** Context3d class responsible for managing the WebGL context. */
export class Context3d {
  public canvas: HTMLCanvasElement
  public gl: WebGLRenderingContext
  public ready = false
  public dpr = 1
  public atlas: Atlas | null = null

  private texture: WebGLTexture | null = null // The WebGL texture containing all sprites

  // WebGL rendering state
  private shaderProgram: WebGLProgram | null = null

  // Shader locations
  private positionLocation = -1
  private texcoordLocation = -1
  private colorLocation = -1
  private canvasSizeLocation: WebGLUniformLocation | null = null
  private samplerLocation: WebGLUniformLocation | null = null

  // Mesh management
  private meshes: Map<string, Mesh> = new Map()
  private currentMesh: Mesh | null = null
  public currentMeshName = ''

  // Transformation state
  private currentTransform: Mat3f
  private transformStack: Mat3f[] = []

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas
    const gl = canvas.getContext('webgl')
    if (!gl) {
      throw new Error('Failed to get WebGL context')
    }
    this.gl = gl

    // Enable 32-bit index extension for WebGL1
    const uintExtension = gl.getExtension('OES_element_index_uint')
    if (!uintExtension) {
      throw new Error('OES_element_index_uint extension not supported - required for 32-bit indices')
    }

    // Initialize transformation matrix
    this.currentTransform = Mat3f.identity()
  }

  /** Create or switch to a mesh with the given name. */
  useMesh(name: string) {
    if (!this.gl || !this.ready) {
      throw new Error('Cannot use mesh before initialization')
    }

    // If we already have this mesh, set it as current
    if (this.meshes.has(name)) {
      this.currentMesh = this.meshes.get(name)!
      this.currentMeshName = name
      return
    }

    // Otherwise, create a new mesh
    const newMesh = new Mesh(name, this.gl)
    newMesh.createBuffers()
    this.meshes.set(name, newMesh)
    this.currentMesh = newMesh
    this.currentMeshName = name
  }

  /** Sets the scissor rect for the current mesh. */
  setScissorRect(x: number, y: number, width: number, height: number) {
    this.ensureMeshSelected()
    this.currentMesh!.scissorEnabled = true
    this.currentMesh!.scissorRect = [x, y, width, height]
  }

  /** Disable scissoring for the current mesh. */
  disableScissor() {
    this.ensureMeshSelected()
    this.currentMesh!.scissorEnabled = false
  }

  /** Set whether the current mesh should be cached between frames. */
  setCacheable(cacheable: boolean) {
    this.ensureMeshSelected()
    this.currentMesh!.cacheable = cacheable
  }

  /** Clear the current mesh even if it's cacheable. */
  clearMesh() {
    this.ensureMeshSelected()
    this.currentMesh?.forceClear()
  }

  /** Helper method to ensure a mesh is selected before drawing. */
  private ensureMeshSelected() {
    if (!this.currentMesh) {
      throw new Error('No mesh selected. Call useMesh() before drawing.')
    }
  }

  /** Save the current transform. */
  save() {
    // Push a copy of the current transform onto the stack
    this.transformStack.push(
      new Mat3f(
        this.currentTransform.get(0, 0),
        this.currentTransform.get(0, 1),
        this.currentTransform.get(0, 2),
        this.currentTransform.get(1, 0),
        this.currentTransform.get(1, 1),
        this.currentTransform.get(1, 2),
        this.currentTransform.get(2, 0),
        this.currentTransform.get(2, 1),
        this.currentTransform.get(2, 2)
      )
    )
  }

  /** Restore the last transform. */
  restore() {
    // Pop the last transform from the stack
    if (this.transformStack.length > 0) {
      this.currentTransform = this.transformStack.pop()!
    } else {
      console.warn('Transform stack is empty')
    }
  }

  /** Translate the current transform. */
  translate(x: number, y: number) {
    const translateMatrix = Mat3f.translate(x, y)
    this.currentTransform = this.currentTransform.mul(translateMatrix)
  }

  /** Rotate the current transform. */
  rotate(angle: number) {
    const rotateMatrix = Mat3f.rotate(angle)
    this.currentTransform = this.currentTransform.mul(rotateMatrix)
  }

  /** Scale the current transform. */
  scale(x: number, y: number) {
    const scaleMatrix = Mat3f.scale(x, y)
    this.currentTransform = this.currentTransform.mul(scaleMatrix)
  }

  /** Reset the current transform. */
  resetTransform() {
    this.currentTransform = Mat3f.identity()
  }

  /** Initialize the context. */
  async init(atlasJsonUrl: string, atlasImageUrl: string): Promise<boolean> {
    this.dpr = 1.0
    if (window.devicePixelRatio > 1.0) {
      this.dpr = 2.0 // Retina display only, we don't support other DPI scales.
    }

    const result = await loadAtlas(this.gl, atlasJsonUrl, atlasImageUrl)
    if (!result || !result.atlas || !result.texture) {
      this.fail('Failed to load atlas')
      return false
    }
    this.atlas = result.atlas
    this.texture = result.texture

    // Create and compile shaders
    const vertexShader = this.createShader(this.gl.VERTEX_SHADER, VERTEX_SHADER_SOURCE)
    const fragmentShader = this.createShader(this.gl.FRAGMENT_SHADER, FRAGMENT_SHADER_SOURCE)

    if (!vertexShader || !fragmentShader) {
      this.fail('Failed to create shaders')
      return false
    }

    // Create shader program
    this.shaderProgram = this.createProgram(vertexShader, fragmentShader)
    if (!this.shaderProgram) {
      this.fail('Failed to create shader program')
      return false
    }

    // Get attribute and uniform locations
    this.positionLocation = this.gl.getAttribLocation(this.shaderProgram, 'a_position')
    this.texcoordLocation = this.gl.getAttribLocation(this.shaderProgram, 'a_texcoord')
    this.colorLocation = this.gl.getAttribLocation(this.shaderProgram, 'a_color')
    this.canvasSizeLocation = this.gl.getUniformLocation(this.shaderProgram, 'u_canvasSize')
    this.samplerLocation = this.gl.getUniformLocation(this.shaderProgram, 'u_sampler')

    // Enable blending for premultiplied alpha
    this.gl.enable(this.gl.BLEND)
    this.gl.blendFunc(this.gl.ONE, this.gl.ONE_MINUS_SRC_ALPHA)

    this.ready = true
    return true
  }

  /** Fail the context. */
  private fail(msg: string) {
    console.error(msg)
    const failDiv = document.createElement('div')
    failDiv.id = 'fail'
    failDiv.textContent = `Initialization Error: ${msg}. See console for details.`
    document.body.appendChild(failDiv)
  }

  /** Create and compile a shader. */
  private createShader(type: number, source: string): WebGLShader | null {
    const shader = this.gl.createShader(type)
    if (!shader) {
      return null
    }

    this.gl.shaderSource(shader, source)
    this.gl.compileShader(shader)

    if (!this.gl.getShaderParameter(shader, this.gl.COMPILE_STATUS)) {
      console.error('Error compiling shader:', this.gl.getShaderInfoLog(shader))
      this.gl.deleteShader(shader)
      return null
    }

    return shader
  }

  /** Create and link a shader program. */
  private createProgram(vertexShader: WebGLShader, fragmentShader: WebGLShader): WebGLProgram | null {
    const program = this.gl.createProgram()
    if (!program) {
      return null
    }

    this.gl.attachShader(program, vertexShader)
    this.gl.attachShader(program, fragmentShader)
    this.gl.linkProgram(program)

    if (!this.gl.getProgramParameter(program, this.gl.LINK_STATUS)) {
      console.error('Error linking program:', this.gl.getProgramInfoLog(program))
      this.gl.deleteProgram(program)
      return null
    }

    return program
  }

  /** Clears all meshes for a new frame. */
  clear() {
    if (!this.ready) {
      return
    }

    // Clear all meshes in the map
    for (const mesh of this.meshes.values()) {
      mesh.clear()
    }

    // Reset transform for new frame
    this.resetTransform()
    this.transformStack = []
  }

  /** Draws a textured rectangle with the given coordinates and UV mapping. */
  drawRect(
    x: number,
    y: number,
    width: number,
    height: number,
    u0: number,
    v0: number,
    u1: number,
    v1: number,
    color: number[] = [1, 1, 1, 1]
  ) {
    if (!this.ready) {
      throw new Error('Drawer not initialized')
    }

    this.ensureMeshSelected()

    const pos = new Vec2f(x, y)

    // Calculate vertex positions (screen pixels, origin top-left) - 4 vertices for a quad.
    const untransformedTopLeft = pos
    const untransformedBottomLeft = new Vec2f(pos.x(), pos.y() + height)
    const untransformedTopRight = new Vec2f(pos.x() + width, pos.y())
    const untransformedBottomRight = new Vec2f(pos.x() + width, pos.y() + height)

    // Apply current transformation to each vertex
    const topLeft = this.currentTransform.transform(untransformedTopLeft)
    const bottomLeft = this.currentTransform.transform(untransformedBottomLeft)
    const topRight = this.currentTransform.transform(untransformedTopRight)
    const bottomRight = this.currentTransform.transform(untransformedBottomRight)

    // Send pre-transformed vertices to the mesh
    this.currentMesh?.drawRectWithTransform(topLeft, bottomLeft, topRight, bottomRight, u0, v0, u1, v1, color)
  }

  /** Check if the image is in the atlas. */
  hasImage(imageName: string): boolean {
    if (this.atlas) {
      return hasSprite(this.atlas, imageName)
    }
    return false
  }

  /** Draws an image from the atlas with its top-left corner at (x, y). */
  drawImage(imageName: string, x: number, y: number, color: number[] = [1, 1, 1, 1]) {
    if (!this.ready || !this.atlas) {
      throw new Error('Drawer not initialized')
    }

    this.ensureMeshSelected()

    const bounds = getSpriteBounds(this.atlas, imageName)
    if (!bounds) {
      console.error(`Image "${imageName}" not found in atlas`)
      return
    }

    // Draw the rectangle with the image's texture coordinates.
    // Note: bounds x,y already include margin, but we need to offset back for correct positioning
    this.drawRect(x, y, bounds.width, bounds.height, bounds.u0, bounds.v0, bounds.u1, bounds.v1, color)
  }

  /* Draws a sprite from the texture atlas with its center at (centerX, centerY). */
  drawSprite(
    imageName: string, // Name of the image in the atlas (e.g., 'player.png')
    centerX: number, // X coordinate of the sprite's center
    centerY: number, // Y coordinate of the sprite's center
    color: number[] = [1, 1, 1, 1], // RGBA color multiplier [r, g, b, a] where each component is 0.0-1.0
    scale: number | [number, number] = 1, // Uniform scale (number) or non-uniform scale [scaleX, scaleY]
    rotation = 0 // Rotation angle in radians (positive = clockwise)
  ) {
    if (!this.ready || !this.atlas) {
      throw new Error('Drawer not initialized')
    }

    this.ensureMeshSelected()

    const bounds = getSpriteBounds(this.atlas, imageName)
    if (!bounds) {
      console.error(`Image "${imageName}" not found in atlas`)
      return
    }

    // Parse scale parameter - convert uniform scale to [scaleX, scaleY]
    const [scaleX, scaleY] = typeof scale === 'number' ? [scale, scale] : scale

    // Apply transformations if needed (scale or rotation)
    if (scaleX !== 1 || scaleY !== 1 || rotation !== 0) {
      this.save()
      this.translate(centerX, centerY) // Move origin to sprite center
      this.rotate(rotation) // Apply rotation
      this.scale(scaleX, scaleY) // Apply scaling
      this.drawRect(
        -bounds.width / 2, // Left edge: center minus half width
        -bounds.height / 2, // Top edge: center minus half height
        bounds.width, // Total width
        bounds.height, // Total height
        bounds.u0,
        bounds.v0,
        bounds.u1,
        bounds.v1,
        color
      )
      this.restore()
    } else {
      // Fast path: no transformations needed, draw centered
      this.drawRect(
        centerX - bounds.width / 2, // Left edge position
        centerY - bounds.height / 2, // Top edge position
        bounds.width, // Total width
        bounds.height, // Total height
        bounds.u0,
        bounds.v0,
        bounds.u1,
        bounds.v1,
        color
      )
    }
  }

  /** Draws a solid filled rectangle. */
  drawSolidRect(x: number, y: number, width: number, height: number, color: number[]) {
    if (!this.ready || !this.atlas) {
      throw new Error('Drawer not initialized')
    }

    this.ensureMeshSelected()

    const whiteUV = getWhiteUV(this.atlas)
    if (!whiteUV) {
      console.error('White pixel sprite not found in atlas')
      return
    }

    this.drawRect(x, y, width, height, whiteUV.u, whiteUV.v, whiteUV.u, whiteUV.v, color)
  }

  /** Draws a stroked rectangle with set stroke width. */
  drawStrokeRect(x: number, y: number, width: number, height: number, strokeWidth: number, color: number[]) {
    // Draw 4 rectangles as borders for the stroke rectangle.
    // Top border.
    this.drawSolidRect(x, y, width, strokeWidth, color)
    // Bottom border.
    this.drawSolidRect(x, y + height - strokeWidth, width, strokeWidth, color)
    // Left border.
    this.drawSolidRect(x, y + strokeWidth, strokeWidth, height - 2 * strokeWidth, color)
    // Right border.
    this.drawSolidRect(x + width - strokeWidth, y + strokeWidth, strokeWidth, height - 2 * strokeWidth, color)
  }

  /**
   * Draw text using sprites from the atlas.
   *
   * x,y specify the baseline origin of the first line (top-left origin screen coordinates).
   * fonts are generated by gen_atlas.py.
   */
  drawText(fontName: string, text: string, x: number, y: number, color: number[] = [1, 1, 1, 1], scale = 1) {
    if (!this.ready || !this.atlas) {
      throw new Error('Drawer not initialized')
    }
    this.ensureMeshSelected()

    const font = getFont(this.atlas, fontName)
    if (!font) {
      console.error(`Font "${fontName}" not found in atlas`)
      return
    }

    const glyphInnerPadding = font.glyphInnerPadding
    const mBase = this.atlas.margin

    // Draw relative to (0,0) under a local transform anchored at (x,y)
    this.save()
    this.translate(x, y)
    if (scale !== 1) {
      this.scale(scale, scale)
    }

    let penX = 0
    let penY = 0
    let prevChar: string | null = null

    for (const ch of text) {
      if (ch === '\n') {
        penX = 0
        penY += font.lineHeight
        prevChar = null
        continue
      }

      const glyph = getGlyph(this.atlas, fontName, ch)

      // Special handling: advance for spaces/tabs without rendering a quad.
      if (ch === ' ') {
        penX += glyph ? glyph.advance : 0
        prevChar = ch
        continue
      }
      if (ch === '\t') {
        const spaceGlyph = getGlyph(this.atlas, fontName, ' ')
        const tabAdvance = (spaceGlyph ? spaceGlyph.advance : glyph ? glyph.advance : 0) * 4
        penX += tabAdvance
        prevChar = ch
        continue
      }

      if (!glyph) {
        console.warn(`Glyph for character "${ch}" not found in font "${fontName}"`)
        continue
      }
      if (!glyph.rect) {
        // Glyph has no visible pixels, just advance
        penX += glyph.advance
        prevChar = ch
        continue
      }

      // Kerning adjustment from previous glyph
      if (prevChar) {
        const kerning = getKerning(this.atlas, fontName, prevChar, ch)
        penX += kerning
      }

      const [sx, sy, sw, sh] = glyph.rect
      const m = mBase
      const u0 = (sx - mBase) / this.atlas.atlasWidth
      const v0 = (sy - mBase) / this.atlas.atlasHeight
      const u1 = (sx + sw + mBase) / this.atlas.atlasWidth
      const v1 = (sy + sh + mBase) / this.atlas.atlasHeight

      // Position the glyph image so that its baseline aligns at (penX, penY).
      const drawX = penX + glyph.bearingX - glyphInnerPadding - m
      const drawY = penY + glyph.bearingY - glyphInnerPadding - m
      const drawW = sw + 2 * m
      const drawH = sh + 2 * m

      // Main glyph.
      this.drawRect(drawX, drawY, drawW, drawH, u0, v0, u1, v1, color)

      // Advance pen position
      penX += glyph.advance
      prevChar = ch
    }

    this.restore()
  }

  /** Flushes all non-empty meshes to the screen. */
  flush() {
    if (!this.ready || !this.gl || !this.shaderProgram || !this.atlas || !this.texture) {
      return
    }

    // If no meshes have been created, nothing to do
    if (this.meshes.size === 0) {
      return
    }

    // Handle high-DPI displays by resizing the canvas if necessary.
    const clientWidth = window.innerWidth
    const clientHeight = window.innerHeight
    const screenWidth = Math.round(clientWidth * this.dpr)
    const screenHeight = Math.round(clientHeight * this.dpr)
    if (this.canvas.width !== screenWidth || this.canvas.height !== screenHeight) {
      this.canvas.width = screenWidth
      this.canvas.height = screenHeight
      this.canvas.style.width = `${clientWidth}px`
      this.canvas.style.height = `${clientHeight}px`
      this.gl.viewport(0, 0, screenWidth, screenHeight)
    }

    // Clear the canvas
    this.gl.clearColor(0.1, 0.1, 0.1, 1.0) // Dark grey clear
    this.gl.clear(this.gl.COLOR_BUFFER_BIT)

    // Use shader program
    this.gl.useProgram(this.shaderProgram)

    // Set canvas size uniform
    this.gl.uniform2f(this.canvasSizeLocation, screenWidth, screenHeight)

    // Bind texture
    this.gl.activeTexture(this.gl.TEXTURE0)
    this.gl.bindTexture(this.gl.TEXTURE_2D, this.texture)
    this.gl.uniform1i(this.samplerLocation, 0)

    // Draw each mesh that has quads
    for (const mesh of this.meshes.values()) {
      const quadCount = mesh.currentQuad
      if (quadCount === 0) {
        continue
      }

      const vertexBuffer = mesh.vertexBuffer
      const indexBuffer = mesh.indexBuffer

      if (!vertexBuffer || !indexBuffer) {
        continue
      }

      // Calculate data sizes
      const vertexDataCount = mesh.currentVertex * 8 // 8 floats per vertex
      const indexDataCount = quadCount * 6 // 6 indices per quad

      // Update vertex buffer with current data only if dirty
      this.gl.bindBuffer(this.gl.ARRAY_BUFFER, vertexBuffer)
      if (mesh.isDirty) {
        this.gl.bufferSubData(this.gl.ARRAY_BUFFER, 0, mesh.vertexData.subarray(0, vertexDataCount))
        mesh.isDirty = false
      }

      // Set up attributes
      this.gl.enableVertexAttribArray(this.positionLocation)
      this.gl.vertexAttribPointer(this.positionLocation, 2, this.gl.FLOAT, false, 8 * 4, 0) // position (2 floats)

      this.gl.enableVertexAttribArray(this.texcoordLocation)
      this.gl.vertexAttribPointer(this.texcoordLocation, 2, this.gl.FLOAT, false, 8 * 4, 2 * 4) // texcoord (2 floats)

      this.gl.enableVertexAttribArray(this.colorLocation)
      this.gl.vertexAttribPointer(this.colorLocation, 4, this.gl.FLOAT, false, 8 * 4, 4 * 4) // color (4 floats)

      // Bind index buffer
      this.gl.bindBuffer(this.gl.ELEMENT_ARRAY_BUFFER, indexBuffer)

      // Apply scissor if enabled for this mesh
      if (mesh.scissorEnabled) {
        const [x, y, width, height] = mesh.scissorRect
        const w = Math.floor(screenWidth)
        const h = Math.floor(screenHeight)
        this.gl.enable(this.gl.SCISSOR_TEST)
        this.gl.scissor(
          clamp(Math.floor(x), 0, w),
          clamp(Math.floor(h - y - height), 0, h), // WebGL scissor Y is bottom-up
          clamp(Math.floor(width), 0, w - Math.floor(x)),
          clamp(Math.floor(height), 0, h - Math.floor(y))
        )
      } else {
        this.gl.disable(this.gl.SCISSOR_TEST)
      }

      // Draw the mesh
      this.gl.drawElements(this.gl.TRIANGLES, indexDataCount, this.gl.UNSIGNED_INT, 0)
    }

    // Disable scissor test for next frame
    this.gl.disable(this.gl.SCISSOR_TEST)

    // Reset all mesh counters after rendering
    for (const mesh of this.meshes.values()) {
      mesh.resetCounters()
    }
  }

  /**
   * Draws a line of sprites.
   * The line is drawn from (x0, y0) to (x1, y1).
   * The spacing is the distance between the centers of the sprites.
   * The color is the color of the sprites.
   * The skipStart and skipEnd are the number of sprites to skip at the start
   * and end of the line.
   */
  drawSpriteLine(
    imageName: string,
    x0: number,
    y0: number,
    x1: number,
    y1: number,
    spacing: number,
    color: number[],
    skipStart = 0,
    skipEnd = 0
  ) {
    // Compute the angle of the line.
    const angle = Math.atan2(y1 - y0, x1 - x0)
    // Compute the length of the line.
    const x = x1 - x0
    const y = y1 - y0
    const length = Math.sqrt(x ** 2 + y ** 2)
    // Compute the number of dashes.
    const numDashes = Math.floor(length / spacing) + 1
    // Compute the delta of each dash.
    const dx = x / numDashes
    const dy = y / numDashes
    // Draw the dashes.
    for (let i = 0; i < numDashes; i++) {
      if (i < skipStart || i >= numDashes - skipEnd) {
        continue
      }
      this.drawSprite(imageName, x0 + i * dx, y0 + i * dy, color, 1, -angle)
    }
  }
}
