import { getSpriteBounds, getWhiteUV, hasSprite, loadAtlas, SpriteBounds, type Atlas } from './atlas.js'
import type { RGBA } from './htmlutils.js'
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

  public mainAtlas: Atlas | null = null
  public fontAtlas: Atlas | null = null

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
  async init(atlasJsonUrl: string, atlasImageUrl: string, fontJsonUrl: string, fontImageUrl: string): Promise<boolean> {
    this.dpr = 1.0
    if (window.devicePixelRatio > 1.0) {
      this.dpr = 2.0 // Retina display only, we don't support other DPI scales.
    }

    this.mainAtlas = await loadAtlas(this.gl, atlasJsonUrl, atlasImageUrl)
    this.fontAtlas = await loadAtlas(this.gl, fontJsonUrl, fontImageUrl)

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
    color: RGBA = [1, 1, 1, 1]
  ) {
    if (!this.ready) {
      throw new Error('Drawer not initialized')
    }

    this.ensureMeshSelected()

    const pos = new Vec2f(x, y)

    // Calculate vertex positions (screen pixels, origin top-left)
    // We'll make 4 vertices for a quad
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
    this.currentMesh?.addQuad(topLeft, bottomLeft, topRight, bottomRight, u0, v0, u1, v1, color)
  }

  /** Check if the image is in the atlas. */
  hasImage(imageName: string): boolean {
    return this.mainAtlas ? hasSprite(this.mainAtlas, imageName) : false
  }

  /** Draws an image from the atlas with its top-left corner at (x, y). */
  drawImage(imageName: string, x: number, y: number, color: RGBA = [1, 1, 1, 1]) {
    if (!this.ready || !this.mainAtlas) {
      throw new Error('Drawer not initialized')
    }

    this.ensureMeshSelected()

    const bounds = getSpriteBounds(this.mainAtlas, imageName)
    if (!bounds) {
      console.error(`Image "${imageName}" not found in atlas`)
      return
    }

    // Draw the rectangle with the image's texture coordinates
    // The bounds already include margin adjustments
    this.drawRect(
      x + bounds.x,
      y + bounds.y,
      bounds.width,
      bounds.height,
      bounds.u0,
      bounds.v0,
      bounds.u1,
      bounds.v1,
      color
    )
  }

  /* Draws a sprite from the texture atlas with its center at (centerX, centerY). */
  drawSprite(
    imageName: string,
    centerX: number,
    centerY: number,
    color: RGBA = [1, 1, 1, 1],
    scale: number | [number, number] = 1,
    rotation: number = 0
  ) {
    if (!this.ready || !this.mainAtlas) {
      throw new Error('Drawer not initialized')
    }
    this.ensureMeshSelected()

    // lookup sprite bounds
    const bounds = getSpriteBounds(this.mainAtlas, imageName)
    if (!bounds) {
      console.error(`Image "${imageName}" not found in atlas`)
      return
    }

    // calculate scale components
    const [scaleX, scaleY] = typeof scale === 'number' ? [scale, scale] : scale

    // draw (with or without transform)
    if (scaleX !== 1 || scaleY !== 1 || rotation !== 0) {
      this.save()
      this.translate(centerX, centerY)
      this.rotate(rotation)
      this.scale(scaleX, scaleY)
      this.drawRect(
        -bounds.width / 2,
        -bounds.height / 2,
        bounds.width,
        bounds.height,
        bounds.u0,
        bounds.v0,
        bounds.u1,
        bounds.v1,
        color
      )
      this.restore()
    } else {
      this.drawRect(
        centerX - bounds.width / 2,
        centerY - bounds.height / 2,
        bounds.width,
        bounds.height,
        bounds.u0,
        bounds.v0,
        bounds.u1,
        bounds.v1,
        color
      )
    }
  }

  /** Draws a solid filled rectangle. */
  drawSolidRect(x: number, y: number, width: number, height: number, color: RGBA) {
    if (!this.ready || !this.mainAtlas) {
      throw new Error('Drawer not initialized')
    }

    this.ensureMeshSelected()

    const whiteUV = getWhiteUV(this.mainAtlas)
    if (!whiteUV) {
      throw new Error('white.png not found in atlas')
    }

    this.drawRect(x, y, width, height, whiteUV.u, whiteUV.v, whiteUV.u, whiteUV.v, color)
  }

  /** Draws a stroked rectangle with set stroke width. */
  drawStrokeRect(x: number, y: number, width: number, height: number, strokeWidth: number, color: RGBA) {
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

  /* Draws text within the specified bounding box. */
  drawText(
    text: string,
    bbox: [number, number, number, number],
    color: RGBA = [1, 1, 1, 1],
    mode: 'scale' | 'stretch' = 'scale',
    align: 'left' | 'center' | 'right' = 'left',
    valign: 'top' | 'middle' | 'bottom' = 'top',
    spacing = 0
  ) {
    if (!this.ready) {
      throw new Error('Context not ready')
    }
    if (!this.fontAtlas || !this.fontAtlas.texture) {
      console.error('Font atlas not loaded or texture missing')
      return
    }

    // Save the current mesh by name
    const previousMeshName = this.currentMeshName

    // Switch to font mesh or create it if needed
    this.useMesh('font')
    this.currentMesh!.texture = this.fontAtlas.texture

    // Cast fontAtlasData to any to access our custom structure
    const metadata = this.fontAtlas.metadata as any

    // Get emoji codes from font data
    const emojiCodes = metadata.emojiCodes || {}
    const emojiValues = new Set(Object.values(emojiCodes))

    // Build regex pattern from actual emoji codes in the font atlas
    const emojiCodePattern = Object.keys(emojiCodes)
      .map((code) => code.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'))
      .join('|')
    const emojiPattern = emojiCodePattern ? new RegExp(`^(${emojiCodePattern})`) : null

    // Helper function to get character info with emoji code support
    const getCharInfo = (char: string): SpriteBounds | undefined => {
      // First check if it's an emoji code
      if (emojiCodes[char]) {
        const emojiChar = emojiCodes[char]
        return this.fontAtlas!.data[emojiChar]
      }

      // For regular characters, look them up directly
      return this.fontAtlas!.data[char]
    }

    // Parse text to handle multi-character emoji codes
    const parseText = (text: string): string[] => {
      const chars: string[] = []
      let remaining = text

      while (remaining.length > 0) {
        let matched = false

        // Try to match emoji codes first
        if (emojiPattern) {
          const match = remaining.match(emojiPattern)
          if (match && match.index === 0) {
            chars.push(match[0])
            remaining = remaining.substring(match[0].length)
            matched = true
          }
        }

        // If no emoji code matched, take the next character
        if (!matched) {
          chars.push(remaining[0])
          remaining = remaining.substring(1)
        }
      }

      return chars
    }

    // Parse the text into characters/emoji codes
    const textChars = parseText(text)

    // Calculate actual text dimensions using real character widths
    let textWidth = 0
    const textHeight = metadata.cellHeight // Use cell height as text height

    // Build character info array while calculating width
    const charInfos: Array<{ char: string; info: SpriteBounds; width: number }> = []

    for (const char of textChars) {
      const info = getCharInfo(char)
      if (!info) {
        console.warn(`Character not found in font atlas: "${char}" (code: ${char.charCodeAt(0)})`)
        continue
      }
      const [_x, _y, width, _height] = info
      charInfos.push({ char, info, width })
      textWidth += width
    }

    // Add spacing between characters
    if (charInfos.length > 1) {
      textWidth += (charInfos.length - 1) * spacing
    }

    // Handle case where no valid characters were found
    if (textWidth === 0 || charInfos.length === 0) {
      console.warn('No valid characters found to render')
      if (previousMeshName) {
        this.useMesh(previousMeshName)
      }
      return
    }

    // Calculate scale factors based on mode
    const [bx, by, bw, bh] = bbox
    let scaleFactorX = 1
    let scaleFactorY = 1

    if (mode === 'scale') {
      // Scale as large as possible while maintaining aspect ratio
      const widthScale = bw / textWidth
      const heightScale = bh / textHeight
      scaleFactorX = scaleFactorY = Math.min(widthScale, heightScale)
    } else if (mode === 'stretch') {
      // Stretch to fill entire bbox (may distort)
      scaleFactorX = bw / textWidth
      scaleFactorY = bh / textHeight
    }

    // Calculate actual rendered dimensions
    const scaledWidth = textWidth * scaleFactorX
    const scaledHeight = textHeight * scaleFactorY

    // Calculate starting position based on alignment
    let cursorX = bx
    if (align === 'center') {
      cursorX = bx + (bw - scaledWidth) / 2
    } else if (align === 'right') {
      cursorX = bx + bw - scaledWidth
    }

    let cursorY = by
    if (valign === 'middle') {
      cursorY = by + (bh - scaledHeight) / 2
    } else if (valign === 'bottom') {
      cursorY = by + bh - scaledHeight
    }

    // Draw each character
    for (const { char, info, width } of charInfos) {
      const [sx, sy, sw, sh] = info

      // Check if this is an emoji code and convert to emoji character
      const spriteChar = emojiCodes[char] || char

      // Check if this is an emoji
      const isEmoji = emojiValues.has(spriteChar) || emojiCodes.hasOwnProperty(char)
      const drawColor: RGBA = isEmoji ? [1, 1, 1, 1] : color

      // Calculate UV coordinates
      const u0 = sx / this.fontAtlas.size.x()
      const v0 = sy / this.fontAtlas.size.y()
      const u1 = (sx + sw) / this.fontAtlas.size.x()
      const v1 = (sy + sh) / this.fontAtlas.size.y()

      // Debug: Check if UV coordinates are valid
      if (u0 < 0 || u0 > 1 || v0 < 0 || v0 > 1 || u1 < 0 || u1 > 1 || v1 < 0 || v1 > 1) {
        console.warn(`Invalid UV coordinates for char "${char}": u0=${u0}, v0=${v0}, u1=${u1}, v1=${v1}`)
      }

      // Draw the character rectangle
      this.drawRect(cursorX, cursorY, width * scaleFactorX, textHeight * scaleFactorY, u0, v0, u1, v1, drawColor)

      cursorX += width * scaleFactorX + spacing * scaleFactorX
    }

    // Restore previous mesh
    if (previousMeshName) {
      this.useMesh(previousMeshName)
    }
  }

  /** Flushes all non-empty meshes to the screen. */
  flush() {
    if (!this.ready || !this.gl || !this.shaderProgram) {
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

    // Draw each mesh that has quads
    for (const mesh of this.meshes.values()) {
      // Bind texture
      const texture = mesh.texture || this.mainAtlas!.texture
      this.gl.activeTexture(this.gl.TEXTURE0)
      this.gl.bindTexture(this.gl.TEXTURE_2D, texture)
      this.gl.uniform1i(this.samplerLocation, 0)

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
    color: RGBA,
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
