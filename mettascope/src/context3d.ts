import { type Atlas, getSpriteBounds, getWhiteUV, hasSprite, loadAtlas } from './atlas.js'
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
    // Do the premultiplied alpha conversion.
    vec4 premultipliedColor = vec4(texColor.rgb * texColor.a, texColor.a);
    gl_FragColor = premultipliedColor * v_color;
  }
`

/**
 * Clamp a value between minimum and maximum bounds.
 *
 * @param value - The value to clamp
 * @param min - Minimum allowed value
 * @param max - Maximum allowed value
 * @returns The clamped value
 */
export function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(value, max))
}

/**
 * Context3d manages the WebGL rendering context and provides a high-level 2D drawing API.
 *
 * This class serves as the main interface for rendering sprites, shapes, and text.
 * It handles:
 * - WebGL context initialization and shader management
 * - Texture atlas loading and sprite rendering
 * - Transform stack for hierarchical transformations
 * - Mesh management for batched rendering
 * - Scissor rectangles for clipping
 *
 * The rendering pipeline uses a single shader program with premultiplied alpha
 * blending for correct transparency handling.
 *
 * @example
 * const ctx = new Context3d(canvas)
 * await ctx.init('atlas.json', 'atlas.png')
 *
 * // Each frame:
 * ctx.clear()
 * ctx.useMesh('sprites')
 * ctx.drawSprite('player.png', 100, 100)
 * ctx.drawSolidRect(0, 0, 50, 50, [1, 0, 0, 1])
 * ctx.flush()
 */
export class Context3d {
  public canvas: HTMLCanvasElement
  public gl: WebGLRenderingContext
  public ready = false
  public dpr = 1

  // Atlas
  private atlas: Atlas | null = null

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

  /**
   * Create a new rendering context.
   *
   * @param canvas - The HTML canvas element to render to
   * @throws Error if WebGL context creation fails or required extensions are missing
   */
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

  /**
   * Create or switch to a named mesh.
   *
   * Meshes are used to batch draw calls. All drawing operations are added
   * to the current mesh. Switching meshes allows organizing draws into
   * different batches (e.g., background, sprites, UI).
   *
   * @param name - Unique name for the mesh
   * @throws Error if called before initialization
   *
   * @example
   * ctx.useMesh('background')
   * ctx.drawSolidRect(0, 0, 800, 600, [0.5, 0.5, 0.5, 1])
   *
   * ctx.useMesh('sprites')
   * ctx.drawSprite('player.png', 100, 100)
   */
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

    // Otherwise, create a new mesh with the name and gl context
    const newMesh = new Mesh(name, this.gl)
    this.meshes.set(name, newMesh)
    this.currentMesh = newMesh
    this.currentMeshName = name
  }

  /**
   * Set a scissor rectangle for the current mesh.
   *
   * The scissor test clips all rendering to the specified rectangle.
   * Useful for implementing scrollable areas or viewports.
   *
   * @param x - Left edge in pixels
   * @param y - Top edge in pixels
   * @param width - Width in pixels
   * @param height - Height in pixels
   */
  setScissorRect(x: number, y: number, width: number, height: number) {
    this.ensureMeshSelected()

    this.currentMesh!.scissorEnabled = true
    this.currentMesh!.scissorRect = [x, y, width, height]
  }

  /**
   * Disable scissor testing for the current mesh.
   */
  disableScissor() {
    this.ensureMeshSelected()
    this.currentMesh!.scissorEnabled = false
  }

  /**
   * Set whether the current mesh should be cached between frames.
   *
   * @param cacheable - If true, mesh content persists between frames
   */
  setCacheable(cacheable: boolean) {
    this.ensureMeshSelected()
    this.currentMesh!.setCacheable(cacheable)
  }

  /**
   * Clear the current mesh even if it's cacheable.
   */
  clearMesh() {
    this.ensureMeshSelected()
    this.currentMesh!.forceClear()
  }

  /**
   * Helper method to ensure a mesh is selected before drawing.
   *
   * @private
   * @throws Error if no mesh is selected
   */
  private ensureMeshSelected() {
    if (!this.currentMesh) {
      throw new Error('No mesh selected. Call useMesh() before drawing.')
    }
  }

  /**
   * Save the current transformation matrix.
   *
   * Pushes a copy of the current transform onto the stack. Used with
   * restore() to create hierarchical transformations.
   *
   * @example
   * ctx.save()
   * ctx.translate(100, 100)
   * ctx.rotate(Math.PI / 4)
   * ctx.drawSprite('rotated.png', 0, 0)
   * ctx.restore() // Back to original transform
   */
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

  /**
   * Restore the previously saved transformation matrix.
   *
   * Pops a transform from the stack. Must be paired with a previous save().
   */
  restore() {
    // Pop the last transform from the stack
    if (this.transformStack.length > 0) {
      this.currentTransform = this.transformStack.pop()!
    } else {
      console.warn('Transform stack is empty')
    }
  }

  /**
   * Translate (move) the current transformation.
   *
   * @param x - Horizontal translation in pixels
   * @param y - Vertical translation in pixels
   */
  translate(x: number, y: number) {
    const translateMatrix = Mat3f.translate(x, y)
    this.currentTransform = this.currentTransform.mul(translateMatrix)
  }

  /**
   * Rotate the current transformation.
   *
   * @param angle - Rotation angle in radians (positive = clockwise)
   */
  rotate(angle: number) {
    const rotateMatrix = Mat3f.rotate(angle)
    this.currentTransform = this.currentTransform.mul(rotateMatrix)
  }

  /**
   * Scale the current transformation.
   *
   * @param x - Horizontal scale factor (1 = normal, 2 = double size, -1 = flip)
   * @param y - Vertical scale factor
   */
  scale(x: number, y: number) {
    const scaleMatrix = Mat3f.scale(x, y)
    this.currentTransform = this.currentTransform.mul(scaleMatrix)
  }

  /**
   * Reset the transformation matrix to identity.
   */
  resetTransform() {
    this.currentTransform = Mat3f.identity()
  }

  /**
   * Initialize the rendering context with a texture atlas.
   *
   * Must be called before any drawing operations. Loads the atlas data
   * and texture, creates shaders, and sets up WebGL state.
   *
   * @param atlasJsonUrl - URL to the atlas JSON file
   * @param atlasImageUrl - URL to the atlas image file
   * @returns Promise resolving to true if initialization succeeded
   *
   * @example
   * const success = await ctx.init('assets/sprites.json', 'assets/sprites.png')
   * if (success) {
   *   // Ready to draw
   * }
   */
  async init(atlasJsonUrl: string, atlasImageUrl: string): Promise<boolean> {
    this.dpr = 1.0
    if (window.devicePixelRatio > 1.0) {
      this.dpr = 2.0 // Retina display only, we don't support other DPI scales.
    }

    // Load atlas using the utility function
    this.atlas = await loadAtlas(this.gl, atlasJsonUrl, atlasImageUrl, {
      wrapS: this.gl.REPEAT,
      wrapT: this.gl.REPEAT,
      minFilter: this.gl.LINEAR_MIPMAP_LINEAR,
      magFilter: this.gl.LINEAR,
    })

    if (!this.atlas) {
      this.fail('Failed to load atlas')
      return false
    }

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

  /**
   * Display initialization failure message.
   *
   * @private
   * @param msg - Error message to display
   */
  private fail(msg: string) {
    console.error(msg)
    const failDiv = document.createElement('div')
    failDiv.id = 'fail'
    failDiv.textContent = `Initialization Error: ${msg}. See console for details.`
    document.body.appendChild(failDiv)
  }

  /**
   * Create and compile a WebGL shader.
   *
   * @private
   * @param type - Shader type (VERTEX_SHADER or FRAGMENT_SHADER)
   * @param source - GLSL source code
   * @returns Compiled shader or null if compilation failed
   */
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

  /**
   * Create and link a shader program.
   *
   * @private
   * @param vertexShader - Compiled vertex shader
   * @param fragmentShader - Compiled fragment shader
   * @returns Linked shader program or null if linking failed
   */
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

  /**
   * Clear all meshes for a new frame.
   *
   * Resets all mesh vertex counters and the transformation matrix.
   * Should be called at the start of each frame.
   */
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

  /**
   * Draw a textured rectangle with explicit UV coordinates.
   *
   * Low-level drawing function. Most code should use drawSprite() or
   * drawImage() instead, which handle UV calculation automatically.
   *
   * @param x - Left edge position
   * @param y - Top edge position
   * @param width - Rectangle width
   * @param height - Rectangle height
   * @param u0 - Left texture coordinate (0-1)
   * @param v0 - Top texture coordinate (0-1)
   * @param u1 - Right texture coordinate (0-1)
   * @param v1 - Bottom texture coordinate (0-1)
   * @param color - RGBA color multiplier
   */
  drawRect(
    x: number,
    y: number,
    width: number,
    height: number,
    u0: number,
    v0: number,
    u1: number,
    v1: number,
    color: [number, number, number, number] = [1, 1, 1, 1]
  ) {
    if (!this.ready) {
      throw new Error('Drawer not initialized')
    }

    this.ensureMeshSelected()

    const pos = new Vec2f(x, y)

    // Calculate vertex positions (screen pixels, origin top-left)
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
    this.currentMesh!.addQuad(topLeft, bottomLeft, topRight, bottomRight, u0, v0, u1, v1, color)
  }

  /**
   * Check if a sprite exists in the atlas.
   *
   * @param imageName - Name of the sprite to check
   * @returns True if the sprite exists
   */
  hasImage(imageName: string): boolean {
    return this.atlas ? hasSprite(this.atlas, imageName) : false
  }

  /**
   * Draw a sprite from the atlas with its top-right corner at (x, y).
   *
   * @param imageName - Name of the sprite in the atlas
   * @param x - Right edge position
   * @param y - Top edge position
   * @param color - RGBA color multiplier (default: white/no tint)
   *
   * @example
   * ctx.drawImage('button.png', 10, 10)
   * ctx.drawImage('button.png', 10, 50, [0.5, 0.5, 0.5, 1]) // Darkened
   */
  drawImage(imageName: string, x: number, y: number, color: [number, number, number, number] = [1, 1, 1, 1]) {
    if (!this.ready || !this.atlas) {
      throw new Error('Drawer not initialized')
    }

    this.ensureMeshSelected()

    const bounds = getSpriteBounds(this.atlas, imageName)
    if (!bounds) {
      console.error(`Image "${imageName}" not found in atlas`)
      return
    }

    // Adjust the position to account for top-right corner:
    const adjustedX = x - bounds.width // Subtract width to align with the top-right corner
    const adjustedY = y

    // Draw the rectangle with the image's texture coordinates
    this.drawRect(
      adjustedX + bounds.x,
      adjustedY + bounds.y,
      bounds.width,
      bounds.height,
      bounds.u0,
      bounds.v0,
      bounds.u1,
      bounds.v1,
      color
    )
  }

  /**
   * Draw a sprite from the texture atlas centered at the specified position.
   *
   * @param imageName - Name of the image in the atlas (e.g., 'player.png')
   * @param x - X coordinate of the sprite's center
   * @param y - Y coordinate of the sprite's center
   * @param color - RGBA color multiplier [r, g, b, a] where each component is 0.0-1.0
   * @param scale - Uniform scale (number) or non-uniform scale [scaleX, scaleY]
   * @param rotation - Rotation angle in radians (positive = clockwise)
   *
   * @example
   * // Draw at original size
   * ctx.drawSprite('player.png', 100, 200)
   *
   * // Draw with uniform scale
   * ctx.drawSprite('player.png', 100, 200, [1, 1, 1, 1], 2)
   *
   * // Draw mirrored horizontally
   * ctx.drawSprite('player.png', 100, 200, [1, 1, 1, 1], [-1, 1])
   *
   * // Draw with rotation (45 degrees)
   * ctx.drawSprite('player.png', 100, 200, [1, 1, 1, 1], 1, Math.PI / 4)
   */
  drawSprite(
    imageName: string,
    x: number,
    y: number,
    color: [number, number, number, number] = [1, 1, 1, 1],
    scale: number | [number, number] = 1,
    rotation = 0
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

    // biome-ignore format: keep comments aligned
    // Apply transformations if needed (scale or rotation)
    if (scaleX !== 1 || scaleY !== 1 || rotation !== 0) {
      this.save()
      this.translate(x, y) // Move origin to sprite center
      this.rotate(rotation) // Apply rotation
      this.scale(scaleX, scaleY) // Apply scaling
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
      // Fast path: no transformations needed, draw centered
      this.drawRect(
        x - bounds.width / 2,
        y - bounds.height / 2,
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

  /**
   * Draw a solid filled rectangle.
   *
   * Uses a white pixel from the atlas and tints it with the specified color.
   * The atlas must contain 'white.png' for this to work.
   *
   * @param x - Left edge position
   * @param y - Top edge position
   * @param width - Rectangle width
   * @param height - Rectangle height
   * @param color - RGBA fill color
   *
   * @example
   * ctx.drawSolidRect(0, 0, 100, 50, [1, 0, 0, 1])      // Red rectangle
   * ctx.drawSolidRect(0, 0, 800, 600, [0, 0, 0, 0.5])  // Semi-transparent black overlay
   */
  drawSolidRect(x: number, y: number, width: number, height: number, color: [number, number, number, number]) {
    if (!this.ready || !this.atlas) {
      throw new Error('Drawer not initialized')
    }

    this.ensureMeshSelected()

    const whiteUV = getWhiteUV(this.atlas)
    if (!whiteUV) {
      throw new Error('white.png not found in atlas')
    }

    this.drawRect(x, y, width, height, whiteUV.u, whiteUV.v, whiteUV.u, whiteUV.v, color)
  }

  /**
   * Draw a stroked (outline) rectangle.
   *
   * Draws four rectangles to form the border.
   *
   * @param x - Left edge position
   * @param y - Top edge position
   * @param width - Rectangle width
   * @param height - Rectangle height
   * @param strokeWidth - Border thickness in pixels
   * @param color - RGBA stroke color
   *
   * @example
   * ctx.drawStrokeRect(10, 10, 100, 100, 2, [0, 0, 0, 1]) // 2px black border
   */
  drawStrokeRect(
    x: number,
    y: number,
    width: number,
    height: number,
    strokeWidth: number,
    color: [number, number, number, number]
  ) {
    // Draw 4 rectangles as borders for the stroke rectangle
    // Top border
    this.drawSolidRect(x, y, width, strokeWidth, color)
    // Bottom border
    this.drawSolidRect(x, y + height - strokeWidth, width, strokeWidth, color)
    // Left border
    this.drawSolidRect(x, y + strokeWidth, strokeWidth, height - 2 * strokeWidth, color)
    // Right border
    this.drawSolidRect(x + width - strokeWidth, y + strokeWidth, strokeWidth, height - 2 * strokeWidth, color)
  }

  /**
   * Flush all queued draw calls to the screen.
   *
   * This is the main rendering function that:
   * 1. Resizes the canvas to match the window (handling high-DPI)
   * 2. Clears the screen
   * 3. Renders each mesh in order
   *
   * Must be called after all drawing operations to see results.
   *
   * @example
   * // Game loop
   * function render() {
   *   ctx.clear()
   *   ctx.useMesh('game')
   *   drawGame()
   *   ctx.flush()
   *   requestAnimationFrame(render)
   * }
   */
  flush() {
    if (!this.ready || !this.gl || !this.shaderProgram || !this.atlas) {
      return
    }

    // If no meshes have been created, nothing to do
    if (this.meshes.size === 0) {
      return
    }

    // Handle high-DPI displays by resizing the canvas if necessary
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
    this.gl.bindTexture(this.gl.TEXTURE_2D, this.atlas.texture)
    this.gl.uniform1i(this.samplerLocation, 0)

    // Draw each mesh that has quads
    for (const mesh of this.meshes.values()) {
      if (!mesh.hasContent()) {
        continue
      }

      // Upload vertex data to GPU
      mesh.uploadToGPU()

      // Bind mesh buffers and set up attributes
      mesh.bind(this.positionLocation, this.texcoordLocation, this.colorLocation)

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
      this.gl.drawElements(this.gl.TRIANGLES, mesh.getIndexCount(), this.gl.UNSIGNED_INT, 0)
    }

    // Disable scissor test for next frame
    this.gl.disable(this.gl.SCISSOR_TEST)

    // Reset all mesh counters after rendering
    for (const mesh of this.meshes.values()) {
      mesh.resetCounters()
    }
  }

  /**
   * Draw a line of sprites between two points.
   *
   * Useful for dotted lines, paths, or decorative elements. The sprites
   * are evenly spaced and rotated to follow the line direction.
   *
   * @param imageName - Name of the sprite to repeat
   * @param x0 - Starting X position
   * @param y0 - Starting Y position
   * @param x1 - Ending X position
   * @param y1 - Ending Y position
   * @param spacing - Distance between sprite centers
   * @param color - RGBA color for all sprites
   * @param skipStart - Number of sprites to skip at the start
   * @param skipEnd - Number of sprites to skip at the end
   *
   * @example
   * // Dotted line
   * ctx.drawSpriteLine('dot.png', 0, 0, 200, 0, 20, [1, 1, 1, 1])
   *
   * // Arrow path with gaps at ends
   * ctx.drawSpriteLine('arrow.png', 0, 0, 200, 200, 30, [0, 1, 0, 1], 1, 1)
   */
  drawSpriteLine(
    imageName: string,
    x0: number,
    y0: number,
    x1: number,
    y1: number,
    spacing: number,
    color: [number, number, number, number],
    skipStart = 0,
    skipEnd = 0
  ) {
    // Compute the angle of the line
    const angle = Math.atan2(y1 - y0, x1 - x0)
    // Compute the length of the line
    const x = x1 - x0
    const y = y1 - y0
    const length = Math.sqrt(x ** 2 + y ** 2)
    // Compute the number of dashes
    const numDashes = Math.floor(length / spacing) + 1
    // Compute the delta of each dash
    const dx = x / numDashes
    const dy = y / numDashes
    // Draw the dashes
    for (let i = 0; i < numDashes; i++) {
      if (i < skipStart || i >= numDashes - skipEnd) {
        continue
      }
      this.drawSprite(imageName, x0 + i * dx, y0 + i * dy, color, 1, -angle)
    }
  }
}
