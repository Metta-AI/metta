import { Vec2f } from './vector_math.js'

/**
 * Mesh class for efficient batched rendering of textured quads.
 *
 * A mesh accumulates draw calls into vertex and index buffers, allowing
 * thousands of sprites to be rendered in a single GPU draw call. This is
 * critical for performance in 2D games and applications.
 *
 * Features:
 * - Pre-allocated buffers with automatic resizing
 * - Batches all quads sharing the same texture
 * - Supports per-quad color tinting
 * - Optional scissor rectangle for clipping
 *
 * The mesh uses an indexed triangle list with 4 vertices per quad and
 * 6 indices (2 triangles). Vertex format: position (2), texcoord (2), color (4).
 *
 * @example
 * const mesh = new Mesh(gl)
 * mesh.clear()
 * mesh.addQuad(topLeft, bottomLeft, topRight, bottomRight, u0, v0, u1, v1, color)
 * mesh.uploadToGPU()
 * mesh.bind(posLoc, texLoc, colorLoc)
 * gl.drawElements(gl.TRIANGLES, mesh.getIndexCount(), gl.UNSIGNED_INT, 0)
 */
export class Mesh {
  /** Number of floats per vertex:
   * - x, y: position
   * - u, v: texture coordinates
   * - r, g, b, a: color
   */
  private static readonly FLOATS_PER_VERTEX = 8
  private static readonly BYTES_PER_VERTEX = Mesh.FLOATS_PER_VERTEX * 4

  /** Byte offsets into a single vertex */
  private static readonly OFFSET_POSITION = 0
  private static readonly OFFSET_TEXCOORD = 2 * 4
  private static readonly OFFSET_COLOR = 4 * 4
  private static readonly VERTICES_PER_QUAD = 4
  private static readonly INDICES_PER_QUAD = 6

  private gl: WebGLRenderingContext
  private vertexBuffer: WebGLBuffer | null = null
  private indexBuffer: WebGLBuffer | null = null

  // Buffer management
  private maxQuads: number
  private vertexCapacity: number
  private indexCapacity: number
  private vertexData: Float32Array
  private indexData: Uint32Array
  private currentQuad: number = 0
  private currentVertex: number = 0

  // Scissor properties
  public scissorEnabled: boolean = false
  public scissorRect: [number, number, number, number] = [0, 0, 0, 0] // x, y, width, height

  // Caching properties
  public cacheable = false
  public isDirty = true

  /**
   * Create a new mesh.
   *
   * @param gl - The WebGL rendering context
   * @param maxQuads - Initial capacity in quads (default: 8192)
   */
  constructor(gl: WebGLRenderingContext, maxQuads: number = 1024 * 8) {
    this.gl = gl
    this.maxQuads = maxQuads

    // Pre-allocated buffers for better performance
    this.vertexCapacity = this.maxQuads * Mesh.VERTICES_PER_QUAD // 4 vertices per quad
    this.indexCapacity = this.maxQuads * Mesh.INDICES_PER_QUAD // 6 indices per quad (2 triangles)

    // Pre-allocated CPU-side buffers

    // Vertex layout per vertex:
    // - x, y:       Position (screen space)
    // - u, v:       Texture coordinates (UV)
    // - r, g, b, a: Color multiplier (float 0-1)
    // Total: 8 floats per vertex
    this.vertexData = new Float32Array(this.vertexCapacity * Mesh.FLOATS_PER_VERTEX)
    this.indexData = new Uint32Array(this.indexCapacity)

    // Create the index pattern once
    this.setupIndexPattern()
    this.createBuffers()
  }

  /**
   * Set up the index buffer pattern for quads.
   *
   * Creates the index pattern for all quads at once. Each quad uses
   * 6 indices to form 2 triangles in counter-clockwise order:
   * - Triangle 1: top-left, bottom-left, top-right
   * - Triangle 2: top-right, bottom-left, bottom-right
   *
   * @private
   */
  private setupIndexPattern() {
    for (let i = 0; i < this.maxQuads; i++) {
      const baseVertex = i * Mesh.VERTICES_PER_QUAD
      const baseIndex = i * Mesh.INDICES_PER_QUAD

      // [Top-left, Bottom-left, Top-right, Top-right, Bottom-left, Bottom-right]
      const indexPattern = [0, 1, 2, 2, 1, 3]
      for (let j = 0; j < Mesh.INDICES_PER_QUAD; j++) {
        this.indexData[baseIndex + j] = baseVertex + indexPattern[j]
      }
    }
  }

  /**
   * Create WebGL vertex and index buffers.
   *
   * Allocates GPU memory and uploads the initial (empty) buffer data.
   * The vertex buffer uses DYNAMIC_DRAW since it's updated every frame.
   * The index buffer uses STATIC_DRAW since the pattern never changes.
   *
   * @private
   */
  private createBuffers() {
    // Create vertex buffer
    this.vertexBuffer = this.gl.createBuffer()
    if (this.vertexBuffer) {
      this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.vertexBuffer)
      this.gl.bufferData(this.gl.ARRAY_BUFFER, this.vertexData, this.gl.DYNAMIC_DRAW)
    }

    // Create index buffer
    this.indexBuffer = this.gl.createBuffer()
    if (this.indexBuffer) {
      this.gl.bindBuffer(this.gl.ELEMENT_ARRAY_BUFFER, this.indexBuffer)
      this.gl.bufferData(this.gl.ELEMENT_ARRAY_BUFFER, this.indexData, this.gl.STATIC_DRAW)
    }
  }

  /**
   * Resize the mesh to accommodate more quads.
   *
   * Automatically called when the mesh runs out of space. Creates new
   * larger buffers, copies existing data, and recreates GPU buffers.
   *
   * @param newMaxQuads - New capacity in quads (must be larger than current)
   * @private
   */
  private resize(newMaxQuads: number) {
    if (newMaxQuads <= this.maxQuads) return

    // Store current data
    const oldVertexData = this.vertexData
    const currentVertexCount = this.currentVertex

    // Update capacities
    this.maxQuads = newMaxQuads
    this.vertexCapacity = this.maxQuads * Mesh.VERTICES_PER_QUAD
    this.indexCapacity = this.maxQuads * Mesh.INDICES_PER_QUAD

    // Create new arrays
    this.vertexData = new Float32Array(this.vertexCapacity * Mesh.FLOATS_PER_VERTEX)
    this.indexData = new Uint32Array(this.indexCapacity)

    // Copy existing data
    this.vertexData.set(oldVertexData.subarray(0, currentVertexCount * Mesh.FLOATS_PER_VERTEX))

    // Rebuild index pattern
    this.setupIndexPattern()

    // Recreate buffers
    if (this.vertexBuffer && this.indexBuffer) {
      this.gl.deleteBuffer(this.vertexBuffer)
      this.gl.deleteBuffer(this.indexBuffer)
      this.createBuffers()

      // Upload existing data
      if (currentVertexCount > 0) {
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.vertexBuffer)
        this.gl.bufferSubData(
          this.gl.ARRAY_BUFFER,
          0,
          this.vertexData.subarray(0, currentVertexCount * Mesh.FLOATS_PER_VERTEX)
        )
      }
    }
  }

  /**
   * Clear the mesh for a new frame.
   *
   * Resets the vertex and quad counters without deallocating memory.
   * Should be called at the start of each frame before adding new quads.
   */
  clear() {
    this.currentQuad = 0
    this.currentVertex = 0
  }

  /**
   * Add a quad to the mesh with pre-transformed vertices.
   *
   * The vertices should already have transformations applied. This method
   * just stores them in the vertex buffer with texture coordinates and color.
   * Automatically resizes the mesh if it runs out of space.
   *
   * @param topLeft - Top-left vertex position
   * @param bottomLeft - Bottom-left vertex position
   * @param topRight - Top-right vertex position
   * @param bottomRight - Bottom-right vertex position
   * @param u0 - Left texture coordinate (0-1)
   * @param v0 - Top texture coordinate (0-1)
   * @param u1 - Right texture coordinate (0-1)
   * @param v1 - Bottom texture coordinate (0-1)
   * @param color - RGBA color multiplier [r, g, b, a] where each component is 0-1
   *
   * @example
   * mesh.addQuad(
   *   new Vec2f(0, 0),      // top-left
   *   new Vec2f(0, 32),     // bottom-left
   *   new Vec2f(32, 0),     // top-right
   *   new Vec2f(32, 32),    // bottom-right
   *   0, 0, 1, 1,           // full texture
   *   [1, 1, 1, 1]          // white (no tint)
   * )
   */
  addQuad(
    topLeft: Vec2f,
    bottomLeft: Vec2f,
    topRight: Vec2f,
    bottomRight: Vec2f,
    u0: number,
    v0: number,
    u1: number,
    v1: number,
    color: [number, number, number, number] = [1, 1, 1, 1]
  ) {
    // Check if we need to resize
    if (this.currentQuad >= this.maxQuads) {
      this.resize(this.maxQuads * 2)
    }

    const baseOffset = this.currentVertex * Mesh.FLOATS_PER_VERTEX

    // Define the vertex attributes for each corner
    const corners = [
      { pos: topLeft, uv: [u0, v0] },
      { pos: bottomLeft, uv: [u0, v1] },
      { pos: topRight, uv: [u1, v0] },
      { pos: bottomRight, uv: [u1, v1] },
    ]

    // Set vertex data
    for (let i = 0; i < Mesh.VERTICES_PER_QUAD; i++) {
      const offset = baseOffset + i * Mesh.FLOATS_PER_VERTEX
      const corner = corners[i]

      // Position
      this.vertexData[offset + 0] = corner.pos.x()
      this.vertexData[offset + 1] = corner.pos.y()

      // Texture coordinates
      this.vertexData[offset + 2] = corner.uv[0]
      this.vertexData[offset + 3] = corner.uv[1]

      // Color
      this.vertexData[offset + 4] = color[0]
      this.vertexData[offset + 5] = color[1]
      this.vertexData[offset + 6] = color[2]
      this.vertexData[offset + 7] = color[3]
    }

    this.currentVertex += Mesh.VERTICES_PER_QUAD
    this.currentQuad += 1
  }

  /**
   * Upload vertex data from CPU to GPU.
   *
   * Copies the vertex data to the GPU vertex buffer. Should be called
   * after all quads have been added and before rendering.
   */
  uploadToGPU() {
    if (!this.vertexBuffer || this.currentVertex === 0) return

    const vertexDataCount = this.currentVertex * Mesh.FLOATS_PER_VERTEX
    this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.vertexBuffer)
    this.gl.bufferSubData(this.gl.ARRAY_BUFFER, 0, this.vertexData.subarray(0, vertexDataCount))
  }

  /**
   * Bind mesh buffers and set up vertex attributes.
   *
   * Prepares the mesh for rendering by binding buffers and configuring
   * vertex attribute pointers. Must be called before gl.drawElements().
   *
   * @param positionLoc - Shader attribute location for position
   * @param texcoordLoc - Shader attribute location for texture coordinates
   * @param colorLoc - Shader attribute location for color
   */
  bind(positionLoc: number, texcoordLoc: number, colorLoc: number) {
    if (!this.vertexBuffer || !this.indexBuffer) return

    this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.vertexBuffer)

    this.gl.enableVertexAttribArray(positionLoc)
    this.gl.vertexAttribPointer(positionLoc, 2, this.gl.FLOAT, false, Mesh.BYTES_PER_VERTEX, Mesh.OFFSET_POSITION)

    this.gl.enableVertexAttribArray(texcoordLoc)
    this.gl.vertexAttribPointer(texcoordLoc, 2, this.gl.FLOAT, false, Mesh.BYTES_PER_VERTEX, Mesh.OFFSET_TEXCOORD)

    this.gl.enableVertexAttribArray(colorLoc)
    this.gl.vertexAttribPointer(colorLoc, 4, this.gl.FLOAT, false, Mesh.BYTES_PER_VERTEX, Mesh.OFFSET_COLOR)

    this.gl.bindBuffer(this.gl.ELEMENT_ARRAY_BUFFER, this.indexBuffer)
  }

  /**
   * Get the number of indices to draw.
   *
   * @returns Number of indices (6 per quad)
   */
  getIndexCount(): number {
    return this.currentQuad * Mesh.INDICES_PER_QUAD
  }

  /**
   * Check if the mesh has any content to render.
   *
   * @returns True if there are quads to draw
   */
  hasContent(): boolean {
    return this.currentQuad > 0
  }

  /**
   * Destroy GPU resources.
   *
   * Deletes vertex and index buffers. Should be called when the mesh
   * is no longer needed to free GPU memory.
   */
  destroy() {
    if (this.vertexBuffer) {
      this.gl.deleteBuffer(this.vertexBuffer)
      this.vertexBuffer = null
    }
    if (this.indexBuffer) {
      this.gl.deleteBuffer(this.indexBuffer)
      this.indexBuffer = null
    }
  }
}
