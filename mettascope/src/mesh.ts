import type { Vec2f } from './vector_math.js'

/** Mesh class responsible for managing vertex data. */
export class Mesh {
  private static readonly FLOATS_PER_VERTEX = 8
  private static readonly BYTES_PER_VERTEX = Mesh.FLOATS_PER_VERTEX * 4
  private static readonly OFFSET_POSITION = 0
  private static readonly OFFSET_TEXCOORD = 2 * 4
  private static readonly OFFSET_COLOR = 4 * 4
  private static readonly VERTICES_PER_QUAD = 4
  private static readonly INDICES_PER_QUAD = 6

  public name: string
  public gl: WebGLRenderingContext
  public vertexBuffer: WebGLBuffer | null = null
  public indexBuffer: WebGLBuffer | null = null

  // Buffer management
  public maxQuads: number
  public vertexCapacity: number
  public indexCapacity: number
  public vertexData: Float32Array
  public indexData: Uint32Array
  public currentQuad = 0
  public currentVertex = 0

  // Scissor properties
  public scissorEnabled = false
  public scissorRect: [number, number, number, number] = [0, 0, 0, 0] // x, y, width, height

  // Caching properties
  public cacheable = false
  public isDirty = true

  constructor(name: string, gl: WebGLRenderingContext, maxQuads: number = 1024 * 8) {
    this.name = name
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

    // Create the index pattern once (it's always the same for quads)
    this.setupIndexPattern()
    this.createBuffers()
  }

  /** Set up the index buffer pattern once. */
  private setupIndexPattern() {
    // For each quad: triangles are formed by indices
    // 0-1-2 (top-left, bottom-left, top-right)
    // 2-1-3 (top-right, bottom-left, bottom-right)
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

  /** Create WebGL buffers. */
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

  /** Resize the maximum number of quads the mesh can hold. */
  private resize(newMaxQuads: number) {
    if (newMaxQuads <= this.maxQuads) {
      console.warn('New max quads must be larger than current max quads')
      return
    }

    console.info('Resizing max ', this.name, ' quads from', this.maxQuads, 'to', newMaxQuads)

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

      // Create new buffers with increased capacity
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

  /** Force clear the mesh even if it's cacheable. */
  forceClear() {
    this.currentQuad = 0
    this.currentVertex = 0
    this.isDirty = true
    this.scissorEnabled = false
    this.scissorRect = [0, 0, 0, 0]
  }

  /** Clear the mesh for a new frame. */
  clear() {
    if (this.cacheable) {
      return // Skip clearing cached meshes
    }
    this.forceClear()
  }

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
    this.isDirty = true
  }

  /** Reset the counters. */
  resetCounters() {
    if (this.cacheable) {
      return // Don't reset counters for cacheable meshes
    }
  }

  /* Upload vertex data from CPU to GPU. */
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
}
