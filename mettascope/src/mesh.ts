import type { Vec2f } from './vector_math.js'

/** Mesh class responsible for managing vertex data. */
export class Mesh {
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
    this.vertexCapacity = this.maxQuads * 4 // 4 vertices per quad
    this.indexCapacity = this.maxQuads * 6 // 6 indices per quad (2 triangles)

    // Pre-allocated CPU-side buffers
    this.vertexData = new Float32Array(this.vertexCapacity * 8) // 8 floats per vertex (pos*2, uv*2, color*4)
    this.indexData = new Uint32Array(this.indexCapacity)

    // Create the index pattern once (it's always the same for quads)
    this.setupIndexPattern()
  }

  /** Set up the index buffer pattern once. */
  setupIndexPattern() {
    // For each quad: triangles are formed by indices
    // 0-1-2 (top-left, bottom-left, top-right)
    // 2-1-3 (top-right, bottom-left, bottom-right)
    for (let i = 0; i < this.maxQuads; i++) {
      const baseVertex = i * 4
      const baseIndex = i * 6

      // [Top-left, Bottom-left, Top-right, Top-right, Bottom-left, Bottom-right]
      const indexPattern = [0, 1, 2, 2, 1, 3]
      for (let j = 0; j < 6; j++) {
        this.indexData[baseIndex + j] = baseVertex + indexPattern[j]
      }
    }
  }

  /** Create WebGL buffers. */
  createBuffers() {
    if (!this.gl) {
      return
    }

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
  resizeMaxQuads(newMaxQuads: number) {
    console.info('Resizing max ', this.name, ' quads from', this.maxQuads, 'to', newMaxQuads)

    if (newMaxQuads <= this.maxQuads) {
      console.warn('New max quads must be larger than current max quads')
      return
    }

    // Store the current data and state
    const oldVertexData = this.vertexData
    const currentVertexCount = this.currentVertex

    // Update capacities
    this.maxQuads = newMaxQuads
    this.vertexCapacity = this.maxQuads * 4 // 4 vertices per quad
    this.indexCapacity = this.maxQuads * 6 // 6 indices per quad (2 triangles)

    // Create new CPU-side arrays with increased capacity
    this.vertexData = new Float32Array(this.vertexCapacity * 8) // 8 floats per vertex
    this.indexData = new Uint32Array(this.indexCapacity)

    // Copy existing vertex data to the new array
    this.vertexData.set(oldVertexData.subarray(0, currentVertexCount * 8))

    // Rebuild index data (includes the new pattern for additional quads)
    this.setupIndexPattern()

    // If we already have WebGL buffers, we need to recreate them
    if (this.vertexBuffer && this.indexBuffer && this.gl) {
      // Delete old buffers
      this.gl.deleteBuffer(this.vertexBuffer)
      this.gl.deleteBuffer(this.indexBuffer)

      // Create new buffers with increased capacity
      this.createBuffers()

      // Write the existing vertex data to the new vertex buffer
      if (currentVertexCount > 0) {
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.vertexBuffer)
        this.gl.bufferSubData(this.gl.ARRAY_BUFFER, 0, this.vertexData.subarray(0, currentVertexCount * 8))
      }
    }
  }

  /** Clear the mesh for a new frame. */
  clear() {
    if (this.cacheable) {
      return // Skip clearing cached meshes
    }

    // Reset counters instead of recreating arrays
    this.currentQuad = 0
    this.currentVertex = 0
    this.isDirty = true

    // Reset scissor settings
    this.scissorEnabled = false
    this.scissorRect = [0, 0, 0, 0]
  }

  /** Force clear the mesh even if it's cacheable. */
  forceClear() {
    this.currentQuad = 0
    this.currentVertex = 0
    this.isDirty = true
    this.scissorEnabled = false
    this.scissorRect = [0, 0, 0, 0]
  }

  /** Draws a pre-transformed textured rectangle. */
  drawRectWithTransform(
    topLeft: Vec2f,
    bottomLeft: Vec2f,
    topRight: Vec2f,
    bottomRight: Vec2f,
    u0: number,
    v0: number,
    u1: number,
    v1: number,
    color: number[] = [1, 1, 1, 1]
  ) {
    // Check if we need to resize before adding more vertices
    if (this.currentQuad >= this.maxQuads) {
      this.resizeMaxQuads(this.maxQuads * 2)
    }

    // Calculate base offset for this quad in the vertex data array
    const baseVertex = this.currentVertex
    const baseOffset = baseVertex * 8 // Each vertex has 8 floats

    // Define the vertex attributes for each corner
    const corners = [
      { pos: topLeft, uv: [u0, v0] }, // Top-left
      { pos: bottomLeft, uv: [u0, v1] }, // Bottom-left
      { pos: topRight, uv: [u1, v0] }, // Top-right
      { pos: bottomRight, uv: [u1, v1] }, // Bottom-right
    ]

    // Loop through each corner and set its vertex data
    for (let i = 0; i < 4; i++) {
      const offset = baseOffset + i * 8
      const corner = corners[i]

      // Position
      this.vertexData[offset + 0] = corner.pos.x()
      this.vertexData[offset + 1] = corner.pos.y()

      // Texture coordinates
      this.vertexData[offset + 2] = corner.uv[0]
      this.vertexData[offset + 3] = corner.uv[1]

      // Color (same for all vertices)
      this.vertexData[offset + 4] = color[0]
      this.vertexData[offset + 5] = color[1]
      this.vertexData[offset + 6] = color[2]
      this.vertexData[offset + 7] = color[3]
    }

    // Update counters
    this.currentVertex += 4
    this.currentQuad += 1
    this.isDirty = true
  }

  /** Reset the counters. */
  resetCounters() {
    if (this.cacheable) {
      return // Don't reset counters for cacheable meshes
    }

    this.currentQuad = 0
    this.currentVertex = 0
    this.isDirty = true
  }
}
