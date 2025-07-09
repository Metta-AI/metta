
// TODO First step, stub everything out and get things to start up

/** WebGL Mesh class for managing vertex data and buffers. */
class WebGLMesh {
  private name: string
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
  setupIndexPattern(): void {
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
  createBuffers(): void {
    if (!this.gl) return

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

  /** Clear the mesh for a new frame. */
  clear(): void {
    // Reset counters instead of recreating arrays
    this.currentQuad = 0
    this.currentVertex = 0

    // Reset scissor settings
    this.scissorEnabled = false
    this.scissorRect = [0, 0, 0, 0]
  }

  /** Get the number of quads in the mesh. */
  getQuadCount(): number {
    return this.currentQuad
  }

  /** Get the current vertex count. */
  getCurrentVertexCount(): number {
    return this.currentVertex
  }

  /** Get the vertex buffer. */
  getVertexBuffer(): WebGLBuffer | null {
    return this.vertexBuffer
  }

  /** Get the index buffer. */
  getIndexBuffer(): WebGLBuffer | null {
    return this.indexBuffer
  }

  /** Get the vertex data. */
  getVertexData(): Float32Array {
    return this.vertexData
  }

  /** Reset the counters. */
  resetCounters(): void {
    this.currentQuad = 0
    this.currentVertex = 0
  }
}

/** ContextWebgl is a class that manages the WebGL context. */
export class ContextWebgl {
  public canvas: HTMLCanvasElement
  public gl: WebGLRenderingContext
  public ready: boolean = false
  public dpr: number = 1
  public atlasData: any = null

  // Mesh management
  private meshes: Map<string, WebGLMesh> = new Map()
  private currentMesh: WebGLMesh | null = null
  private currentMeshName: string = ''

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas
    const gl = canvas.getContext('webgl')
    if (!gl) {
      throw new Error('Failed to get WebGL context')
    }
    this.gl = gl
    console.log('constructor done')
  }

  
  /** Clears all meshes for a new frame. */
  clear(): void {
    if (!this.ready) return

    // Clear all meshes in the map
    for (const mesh of this.meshes.values()) {
      mesh.clear()
    }
  }
  
  /** Create or switch to a mesh with the given name. */
  useMesh(name: string): void {
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
    const newMesh = new WebGLMesh(name, this.gl)
    newMesh.createBuffers()
    this.meshes.set(name, newMesh)
    this.currentMesh = newMesh
    this.currentMeshName = name
  }

  /** Helper method to ensure a mesh is selected before drawing. */
  private ensureMeshSelected(): void {
    if (!this.currentMesh) {
      throw new Error('No mesh selected. Call useMesh() before drawing.')
    }
  }
  
  setScissorRect(x: number, y: number, width: number, height: number): void {
    console.log('setScissorRect stub', x, y, width, height)
  }
  
  disableScissor(): void {
    console.log('disableScissor stub')
  }
  
  save(): void {
    console.log('save stub')
  }
  
  restore(): void {
    console.log('restore stub')
  }
  
  translate(x: number, y: number): void {
    console.log('translate stub', x, y)
  }
  
  rotate(angle: number): void {
    console.log('rotate stub', angle)
  }
  
  scale(x: number, y: number): void {
    console.log('scale stub', x, y)
  }
  
  resetTransform(): void {
    console.log('resetTransform stub')
  }
  
  async init(atlasJsonUrl: string, atlasImageUrl: string): Promise<boolean> {
    console.log('init stub', atlasJsonUrl, atlasImageUrl)
    this.ready = true
    return true
  }
  
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
  ): void {
    console.log('drawRect stub', x, y, width, height, u0, v0, u1, v1, color)
  }

  hasImage(imageName: string): boolean {
    console.log('hasImage stub', imageName)
    return false
  }

  drawImage(imageName: string, x: number, y: number, color: number[] = [1, 1, 1, 1]): void {
    console.log('drawImage stub', imageName, x, y, color)
  }

  drawSprite(imageName: string, x: number, y: number, color: number[] = [1, 1, 1, 1], scale = 1, rotation = 0): void {
    console.log('drawSprite stub', imageName, x, y, color, scale, rotation)
  }

  drawSolidRect(x: number, y: number, width: number, height: number, color: number[]): void {
    console.log('drawSolidRect stub', x, y, width, height, color)
  }

  drawStrokeRect(x: number, y: number, width: number, height: number, strokeWidth: number, color: number[]): void {
    console.log('drawStrokeRect stub', x, y, width, height, strokeWidth, color)
  }

  drawSpriteLine(
    imageName: string,
    x0: number,
    y0: number,
    x1: number,
    y1: number,
    spacing: number,
    color: number[],
    skipStart: number = 0,
    skipEnd: number = 0
  ): void {
    console.log('drawSpriteLine stub', imageName, x0, y0, x1, y1, spacing, color, skipStart, skipEnd)
  }

  flush(): void {
    console.log('flush stub')
  }
}
