
// TODO First step, stub everything out and get things to start up

import { Mat3f } from './vector_math.js'

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
/** Type definition for atlas data. */
interface AtlasData {
  [key: string]: [number, number, number, number] // [x, y, width, height]
}

export class ContextWebgl {
  public canvas: HTMLCanvasElement
  public gl: WebGLRenderingContext
  public ready: boolean = false
  public dpr: number = 1
  public atlasData: AtlasData | null = null

  // WebGL rendering state
  private shaderProgram: WebGLProgram | null = null
  private atlasTexture: WebGLTexture | null = null
  private textureSize: { width: number; height: number } = { width: 0, height: 0 }
  private atlasMargin: number = 4

  // Shader locations
  private positionLocation: number = -1
  private texcoordLocation: number = -1
  private colorLocation: number = -1
  private canvasSizeLocation: WebGLUniformLocation | null = null
  private samplerLocation: WebGLUniformLocation | null = null

  // Mesh management
  private meshes: Map<string, WebGLMesh> = new Map()
  private currentMesh: WebGLMesh | null = null
  private currentMeshName: string = ''

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

    // Initialize transformation matrix
    this.currentTransform = Mat3f.identity()

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
  
  /** Sets the scissor rect for the current mesh. */
  setScissorRect(x: number, y: number, width: number, height: number): void {
    this.ensureMeshSelected()

    this.currentMesh!.scissorEnabled = true
    this.currentMesh!.scissorRect = [x, y, width, height]
  }
  
  /** Disable scissoring for the current mesh. */
  disableScissor(): void {
    this.ensureMeshSelected()
    this.currentMesh!.scissorEnabled = false
  }
  
  /** Save the current transform. */
  save(): void {
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
  restore(): void {
    // Pop the last transform from the stack
    if (this.transformStack.length > 0) {
      this.currentTransform = this.transformStack.pop()!
    } else {
      console.warn('Transform stack is empty')
    }
  }
  
  /** Translate the current transform. */
  translate(x: number, y: number): void {
    const translateMatrix = Mat3f.translate(x, y)
    this.currentTransform = this.currentTransform.mul(translateMatrix)
  }
  
  /** Rotate the current transform. */
  rotate(angle: number): void {
    const rotateMatrix = Mat3f.rotate(angle)
    this.currentTransform = this.currentTransform.mul(rotateMatrix)
  }
  
  /** Scale the current transform. */
  scale(x: number, y: number): void {
    const scaleMatrix = Mat3f.scale(x, y)
    this.currentTransform = this.currentTransform.mul(scaleMatrix)
  }
  
  /** Reset the current transform. */
  resetTransform(): void {
    this.currentTransform = Mat3f.identity()
  }
  
  /** Initialize the WebGL context. */
  async init(atlasJsonUrl: string, atlasImageUrl: string): Promise<boolean> {
    this.dpr = 1.0
    if (window.devicePixelRatio > 1.0) {
      this.dpr = 2.0 // Retina display only, we don't support other DPI scales.
    }

    // Load Atlas and Texture.
    const [atlasData, source] = await Promise.all([
      this.loadAtlasJson(atlasJsonUrl),
      this.loadAtlasImage(atlasImageUrl),
    ])

    if (!atlasData || !source) {
      this.fail('Failed to load atlas or texture')
      return false
    }
    this.atlasData = atlasData
    this.textureSize = { width: source.width, height: source.height }

    // Create and compile shaders
    const vertexShader = this.createShader(this.gl.VERTEX_SHADER, this.getVertexShaderSource())
    const fragmentShader = this.createShader(this.gl.FRAGMENT_SHADER, this.getFragmentShaderSource())

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

    // Create texture
    this.atlasTexture = this.gl.createTexture()
    if (!this.atlasTexture) {
      this.fail('Failed to create texture')
      return false
    }

    // Upload texture data
    this.gl.bindTexture(this.gl.TEXTURE_2D, this.atlasTexture)
    this.gl.texImage2D(this.gl.TEXTURE_2D, 0, this.gl.RGBA, this.gl.RGBA, this.gl.UNSIGNED_BYTE, source)

    // Generate mipmaps
    this.gl.generateMipmap(this.gl.TEXTURE_2D)

    // Set texture parameters
    this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_S, this.gl.REPEAT)
    this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_T, this.gl.REPEAT)
    this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MIN_FILTER, this.gl.LINEAR_MIPMAP_LINEAR)
    this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MAG_FILTER, this.gl.LINEAR)

    // Enable blending for premultiplied alpha
    this.gl.enable(this.gl.BLEND)
    this.gl.blendFunc(this.gl.ONE, this.gl.ONE_MINUS_SRC_ALPHA)

    this.ready = true
    return true
  }

  /** Fail the context. */
  private fail(msg: string): void {
    console.error(msg)
    const failDiv = document.createElement('div')
    failDiv.id = 'fail'
    failDiv.textContent = `Initialization Error: ${msg}. See console for details.`
    document.body.appendChild(failDiv)
  }

  /** Load the atlas image. */
  private async loadAtlasImage(url: string): Promise<HTMLImageElement | null> {
    try {
      const img = new Image()
      img.crossOrigin = 'anonymous'
      return new Promise((resolve, reject) => {
        img.onload = () => resolve(img)
        img.onerror = () => reject(new Error(`Failed to load image: ${url}`))
        img.src = url
      })
    } catch (err) {
      console.error(`Error loading image ${url}:`, err)
      return null
    }
  }

  /** Load the atlas JSON. */
  private async loadAtlasJson(url: string): Promise<AtlasData | null> {
    try {
      const res = await fetch(url)
      if (!res.ok) {
        throw new Error(`Failed to fetch atlas: ${res.statusText}`)
      }
      return await res.json()
    } catch (err) {
      console.error(`Error loading atlas ${url}:`, err)
      return null
    }
  }

  /** Create and compile a shader. */
  private createShader(type: number, source: string): WebGLShader | null {
    const shader = this.gl.createShader(type)
    if (!shader) return null

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
    if (!program) return null

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

  /** Get vertex shader source. */
  private getVertexShaderSource(): string {
    return `
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
  }

  /** Get fragment shader source. */
  private getFragmentShaderSource(): string {
    return `
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
