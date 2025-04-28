/// <reference types="@webgpu/types" />

import { Vec2f, Mat3f } from './vector_math.js';

// Type definition for atlas data
interface AtlasData {
  [key: string]: [number, number, number, number]; // [x, y, width, height]
}

/**
 * Mesh class responsible for managing vertex data
 */
class Mesh {
  private device: GPUDevice;
  private vertexBuffer: GPUBuffer | null = null;
  private indexBuffer: GPUBuffer | null = null;

  // Buffer management
  private maxQuads: number;
  private vertexCapacity: number;
  private indexCapacity: number;
  private vertexData: Float32Array;
  private indexData: Uint32Array;
  private currentQuad: number = 0;
  private currentVertex: number = 0;

  // Scissor properties
  public scissorEnabled: boolean = false;
  public scissorRect: [number, number, number, number] = [0, 0, 0, 0]; // x, y, width, height

  constructor(device: GPUDevice, maxQuads: number = 65536) {
    this.device = device;
    this.maxQuads = maxQuads;

    // Pre-allocated buffers for better performance
    this.vertexCapacity = this.maxQuads * 4; // 4 vertices per quad
    this.indexCapacity = this.maxQuads * 6; // 6 indices per quad (2 triangles)

    // Pre-allocated CPU-side buffers
    this.vertexData = new Float32Array(this.vertexCapacity * 8); // 8 floats per vertex (pos*2, uv*2, color*4)
    this.indexData = new Uint32Array(this.indexCapacity);

    // Create the index pattern once (it's always the same for quads)
    this.setupIndexPattern();
  }

  // Set up the index buffer pattern once
  setupIndexPattern(): void {
    // For each quad: triangles are formed by indices
    // 0-1-2 (top-left, bottom-left, top-right)
    // 2-1-3 (top-right, bottom-left, bottom-right)
    for (let i = 0; i < this.maxQuads; i++) {
      const baseVertex = i * 4;
      const baseIndex = i * 6;

      // [Top-left, Bottom-left, Top-right, Top-right, Bottom-left, Bottom-right]
      const indexPattern = [0, 1, 2, 2, 1, 3];
      for (let j = 0; j < 6; j++) {
        this.indexData[baseIndex + j] = baseVertex + indexPattern[j];
      }
    }
  }

  // Create GPU buffers
  createBuffers(): void {
    if (!this.device) return;

    // Create vertex buffer
    this.vertexBuffer = this.device.createBuffer({
      label: 'vertex buffer',
      size: this.vertexCapacity * 8 * Float32Array.BYTES_PER_ELEMENT,
      // x, y, u, v, r, g, b, a
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });

    // Create index buffer
    this.indexBuffer = this.device.createBuffer({
      label: 'index buffer',
      size: this.indexCapacity * Uint32Array.BYTES_PER_ELEMENT,
      // Using 32-bit indices
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
    });

    // Write the index pattern to the GPU immediately (it never changes)
    this.device.queue.writeBuffer(
      this.indexBuffer,
      0,
      this.indexData,
      0,
      this.indexData.length
    );
  }

  // Clear the mesh for a new frame
  clear(): void {
    // Reset counters instead of recreating arrays
    this.currentQuad = 0;
    this.currentVertex = 0;

    // Reset scissor settings
    this.scissorEnabled = false;
    this.scissorRect = [0, 0, 0, 0];
  }

  // Draws a pre-transformed textured rectangle
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
  ): void {
    // Check if we need to flush before adding more vertices
    if (this.currentQuad >= this.maxQuads) {
      throw new Error("Max quads reached");
    }

    // Calculate base offset for this quad in the vertex data array
    const baseVertex = this.currentVertex;
    const baseOffset = baseVertex * 8; // Each vertex has 8 floats

    // Define the vertex attributes for each corner
    const corners = [
      { pos: topLeft, uv: [u0, v0] },      // Top-left
      { pos: bottomLeft, uv: [u0, v1] },   // Bottom-left
      { pos: topRight, uv: [u1, v0] },     // Top-right
      { pos: bottomRight, uv: [u1, v1] }   // Bottom-right
    ];

    // Loop through each corner and set its vertex data
    for (let i = 0; i < 4; i++) {
      const offset = baseOffset + i * 8;
      const corner = corners[i];

      // Position
      this.vertexData[offset + 0] = corner.pos.x();
      this.vertexData[offset + 1] = corner.pos.y();

      // Texture coordinates
      this.vertexData[offset + 2] = corner.uv[0];
      this.vertexData[offset + 3] = corner.uv[1];

      // Color (same for all vertices)
      this.vertexData[offset + 4] = color[0];
      this.vertexData[offset + 5] = color[1];
      this.vertexData[offset + 6] = color[2];
      this.vertexData[offset + 7] = color[3];
    }

    // Update counters
    this.currentVertex += 4;
    this.currentQuad += 1;
  }

  // Get the current number of quads
  getQuadCount(): number {
    return this.currentQuad;
  }

  // Get the vertex data for upload to GPU
  getVertexData(): Float32Array {
    return this.vertexData;
  }

  // Get the number of vertices currently in use
  getCurrentVertexCount(): number {
    return this.currentVertex;
  }

  // Get the vertex buffer
  getVertexBuffer(): GPUBuffer | null {
    return this.vertexBuffer;
  }

  // Get the index buffer
  getIndexBuffer(): GPUBuffer | null {
    return this.indexBuffer;
  }

  // Reset counters after rendering
  resetCounters(): void {
    this.currentQuad = 0;
    this.currentVertex = 0;
  }
}

class Drawer {
  // Canvas and WebGPU state
  public canvas: HTMLCanvasElement;
  public device: GPUDevice | null;
  private context: GPUCanvasContext | null;
  private pipeline: GPURenderPipeline | null;
  private sampler: GPUSampler | null;
  private atlasTexture: GPUTexture | null;
  private textureSize: Vec2f;
  public atlasData: AtlasData | null;
  private bindGroup: GPUBindGroup | null;
  private renderPassDescriptor: GPURenderPassDescriptor | null;
  private canvasSizeUniformBuffer: GPUBuffer | null;
  private canvasSize: Vec2f;
  private atlasMargin: number;

  // Mesh management
  private meshes: Map<string, Mesh> = new Map();
  private currentMesh: Mesh | null = null;
  private currentMeshName: string = "";

  // Transformation state
  private currentTransform: Mat3f;
  private transformStack: Mat3f[] = [];

  // State tracking
  public ready: boolean;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    this.device = null;
    this.context = null;
    this.pipeline = null;
    this.sampler = null;
    this.atlasTexture = null;
    this.textureSize = new Vec2f(0, 0);
    this.atlasData = null;
    this.bindGroup = null;
    this.renderPassDescriptor = null;
    this.canvasSizeUniformBuffer = null;
    this.canvasSize = new Vec2f(0, 0);
    this.atlasMargin = 4; // Default margin for texture sampling.

    // Initialize transformation matrix
    this.currentTransform = Mat3f.identity();

    this.ready = false;
  }

  // Create or switch to a mesh with the given name
  useMesh(name: string): void {
    if (!this.device || !this.ready) {
      throw new Error("Cannot use mesh before initialization");
    }

    // If we already have this mesh, set it as current
    if (this.meshes.has(name)) {
      this.currentMesh = this.meshes.get(name)!;
      this.currentMeshName = name;
      return;
    }

    // Otherwise, create a new mesh
    const newMesh = new Mesh(this.device);
    newMesh.createBuffers();
    this.meshes.set(name, newMesh);
    this.currentMesh = newMesh;
    this.currentMeshName = name;
  }

  // Sets the scissor rect for the current mesh
  setScissorRect(x: number, y: number, width: number, height: number): void {
    this.ensureMeshSelected();

    this.currentMesh!.scissorEnabled = true;
    this.currentMesh!.scissorRect = [x, y, Math.max(width, 0), Math.max(height, 0)];
  }

  // Disable scissoring for the current mesh
  disableScissor(): void {
    this.ensureMeshSelected();
    this.currentMesh!.scissorEnabled = false;
  }

  // Helper method to ensure a mesh is selected before drawing
  private ensureMeshSelected(): void {
    if (!this.currentMesh) {
      throw new Error("No mesh selected. Call useMesh() before drawing.");
    }
  }

  // Transform manipulation methods
  save(): void {
    // Push a copy of the current transform onto the stack
    this.transformStack.push(new Mat3f(
      this.currentTransform.get(0, 0), this.currentTransform.get(0, 1), this.currentTransform.get(0, 2),
      this.currentTransform.get(1, 0), this.currentTransform.get(1, 1), this.currentTransform.get(1, 2),
      this.currentTransform.get(2, 0), this.currentTransform.get(2, 1), this.currentTransform.get(2, 2)
    ));
  }

  restore(): void {
    // Pop the last transform from the stack
    if (this.transformStack.length > 0) {
      this.currentTransform = this.transformStack.pop()!;
    } else {
      console.warn("Transform stack is empty");
    }
  }

  translate(x: number, y: number): void {
    // Create a translation matrix and multiply current transform by it
    const translateMatrix = Mat3f.translate(x, y);
    this.currentTransform = this.currentTransform.mul(translateMatrix);
  }

  rotate(angle: number): void {
    // Create a rotation matrix and multiply current transform by it
    const rotateMatrix = Mat3f.rotate(angle);
    this.currentTransform = this.currentTransform.mul(rotateMatrix);
  }

  scale(x: number, y: number): void {
    // Create a scaling matrix and multiply current transform by it
    const scaleMatrix = Mat3f.scale(x, y);
    this.currentTransform = this.currentTransform.mul(scaleMatrix);
  }

  // Reset transform to identity
  resetTransform(): void {
    this.currentTransform = Mat3f.identity();
  }

  async init(atlasJsonUrl: string, atlasImageUrl: string): Promise<boolean> {
    // Initialize WebGPU device.
    const adapter = await navigator.gpu?.requestAdapter();
    this.device = await adapter?.requestDevice() || null;
    if (!this.device) {
      this.fail('Need a browser that supports WebGPU');
      return false;
    }

    // Load Atlas and Texture.
    const [atlasData, source] = await Promise.all([
      this.loadAtlasJson(atlasJsonUrl),
      this.loadAtlasImage(atlasImageUrl)
    ]);

    if (!atlasData || !source) {
      this.fail('Failed to load atlas or texture');
      return false;
    }
    this.atlasData = atlasData;
    this.textureSize = new Vec2f(source.width, source.height);

    // Configure Canvas.
    this.context = this.canvas.getContext('webgpu');
    if (!this.context) {
      this.fail('Failed to get WebGPU context');
      return false;
    }

    const presentationFormat = navigator.gpu.getPreferredCanvasFormat();
    this.context.configure({
      device: this.device,
      format: presentationFormat,
    });

    // Calculate number of mip levels.
    const mipLevels = Math.floor(Math.log2(Math.max(this.textureSize.x(), this.textureSize.y()))) + 1;
    // Create Texture and Sampler.
    this.atlasTexture = this.device.createTexture({
      label: atlasImageUrl,
      format: 'rgba8unorm',
      size: [this.textureSize.x(), this.textureSize.y()],
      usage: GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_DST |
        GPUTextureUsage.RENDER_ATTACHMENT,
      mipLevelCount: mipLevels,
    });
    this.device.queue.copyExternalImageToTexture(
      { source, flipY: false }, // Don't flip Y if UVs start top-left.
      { texture: this.atlasTexture },
      { width: this.textureSize.x(), height: this.textureSize.y() },
    );

    // Generate mipmaps for the texture.
    this.generateMipmaps(this.atlasTexture, this.textureSize.x(), this.textureSize.y());

    this.sampler = this.device.createSampler({
      addressModeU: 'repeat',
      addressModeV: 'repeat',
      magFilter: 'linear', // Normal smooth style (was 'nearest').
      minFilter: 'linear', // Normal smooth style (was 'nearest').
      mipmapFilter: 'linear', // Linear filtering between mipmap levels.
    });

    this.canvasSizeUniformBuffer = this.device.createBuffer({
      label: 'canvas size uniform buffer',
      size: 2 * Float32Array.BYTES_PER_ELEMENT, // vec2f (width, height).
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Shader Module.
    const shaderModule = this.device.createShaderModule({
      label: 'Sprite Shader Module',
      code: `
        struct VertexInput {
          @location(0) position: vec2f,
          @location(1) texcoord: vec2f,
          @location(2) color: vec4f,
        };

        struct VertexOutput {
          @builtin(position) position: vec4f,
          @location(0) texcoord: vec2f,
          @location(1) color: vec4f,
        };

        struct CanvasInfo {
          resolution: vec2f,
        };
        @group(0) @binding(2) var<uniform> canvas: CanvasInfo;

        @vertex fn vs(vert: VertexInput) -> VertexOutput {
          var out: VertexOutput;
          let zero_to_one = vert.position / canvas.resolution;
          let zero_to_two = zero_to_one * 2.0;
          let clip_space = zero_to_two - vec2f(1.0, 1.0);
          out.position = vec4f(clip_space.x, -clip_space.y, 0.0, 1.0);
          out.texcoord = vert.texcoord;
          out.color = vert.color;
          return out;
        }

        @group(0) @binding(0) var imgSampler: sampler;
        @group(0) @binding(1) var imgTexture: texture_2d<f32>;

        @fragment fn fs(in: VertexOutput) -> @location(0) vec4f {
          let texColor = textureSample(imgTexture, imgSampler, in.texcoord);
          // Do the premultiplied alpha conversion.
          let premultipliedColor = vec4f(texColor.rgb * texColor.a, texColor.a);
          return premultipliedColor * in.color;
        }
      `,
    });

    // Render Pipeline.
    this.pipeline = this.device.createRenderPipeline({
      label: 'Sprite Render Pipeline',
      layout: 'auto',
      vertex: {
        module: shaderModule,
        entryPoint: 'vs',
        buffers: [
          {
            // Vertex buffer layout.
            arrayStride: 8 * Float32Array.BYTES_PER_ELEMENT, // 2 pos, 2 uv, 4 color.
            attributes: [
              { shaderLocation: 0, offset: 0, format: 'float32x2' }, // Position.
              { shaderLocation: 1, offset: 2 * Float32Array.BYTES_PER_ELEMENT, format: 'float32x2' }, // Texcoord.
              { shaderLocation: 2, offset: 4 * Float32Array.BYTES_PER_ELEMENT, format: 'float32x4' }, // Color.
            ],
          },
        ],
      },
      fragment: {
        module: shaderModule,
        entryPoint: 'fs',
        targets: [{
          format: presentationFormat,
          blend: {
            color: {
              srcFactor: 'one',
              dstFactor: 'one-minus-src-alpha',
              operation: 'add'
            },
            alpha: {
              srcFactor: 'one',
              dstFactor: 'one-minus-src-alpha',
              operation: 'add'
            }
          }
        }],
      },
      primitive: {
        topology: 'triangle-list', // Each sprite is 2 triangles.
      },
    });

    // Bind Group for the pipeline.
    this.bindGroup = this.device.createBindGroup({
      label: 'Sprite Bind Group',
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this.sampler },
        { binding: 1, resource: this.atlasTexture.createView() },
        { binding: 2, resource: { buffer: this.canvasSizeUniformBuffer } },
      ],
    });

    // Render Pass Descriptor for the pipeline.
    this.renderPassDescriptor = {
      label: 'Canvas Render Pass',
      colorAttachments: [
        {
          // View is acquired later.
          clearValue: { r: 0.1, g: 0.1, b: 0.1, a: 1.0 }, // Dark grey clear.
          loadOp: 'clear',
          storeOp: 'store',
          view: undefined!, // This is set just before render in flush()
        },
      ] as GPURenderPassColorAttachment[],
    } as GPURenderPassDescriptor;

    this.ready = true;
    return true;
  }

  fail(msg: string): void {
    console.error(msg);
    const failDiv = document.createElement('div');
    failDiv.id = 'fail';
    failDiv.textContent =
      `Initialization Error: ${msg}. See console for details.`;
    document.body.appendChild(failDiv);
  }

  async loadAtlasImage(url: string): Promise<ImageBitmap | null> {
    try {
      const res = await fetch(url);
      if (!res.ok) {
        throw new Error(`Failed to fetch image: ${res.statusText}`);
      }
      const blob = await res.blob();
      // Use premultiplied alpha to fix border issues
      return await createImageBitmap(blob, {
        colorSpaceConversion: 'none',
        premultiplyAlpha: 'premultiply'
      });
    } catch (err) {
      console.error(`Error loading image ${url}:`, err);
      return null;
    }
  }

  async loadAtlasJson(url: string): Promise<AtlasData | null> {
    try {
      const res = await fetch(url);
      if (!res.ok) {
        throw new Error(`Failed to fetch atlas: ${res.statusText}`);
      }
      return await res.json();
    } catch (err) {
      console.error(`Error loading atlas ${url}:`, err);
      return null;
    }
  }

  // Clears all meshes for a new frame
  clear(): void {
    if (!this.ready) return;

    // Clear all meshes in the map
    for (const mesh of this.meshes.values()) {
      mesh.clear();
    }

    // Reset transform for new frame
    this.resetTransform();
    this.transformStack = [];
  }

  // Draws a textured rectangle with the given coordinates and UV mapping.
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
    if (!this.ready) {
      throw new Error("Drawer not initialized");
    }

    this.ensureMeshSelected();

    const pos = new Vec2f(x, y);

    // Calculate vertex positions (screen pixels, origin top-left)
    // We'll make 4 vertices for a quad
    const untransformedTopLeft = pos;
    const untransformedBottomLeft = new Vec2f(pos.x(), pos.y() + height);
    const untransformedTopRight = new Vec2f(pos.x() + width, pos.y());
    const untransformedBottomRight = new Vec2f(pos.x() + width, pos.y() + height);

    // Apply current transformation to each vertex
    const topLeft = this.currentTransform.transform(untransformedTopLeft);
    const bottomLeft = this.currentTransform.transform(untransformedBottomLeft);
    const topRight = this.currentTransform.transform(untransformedTopRight);
    const bottomRight = this.currentTransform.transform(untransformedBottomRight);

    // Send pre-transformed vertices to the mesh
    this.currentMesh!.drawRectWithTransform(
      topLeft, bottomLeft, topRight, bottomRight,
      u0, v0, u1, v1, color
    );
  }

  hasImage(imageName: string): boolean {
    return this.atlasData?.[imageName] !== undefined;
  }

  // Draws an image from the atlas with its top-right corner at (x, y).
  drawImage(imageName: string, x: number, y: number, color: number[] = [1, 1, 1, 1]): void {
    if (!this.ready) {
      throw new Error("Drawer not initialized");
    }

    this.ensureMeshSelected();

    if (!this.atlasData?.[imageName]) {
      throw new Error(`Image "${imageName}" not found in atlas`);
    }

    const [sx, sy, sw, sh] = this.atlasData[imageName];
    const m = this.atlasMargin;

    // Calculate UV coordinates (normalized 0.0 to 1.0).
    // Add the margin to allow texture filtering to handle edge anti-aliasing.
    const u0 = (sx - m) / this.textureSize.x();
    const v0 = (sy - m) / this.textureSize.y();
    const u1 = (sx + sw + m) / this.textureSize.x();
    const v1 = (sy + sh + m) / this.textureSize.y();

    // Draw the rectangle with the image's texture coordinates.
    // Adjust both UVs and vertex positions by the margin.
    this.drawRect(
      x - m, // Adjust x position by adding margin (from the right).
      y - m,      // Adjust y position by adding margin.
      sw + 2 * m,   // Reduce width by twice the margin (left and right).
      sh + 2 * m,   // Reduce height by twice the margin (top and bottom).
      u0, v0, u1, v1, color
    );
  }

  // Draws an image from the atlas centered at (x, y).
  drawSprite(imageName: string, x: number, y: number, color: number[] = [1, 1, 1, 1], scale = 1, rotation = 0): void {
    if (!this.ready) {
      throw new Error("Drawer not initialized");
    }

    this.ensureMeshSelected();

    if (!this.atlasData?.[imageName]) {
      throw new Error(`Image "${imageName}" not found in atlas`);
    }

    const [sx, sy, sw, sh] = this.atlasData[imageName]; // Source x, y, width, height from atlas.
    const m = this.atlasMargin;

    // Calculate UV coordinates (normalized 0.0 to 1.0).
    // Add the margin to allow texture filtering to handle edge anti-aliasing.
    const u0 = (sx - m) / this.textureSize.x();
    const v0 = (sy - m) / this.textureSize.y();
    const u1 = (sx + sw + m) / this.textureSize.x();
    const v1 = (sy + sh + m) / this.textureSize.y();

    if (scale != 1 || rotation != 0) {
      this.save();
      this.translate(x, y);
      this.rotate(rotation);
      this.scale(scale, scale);
      this.drawRect(
        - sw / 2 - m, // Center horizontally with margin adjustment.
        - sh / 2 - m, // Center vertically with margin adjustment.
        sw + 2 * m,         // Reduce width by twice the margin.
        sh + 2 * m,         // Reduce height by twice the margin.
        u0, v0, u1, v1, color
      );
      this.restore();
    } else {
      // Draw the rectangle with the image's texture coordinates.
      // For centered drawing, we need to account for the reduced size.
        this.drawRect(
        x - sw / 2 - m, // Center horizontally with margin adjustment.
        y - sh / 2 - m, // Center vertically with margin adjustment.
        sw + 2 * m,         // Reduce width by twice the margin.
        sh + 2 * m,         // Reduce height by twice the margin.
        u0, v0, u1, v1, color
      );
    }
  }

  drawSolidRect(
    x: number,
    y: number,
    width: number,
    height: number,
    color: number[]
  ) {
    if (!this.ready) {
      throw new Error("Drawer not initialized");
    }

    this.ensureMeshSelected();

    const imageName = "white.png";
    if (!this.atlasData?.[imageName]) {
      throw new Error(`Image "${imageName}" not found in atlas`);
    }

    // Get the middle of the white texture.
    const [sx, sy, sw, sh] = this.atlasData[imageName];
    const uvx = (sx + sw / 2) / this.textureSize.x();
    const uvy = (sy + sh / 2) / this.textureSize.y();
    this.drawRect(
      x,
      y,
      width,
      height,
      uvx,
      uvy,
      uvx,
      uvy,
      color
    )
  }

  // Flushes all non-empty meshes to the screen
  flush(): void {
    if (!this.ready || !this.device) {
      return;
    }

    // If no meshes have been created, nothing to do
    if (this.meshes.size === 0) {
      return;
    }

    // Setup for rendering
    const device = this.device;
    this.canvasSize = new Vec2f(this.canvas.width, this.canvas.height);
    device.queue.writeBuffer(
      this.canvasSizeUniformBuffer!,
      0, // Buffer offset.
      this.canvasSize.data // Use Vec2f data directly.
    );

    // Prepare command encoder
    const commandEncoder = device.createCommandEncoder({ label: 'Frame Command Encoder' });

    // Acquire the canvas texture view for the render pass
    if (this.renderPassDescriptor && this.context) {
      const descriptor = this.renderPassDescriptor as {
        colorAttachments: GPURenderPassColorAttachment[];
      };

      descriptor.colorAttachments[0].view = this.context.getCurrentTexture().createView();

      const passEncoder = commandEncoder.beginRenderPass(this.renderPassDescriptor);
      passEncoder.setPipeline(this.pipeline!);
      passEncoder.setBindGroup(0, this.bindGroup!);

      // Draw each mesh that has quads
      for (const mesh of this.meshes.values()) {
        const quadCount = mesh.getQuadCount();
        if (quadCount === 0) continue;

        const vertexBuffer = mesh.getVertexBuffer();
        const indexBuffer = mesh.getIndexBuffer();

        if (!vertexBuffer || !indexBuffer) continue;

        // Calculate data sizes
        const vertexDataCount = mesh.getCurrentVertexCount() * 8; // 8 floats per vertex
        const indexDataCount = quadCount * 6; // 6 indices per quad

        // Write vertex data to the GPU
        device.queue.writeBuffer(
          vertexBuffer,
          0,
          mesh.getVertexData(),
          0,
          vertexDataCount
        );

        // Set buffers
        passEncoder.setVertexBuffer(0, vertexBuffer);
        passEncoder.setIndexBuffer(indexBuffer, 'uint32');

        // Apply scissor if enabled for this mesh
        if (mesh.scissorEnabled) {
          const [x, y, width, height] = mesh.scissorRect;
          passEncoder.setScissorRect(x, y, width, height);
        } else {
          // Reset scissor to full canvas if previously set
          passEncoder.setScissorRect(0, 0, this.canvas.width, this.canvas.height);
        }

        // Draw the mesh
        passEncoder.drawIndexed(indexDataCount);
      }

      passEncoder.end();
    }

    const commandBuffer = commandEncoder.finish();
    device.queue.submit([commandBuffer]);

    // Reset all mesh counters after rendering
    for (const mesh of this.meshes.values()) {
      mesh.resetCounters();
    }
  }

  // Alias for flush (for backward compatibility)
  flushMesh(): void {
    this.flush();
  }

  // Helper method to generate mipmaps for a texture.
  generateMipmaps(texture: GPUTexture, width: number, height: number): void {
    // Don't try to generate mipmaps if the device doesn't support it.
    if (!this.device || !texture) return;

    // Create a render pipeline for mipmap generation.
    const mipmapShaderModule = this.device.createShaderModule({
      label: 'Mipmap Shader',
      code: `
        struct VertexOutput {
          @builtin(position) position: vec4f,
          @location(0) texCoord: vec2f,
        };

        @vertex
        fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
          var pos = array<vec2f, 4>(
            vec2f(-1.0, -1.0),
            vec2f(1.0, -1.0),
            vec2f(-1.0, 1.0),
            vec2f(1.0, 1.0)
          );

          var texCoord = array<vec2f, 4>(
            vec2f(0.0, 1.0),
            vec2f(1.0, 1.0),
            vec2f(0.0, 0.0),
            vec2f(1.0, 0.0)
          );

          var output: VertexOutput;
          output.position = vec4f(pos[vertexIndex], 0.0, 1.0);
          output.texCoord = texCoord[vertexIndex];
          return output;
        }

        @group(0) @binding(0) var imgSampler: sampler;
        @group(0) @binding(1) var imgTexture: texture_2d<f32>;

        @fragment
        fn fragmentMain(@location(0) texCoord: vec2f) -> @location(0) vec4f {
          return textureSample(imgTexture, imgSampler, texCoord);
        }
      `
    });

    const mipmapPipeline = this.device.createRenderPipeline({
      label: 'Mipmap Pipeline',
      layout: 'auto',
      vertex: {
        module: mipmapShaderModule,
        entryPoint: 'vertexMain',
      },
      fragment: {
        module: mipmapShaderModule,
        entryPoint: 'fragmentMain',
        targets: [{ format: 'rgba8unorm' }],
      },
      primitive: {
        topology: 'triangle-strip',
        stripIndexFormat: 'uint32',
      },
    });

    // Create a temporary sampler for mipmap generation.
    const mipmapSampler = this.device.createSampler({
      minFilter: 'linear',
      magFilter: 'linear',
    });

    // Calculate number of mip levels.
    const mipLevelCount = Math.floor(Math.log2(Math.max(width, height))) + 1;

    // Generate each mip level.
    const commandEncoder = this.device.createCommandEncoder({
      label: 'Mipmap Command Encoder',
    });

    // Create bind groups and render passes for each mip level.
    for (let i = 1; i < mipLevelCount; i++) {
      const srcView = texture.createView({
        baseMipLevel: i - 1,
        mipLevelCount: 1,
      });

      const dstView = texture.createView({
        baseMipLevel: i,
        mipLevelCount: 1,
      });

      // Create bind group for this mip level.
      const bindGroup = this.device.createBindGroup({
        layout: mipmapPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: mipmapSampler },
          { binding: 1, resource: srcView },
        ],
      });

      // Render to the next mip level.
      const renderPassDescriptor: GPURenderPassDescriptor = {
        colorAttachments: [
          {
            view: dstView,
            loadOp: 'clear',
            storeOp: 'store',
            clearValue: [0, 0, 0, 0],
          },
        ],
      };

      const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
      passEncoder.setPipeline(mipmapPipeline);
      passEncoder.setBindGroup(0, bindGroup);
      passEncoder.draw(4);
      passEncoder.end();
    }

    this.device.queue.submit([commandEncoder.finish()]);
  }
}

export { Drawer };
