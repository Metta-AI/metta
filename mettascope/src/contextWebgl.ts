
// TODO First step, stub everything out and get things to start up

/** ContextWebgl is a class that manages the WebGL context. */
export class ContextWebgl {
  public canvas: HTMLCanvasElement
  public gl: WebGLRenderingContext
  public ready: boolean = false
  public dpr: number = 1
  public atlasData: any = null

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas
    const gl = canvas.getContext('webgl')
    if (!gl) {
      throw new Error('Failed to get WebGL context')
    }
    this.gl = gl
    console.log('constructor done')
  }

  async init(atlasJsonUrl: string, atlasImageUrl: string): Promise<boolean> {
    console.log('init stub', atlasJsonUrl, atlasImageUrl)
    this.ready = true
    return true
  }

  clear(): void {
    console.log('clear stub')
  }

  useMesh(name: string): void {
    console.log('useMesh stub', name)
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
