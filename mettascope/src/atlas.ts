import { Vec2f } from './vector_math.js'

/** Type definition for atlas data. */
export interface AtlasData {
  [key: string]: [number, number, number, number], // [x, y, width, height]
  metadata?: any // optional extra json data
}

/** Atlas information bundle. */
export interface Atlas {
  data: AtlasData
  texture: WebGLTexture
  size: Vec2f
  margin: number
}

/** Load atlas JSON data. */
export async function loadAtlasJson(url: string): Promise<AtlasData | null> {
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

/** Load atlas image. */
export async function loadAtlasImage(url: string): Promise<ImageBitmap | null> {
  try {
    const res = await fetch(url)
    if (!res.ok) {
      throw new Error(`Failed to fetch image: ${res.statusText}`)
    }
    const blob = await res.blob()
    // Use premultiplied alpha to fix border issues
    return await createImageBitmap(blob, {
      colorSpaceConversion: 'none',
      premultiplyAlpha: 'premultiply',
    })
  } catch (err) {
    console.error(`Error loading image ${url}:`, err)
    return null
  }
}

/** Create a texture from an image. */
export function createTexture(
  gl: WebGLRenderingContext,
  image: ImageBitmap,
  options: {
    wrapS?: number
    wrapT?: number
    minFilter?: number
    magFilter?: number
    generateMipmap?: boolean
  } = {}
): WebGLTexture | null {
  const texture = gl.createTexture()
  if (!texture) return null

  gl.bindTexture(gl.TEXTURE_2D, texture)
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image)

  if (options.generateMipmap !== false) {
    gl.generateMipmap(gl.TEXTURE_2D)
  }

  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, options.wrapS ?? gl.REPEAT)
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, options.wrapT ?? gl.REPEAT)
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, options.minFilter ?? gl.LINEAR_MIPMAP_LINEAR)
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, options.magFilter ?? gl.LINEAR)

  return texture
}

/** Load a complete atlas (data + texture). */
export async function loadAtlas(
  gl: WebGLRenderingContext,
  jsonUrl: string,
  imageUrl: string,
  textureOptions?: Parameters<typeof createTexture>[2]
): Promise<Atlas | null> {
  const [data, image] = await Promise.all([
    loadAtlasJson(jsonUrl),
    loadAtlasImage(imageUrl)
  ])

  if (!data || !image) {
    return null
  }

  const texture = createTexture(gl, image, textureOptions)
  if (!texture) {
    return null
  }

  return {
    data,
    texture,
    size: new Vec2f(image.width, image.height),
    margin: 4 // Default margin
  }
}

/** Get sprite bounds from atlas data. */
export function getSpriteBounds(atlas: Atlas, spriteName: string): {
  x: number
  y: number
  width: number
  height: number
  u0: number
  v0: number
  u1: number
  v1: number
} | null {
  const spriteData = atlas.data[spriteName]
  if (!spriteData) return null

  const [x, y, width, height] = spriteData
  const m = atlas.margin

  return {
    x: x - m,
    y: y - m,
    width: width + 2 * m,
    height: height + 2 * m,
    u0: (x - m) / atlas.size.x(),
    v0: (y - m) / atlas.size.y(),
    u1: (x + width + m) / atlas.size.x(),
    v1: (y + height + m) / atlas.size.y()
  }
}

/** Check if a sprite exists in the atlas. */
export function hasSprite(atlas: Atlas, spriteName: string): boolean {
  return atlas.data[spriteName] !== undefined
}

/** Get the UV coordinates for a solid white color (using white.png). */
export function getWhiteUV(atlas: Atlas): { u: number, v: number } | null {
  const whiteSprite = atlas.data['white.png']
  if (!whiteSprite) return null

  const [x, y, width, height] = whiteSprite
  return {
    u: (x + width / 2) / atlas.size.x(),
    v: (y + height / 2) / atlas.size.y()
  }
}
