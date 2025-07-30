import { Vec2f } from './vector_math.js'

/**
 * Type definition for atlas data loaded from JSON.
 *
 * Contains sprite definitions as [x, y, width, height] arrays, where:
 * - x, y: Top-left corner position in the atlas texture
 * - width, height: Dimensions of the sprite in pixels
 *
 * The optional metadata field can contain additional data like font information.
 *
 * @example
 * {
 *   "player.png": [0, 0, 32, 32],
 *   "enemy.png": [32, 0, 32, 32],
 *   "metadata": { "version": "1.0" }
 * }
 */
export type SpriteBounds = [x: number, y: number, width: number, height: number]

function isSpriteBounds(val: unknown): val is SpriteBounds {
  return Array.isArray(val) && val.length === 4 && val.every((n) => typeof n === 'number')
}

export interface AtlasSpriteMap {
  [key: string]: SpriteBounds
}

export interface AtlasMetadata {
  [key: string]: unknown
}

/**
 * Complete atlas bundle containing both the texture and sprite data.
 *
 * @property data - The atlas JSON data with sprite definitions
 * @property texture - The WebGL texture containing all sprites
 * @property size - Dimensions of the texture in pixels
 * @property margin - Pixel margin added around sprites to prevent texture bleeding
 */
export interface Atlas {
  data: AtlasSpriteMap
  metadata: AtlasMetadata
  texture: WebGLTexture
  size: Vec2f
  margin: number
}


export function validateAtlas(atlas: Atlas): boolean {
  return (
    typeof atlas.margin === 'number' &&
    atlas.size instanceof Vec2f &&
    atlas.texture !== null &&
    Object.values(atlas.data).every(
      (bounds) => Array.isArray(bounds) && bounds.length === 4 && bounds.every((n) => typeof n === 'number')
    )
  )
}


export async function loadAtlasJson(url: string): Promise<[AtlasSpriteMap, AtlasMetadata] | null> {
  try {
    const res = await fetch(url)
    if (!res.ok) {
      throw new Error(`Failed to fetch atlas: ${res.statusText}`)
    }

    const raw = await res.json()

    const metadata = (raw.metadata ?? {}) as AtlasMetadata

    const spriteEntries = Object.entries(raw).filter(([k, v]) => k !== 'metadata' && isSpriteBounds(v))

    const sprites = Object.fromEntries(spriteEntries) as AtlasSpriteMap

    return [sprites as AtlasSpriteMap, metadata as AtlasMetadata]
  } catch (err) {
    console.error(`Error loading atlas ${url}:`, err)
    return null
  }
}


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
  if (!texture) {
    return null
  }

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

export async function loadAtlas(
  gl: WebGLRenderingContext,
  jsonUrl: string,
  imageUrl: string,
  textureOptions?: Parameters<typeof createTexture>[2]
): Promise<Atlas | null> {
  const [json, image] = await Promise.all([loadAtlasJson(jsonUrl), loadAtlasImage(imageUrl)])

  if (!json || !image) {
    return null
  }

  const [data, metadata] = json

  const texture = createTexture(gl, image, textureOptions)
  if (!texture) {
    return null
  }

  return {
    data,
    metadata,
    texture,
    size: new Vec2f(image.width, image.height),
    margin: 4, // Default margin
  }
}


export function getSpriteBounds(
  atlas: Atlas,
  spriteName: string
): {
  x: number // X position with margin applied
  y: number // Y position with margin applied
  width: number // Width including margin on both sides
  height: number // Height including margin on both sides
  u0: number // Left texture coordinate (0-1)
  v0: number // Top texture coordinate (0-1)
  u1: number // Right texture coordinate (0-1)
  v1: number // Bottom texture coordinate (0-1)
} | null {
  const spriteData = atlas.data[spriteName]

  // Check if it's actually sprite data (array of 4 numbers)
  if (!Array.isArray(spriteData) || spriteData.length !== 4) {
    return null
  }

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
    v1: (y + height + m) / atlas.size.y(),
  }
}

/**
 * Check if a sprite exists in the atlas.
 *
 * Only returns true if the entry is a valid sprite definition
 * (array of 4 numbers), not for metadata entries.
 *
 * @param atlas - The atlas to check
 * @param spriteName - Name of the sprite to look for
 * @returns True if the sprite exists, false otherwise
 *
 * @example
 * if (hasSprite(atlas, 'player.png')) {
 *   drawSprite('player.png', x, y)
 * }
 */
export function hasSprite(atlas: Atlas, spriteName: string): boolean {
  const data = atlas.data[spriteName]
  return Array.isArray(data) && data.length === 4
}

/**
 * Get UV coordinates for a solid color pixel.
 *
 * This looks for 'white.png' in the atlas and returns the UV coordinates
 * of its center pixel. Used for drawing solid color rectangles with the
 * same shader/texture as sprites.
 *
 * @param atlas - The atlas containing white.png
 * @returns UV coordinates of the white pixel center, or null if white.png not found
 *
 * @example
 * const whiteUV = getWhiteUV(atlas)
 * if (whiteUV) {
 *   // Draw a solid red rectangle
 *   drawRect(x, y, width, height, whiteUV.u, whiteUV.v, whiteUV.u, whiteUV.v, [1, 0, 0, 1])
 * }
 */
export function getWhiteUV(atlas: Atlas): { u: number; v: number } | null {
  const whiteSprite = atlas.data['white.png']

  if (!Array.isArray(whiteSprite) || whiteSprite.length !== 4) {
    return null
  }

  const [x, y, width, height] = whiteSprite
  return {
    u: (x + width / 2) / atlas.size.x(),
    v: (y + height / 2) / atlas.size.y(),
  }
}
