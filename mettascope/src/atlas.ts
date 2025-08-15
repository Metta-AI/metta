import { Vec2f } from './vector_math.js'

/**
 * Type definition for atlas data loaded from JSON.
 *
 * @example
 * {
 *   "player.png": [0, 0, 32, 32],
 *   "enemy.png": [32, 0, 32, 32],
 *   "metadata": { "version": "1.0" }
 * }
 *
 * The optional metadata field can contain additional data like font information.
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

/* Complete atlas bundle containing both the texture and sprite data. */
export interface Atlas {
  data: AtlasSpriteMap // The atlas JSON data with sprite definitions
  metadata: AtlasMetadata // Atlas metadata
  texture: WebGLTexture // The WebGL texture containing all sprites
  size: Vec2f // Dimensions of the texture in pixels
  margin: number // Pixel margin added around sprites to prevent texture bleeding
}

/* Validates that an atlas object has the correct structure and types. */
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

/* Loads and parses atlas JSON data from a URL, returning sprite map and metadata. */
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

/* Loads an image from a URL and creates a premultiplied alpha ImageBitmap. */
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

type TextureWrapMode =
  | WebGLRenderingContext['REPEAT']
  | WebGLRenderingContext['CLAMP_TO_EDGE']
  | WebGLRenderingContext['MIRRORED_REPEAT']

type TextureFilterMode =
  | WebGLRenderingContext['NEAREST']
  | WebGLRenderingContext['LINEAR']
  | WebGLRenderingContext['NEAREST_MIPMAP_NEAREST']
  | WebGLRenderingContext['LINEAR_MIPMAP_NEAREST']
  | WebGLRenderingContext['NEAREST_MIPMAP_LINEAR']
  | WebGLRenderingContext['LINEAR_MIPMAP_LINEAR']

type TextureMinFilterMode = TextureFilterMode
type TextureMagFilterMode = WebGLRenderingContext['NEAREST'] | WebGLRenderingContext['LINEAR']

/* Checks if a minification filter requires mipmaps. */
function requiresMipmaps(gl: WebGLRenderingContext, minFilter: TextureMinFilterMode): boolean {
  return (
    minFilter === gl.NEAREST_MIPMAP_NEAREST ||
    minFilter === gl.LINEAR_MIPMAP_NEAREST ||
    minFilter === gl.NEAREST_MIPMAP_LINEAR ||
    minFilter === gl.LINEAR_MIPMAP_LINEAR
  )
}

/* Creates a WebGL texture from an image with specified wrap and filter modes. */
export function createTexture(
  gl: WebGLRenderingContext,
  image: ImageBitmap,
  wrapS: TextureWrapMode,
  wrapT: TextureWrapMode,
  minFilter: TextureMinFilterMode,
  magFilter: TextureMagFilterMode,
  generateMipmap: boolean
): WebGLTexture | null {
  const texture = gl.createTexture()
  if (!texture) {
    return null
  }

  gl.bindTexture(gl.TEXTURE_2D, texture)
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image)

  if (generateMipmap !== false) {
    gl.generateMipmap(gl.TEXTURE_2D)
  }

  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, wrapS)
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, wrapT)
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, minFilter)
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, magFilter)

  return texture
}

/* Loads a complete texture atlas from JSON and image URLs. */
export async function loadAtlas(
  gl: WebGLRenderingContext,
  jsonUrl: string,
  imageUrl: string,
  generateMipmap: boolean = false
): Promise<Atlas | null> {
  const [json, image] = await Promise.all([loadAtlasJson(jsonUrl), loadAtlasImage(imageUrl)])

  if (!json || !image) {
    return null
  }

  const [data, metadata] = json

  const texture = createTexture(
    gl,
    image,
    gl.REPEAT, // wrapS
    gl.REPEAT, // wrapT
    gl.LINEAR_MIPMAP_LINEAR, // minFilter
    gl.LINEAR, // magFilter
    generateMipmap
  )
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

/* Retrieves sprite bounds and UV coordinates for a named sprite in the atlas. */
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

/* Checks whether a sprite with the given name exists in the atlas. */
export function hasSprite(atlas: Atlas, spriteName: string): boolean {
  const data = atlas.data[spriteName]
  return Array.isArray(data) && data.length === 4
}

/* Gets the UV coordinates for a solid white color pixel. */
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
