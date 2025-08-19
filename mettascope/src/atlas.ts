import { Vec2f } from './vector_math.js'

/**
 * Type definition for atlas data loaded from JSON.
 *
 * @example
 * {
 *   "images": {
 *     "player.png": [0, 0, 32, 32],
 *     "enemy.png": [32, 0, 32, 32],
 *     "white.png": [64, 0, 1, 1]
 *   },
 *   "fonts": {
 *     "plexSans": {
 *       "ascent": 51.2,
 *       "descent": -16.0,
 *       "lineHeight": 80.0,
 *       "glyphs": {
 *         "U+0041": {
 *           "rect": [64, 0, 20, 24],
 *           "advance": 22.5,
 *           "bearingX": 1.2,
 *           "bearingY": -2.0
 *         },
 *         "U+0020": {
 *           "rect": null,
 *           "advance": 16.0,
 *           "bearingX": 0,
 *           "bearingY": 0
 *         }
 *       },
 *       "kerning": {
 *         "U+0041": {
 *           "U+0056": -2.5
 *         }
 *       },
 *       "fontName": "plexSans",
 *       "fontPath": "data/fonts/IBMPlexSans-Regular.ttf",
 *       "fontSize": 64,
 *       "fontCharset": " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~",
 *       "glyphInnerPadding": 2,
 *       "fontPathMtime": 1234567890,
 *       "fontPathSize": 98765,
 *       "fontConfigHash": "abc123"
 *     }
 *   },
 *   "metadata": {
 *     "version": "1.0.0",
 *     "generator": "atlas-builder"
 *   },
 *   "buildHash": "abc123..."
 * }
 */

export type SpriteBounds = [x: number, y: number, width: number, height: number]

function isSpriteBounds(val: unknown): val is SpriteBounds {
  return Array.isArray(val) && val.length === 4 && val.every((n) => typeof n === 'number')
}

export interface Glyph {
  rect: SpriteBounds | null // coordinates in the atlas or null if glyph has no visible pixels
  advance: number // Horizontal distance to advance the cursor after drawing this glyph (before kerning)
  bearingX: number // Horizontal offset from cursor position to the glyph's bounding box left edge
  bearingY: number // Vertical offset from baseline to the glyph's bounding box top edge
}

export interface FontKerningRow {
  [rightLabel: string]: number
}

export interface AtlasFont {
  ascent: number // distance from baseline to highest point of font
  descent: number // distance from baseline to lowest point of font (typically negative)
  lineHeight: number
  glyphs: { [label: string]: Glyph } // map from labels (e.g., "U+0041" for 'A') to glyph data
  kerning: { [leftLabel: string]: FontKerningRow }
  fontName: string
  fontPath: string
  fontSize: number
  fontCharset: string
  glyphInnerPadding: number
  fontPathMtime: number | null
  fontPathSize: number | null
  fontConfigHash: string
}

export interface AtlasSpriteMap {
  [key: string]: SpriteBounds
}

export interface AtlasMetadata {
  [key: string]: unknown
}

export interface AtlasData {
  images: AtlasSpriteMap
  fonts: { [fontName: string]: AtlasFont }
  metadata?: AtlasMetadata
  buildHash: string
}

/* Complete atlas bundle containing fonts, images, and the texture data. */
export interface Atlas extends AtlasData {
  texture: WebGLTexture // The WebGL texture containing all sprites
  size: Vec2f // Dimensions of the texture in pixels
  margin: number // Pixel margin added around sprites to prevent texture bleeding
}

/* Validates that an atlas object has the correct structure and types. */
export function validateAtlas(atlas: Atlas): boolean {
  return (
    typeof atlas.margin === 'number' &&
    atlas.size instanceof Vec2f &&
    atlas.texture instanceof WebGLTexture &&
    typeof atlas.buildHash === 'string' &&
    typeof atlas.images === 'object' &&
    Object.values(atlas.images).every(isSpriteBounds) &&
    typeof atlas.fonts === 'object' &&
    Object.values(atlas.fonts).every(validateFont)
  )
}

/* Validates a font object structure. */
function validateFont(font: unknown): font is AtlasFont {
  if (typeof font !== 'object' || font === null) return false
  const f = font as AtlasFont
  return (
    typeof f.ascent === 'number' &&
    typeof f.descent === 'number' &&
    typeof f.lineHeight === 'number' &&
    typeof f.fontName === 'string' &&
    typeof f.fontPath === 'string' &&
    typeof f.fontSize === 'number' &&
    typeof f.fontCharset === 'string' &&
    typeof f.glyphInnerPadding === 'number' &&
    typeof f.fontConfigHash === 'string' &&
    typeof f.glyphs === 'object' &&
    typeof f.kerning === 'object'
  )
}

/* Loads and parses atlas JSON data from a URL, returning the complete atlas data. */
export async function loadAtlasJson(url: string): Promise<AtlasData | null> {
  try {
    const res = await fetch(url)
    if (!res.ok) {
      throw new Error(`Failed to fetch atlas: ${res.statusText}`)
    }

    const raw = await res.json()

    // Validate required fields
    if (!raw.images || typeof raw.images !== 'object') {
      throw new Error('Atlas JSON missing required "images" field')
    }
    if (!raw.buildHash || typeof raw.buildHash !== 'string') {
      throw new Error('Atlas JSON missing required "buildHash" field')
    }

    const atlasData: AtlasData = {
      images: raw.images as AtlasSpriteMap,
      fonts: raw.fonts || {},
      metadata: raw.metadata,
      buildHash: raw.buildHash,
    }

    return atlasData
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

/* Loads a complete texture atlas from JSON and image URLs. Texture settings are selected for pixel art. */
export async function loadAtlas(
  gl: WebGLRenderingContext,
  jsonUrl: string,
  imageUrl: string,
  margin: number = 4
): Promise<Atlas | null> {
  const [atlasData, image] = await Promise.all([loadAtlasJson(jsonUrl), loadAtlasImage(imageUrl)])

  if (!atlasData || !image) {
    return null
  }

  const texture = gl.createTexture()
  if (!texture) {
    throw new Error('Failed to create WebGL texture')
  }

  gl.bindTexture(gl.TEXTURE_2D, texture)
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image)

  // gl.generateMipmap(gl.TEXTURE_2D) -- not needed for gl.NEAREST
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST)
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST)

  return {
    ...atlasData, // Spread all AtlasData properties
    texture,
    size: new Vec2f(image.width, image.height),
    margin,
  }
}

/* Retrieves sprite bounds and UV coordinates for a named sprite in the atlas. */
export function getSpriteBounds(
  atlas: Atlas,
  spriteName: string
): {
  x: number // X position with margin applied
  y: number // Y position with margin applied
  width: number // Width excluding margin
  height: number // Height excluding margin
  u0: number // Left texture coordinate (0-1)
  v0: number // Top texture coordinate (0-1)
  u1: number // Right texture coordinate (0-1)
  v1: number // Bottom texture coordinate (0-1)
} | null {
  const spriteData = atlas.images[spriteName]

  if (!isSpriteBounds(spriteData)) {
    return null
  }

  const [x, y, width, height] = spriteData
  const m = atlas.margin

  // Return the actual sprite bounds without including margin in width/height
  return {
    x: x + m,
    y: y + m,
    width: width,
    height: height,
    u0: (x + m) / atlas.size.x(),
    v0: (y + m) / atlas.size.y(),
    u1: (x + width - m) / atlas.size.x(),
    v1: (y + height - m) / atlas.size.y(),
  }
}

/* Checks whether a sprite with the given name exists in the atlas. */
export function hasSprite(atlas: Atlas, spriteName: string): boolean {
  return isSpriteBounds(atlas.images[spriteName])
}

/* Gets the UV coordinates for a solid white color pixel. */
export function getWhiteUV(atlas: Atlas): { u: number; v: number } | null {
  const whiteSprite = atlas.images['white.png']

  if (!isSpriteBounds(whiteSprite)) {
    console.warn('White pixel sprite "white.png" not found in atlas')
    return null
  }

  const [x, y, width, height] = whiteSprite
  return {
    u: (x + width / 2) / atlas.size.x(),
    v: (y + height / 2) / atlas.size.y(),
  }
}

/* Gets font data by name from the atlas. */
export function getFont(atlas: Atlas, fontName: string): AtlasFont | null {
  return atlas.fonts[fontName] || null
}

/* Gets glyph data for a specific character in a font. */
export function getGlyph(atlas: Atlas, fontName: string, char: string): Glyph | null {
  const font = getFont(atlas, fontName)
  if (!font) return null

  // Convert character to Unicode label format
  const codePoint = char.codePointAt(0)
  if (codePoint === undefined) return null

  const label = `U+${codePoint.toString(16).toUpperCase().padStart(4, '0')}`
  return font.glyphs[label] || null
}

/* Gets kerning value between two characters. */
export function getKerning(atlas: Atlas, fontName: string, leftChar: string, rightChar: string): number {
  const font = getFont(atlas, fontName)
  if (!font) return 0

  const leftCode = leftChar.codePointAt(0)
  const rightCode = rightChar.codePointAt(0)
  if (leftCode === undefined || rightCode === undefined) return 0

  const leftLabel = `U+${leftCode.toString(16).toUpperCase().padStart(4, '0')}`
  const rightLabel = `U+${rightCode.toString(16).toUpperCase().padStart(4, '0')}`

  const kerningRow = font.kerning[leftLabel]
  if (!kerningRow) return 0

  return kerningRow[rightLabel] || 0
}
