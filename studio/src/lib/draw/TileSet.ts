type TileInfo = {
  name: string;
  x: number;
  y: number;
};

export type TileSetSource = {
  src: string;
  tileSize: number;
  tiles: TileInfo[];
};

type Tile = {
  info: TileInfo;
  image: ImageBitmap;
  modulateCache: Record<string, ImageBitmap>;
};

async function loadImage(src: string) {
  return new Promise<HTMLImageElement>((resolve) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.src = src;
    return img;
  });
}

export class TileSet {
  private constructor(
    public readonly image: HTMLImageElement,
    public readonly tileSize: number,
    public readonly tiles: Record<string, Tile>
  ) {}

  static async load(source: TileSetSource): Promise<TileSet> {
    const img = await loadImage(source.src);

    const tiles: Record<string, Tile> = {};
    for (const tileInfo of source.tiles) {
      const bitmap = await createImageBitmap(
        img,
        tileInfo.x * source.tileSize,
        tileInfo.y * source.tileSize,
        source.tileSize,
        source.tileSize
      );
      const tile: Tile = {
        info: tileInfo,
        image: bitmap,
        modulateCache: {},
      };
      tiles[tileInfo.name] = tile;
    }
    const tileSet = new TileSet(img, source.tileSize, tiles);
    return tileSet;
  }

  bitmap(name: string, modulate?: { r: number; g: number; b: number }) {
    const tile = this.tiles[name];
    if (!tile) {
      throw new Error(`Tile ${name} not found in tile set`);
    }

    if (modulate) {
      const cacheKey = `${modulate.r},${modulate.g},${modulate.b}`;
      if (!tile.modulateCache[cacheKey]) {
        const offscreen = new OffscreenCanvas(this.tileSize, this.tileSize);
        const ctx = offscreen.getContext("2d");
        if (!ctx) {
          throw new Error("Failed to get context");
        }

        ctx.drawImage(tile.image, 0, 0);

        const imageData = ctx.getImageData(0, 0, this.tileSize, this.tileSize);

        for (let i = 0; i < imageData.data.length; i += 4) {
          imageData.data[i + 0] *= modulate.r;
          imageData.data[i + 1] *= modulate.g;
          imageData.data[i + 2] *= modulate.b;
        }

        ctx.putImageData(imageData, 0, 0);
        tile.modulateCache[cacheKey] = offscreen.transferToImageBitmap();
      }
      return tile.modulateCache[cacheKey];
    }
    return tile.image;
  }
}
