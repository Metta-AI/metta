import { CSSProperties } from "react";

type TileInfo = {
  name: string;
  x: number;
  y: number;
  modulate?: { r: number; g: number; b: number };
};

export type TileSetSource = {
  src: string;
  tileSize: number;
  tiles: TileInfo[];
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
  image: HTMLImageElement;
  tileSize: number;
  tiles: Record<string, TileInfo>;
  bitmaps: Record<string, ImageBitmap>;

  private constructor(
    image: HTMLImageElement,
    tileSize: number,
    tiles: TileInfo[]
  ) {
    this.image = image;
    this.tileSize = tileSize;
    this.tiles = Object.fromEntries(tiles.map((t) => [t.name, t]));
    this.bitmaps = {};
  }

  static async load(source: TileSetSource): Promise<TileSet> {
    const img = await loadImage(source.src);
    const tileSet = new TileSet(img, source.tileSize, source.tiles);
    for (const tile of source.tiles) {
      let bitmap = await createImageBitmap(
        img,
        tile.x * source.tileSize,
        tile.y * source.tileSize,
        source.tileSize,
        source.tileSize
      );
      if (tile.modulate) {
        const tileSize = source.tileSize;
        const offscreen = new OffscreenCanvas(tileSize, tileSize);
        const ctx = offscreen.getContext("2d");
        if (!ctx) {
          throw new Error("Failed to get context");
        }

        ctx.drawImage(bitmap, 0, 0);

        const imageData = ctx.getImageData(0, 0, tileSize, tileSize);

        for (let i = 0; i < imageData.data.length; i += 4) {
          imageData.data[i + 0] *= tile.modulate.r;
          imageData.data[i + 1] *= tile.modulate.g;
          imageData.data[i + 2] *= tile.modulate.b;
        }

        ctx.putImageData(imageData, 0, 0);
        bitmap = offscreen.transferToImageBitmap();
      }

      tileSet.bitmaps[tile.name] = bitmap;
    }
    return tileSet;
  }

  draw(
    name: string,
    ctx: CanvasRenderingContext2D,
    x: number,
    y: number,
    size: number
  ) {
    const tile = this.tiles[name];
    if (!tile) {
      throw new Error(`Tile ${name} not found in tile set`);
    }
    ctx.drawImage(this.bitmaps[name], x, y, size, size);
  }

  css(
    name: string,
    size: number
  ): { wrapper: CSSProperties; inner: CSSProperties } {
    // Written by ChatGPT, maybe possible to simplify
    // find the tileset & tile coords
    const tile = this.tiles[name];

    const tileW = this.tileSize;
    const tileH = this.tileSize;

    // how much to scale the natural tile to your desired size
    const scale = size / tileW;

    const wrapper: CSSProperties = {
      width: size,
      height: size,
      overflow: "hidden",
    };

    const inner: CSSProperties = {
      width: tileW,
      height: tileH,
      backgroundImage: `url(${this.image.src})`,
      backgroundRepeat: "no-repeat",
      // position the correct tile in the sheet:
      backgroundPosition: `-${tile.x * tileW}px -${tile.y * tileH}px`,
      // then grow/shrink it to fill the wrapper:
      transform: `scale(${scale})`,
      transformOrigin: "top left",
    };

    return { wrapper, inner };
  }
}
