import { CSSProperties } from "react";

export type TileSetSource = {
  src: string;
  tileSize: number;
  tiles: Record<string, [number, number]>;
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
  tiles: Record<string, [number, number]>;

  private constructor(
    image: HTMLImageElement,
    tileSize: number,
    tiles: Record<string, [number, number]>
  ) {
    this.image = image;
    this.tileSize = tileSize;
    this.tiles = tiles;
  }

  static async load(tileSet: TileSetSource): Promise<TileSet> {
    const img = await loadImage(tileSet.src);
    return new TileSet(img, tileSet.tileSize, tileSet.tiles);
  }

  draw(
    name: string,
    ctx: CanvasRenderingContext2D,
    x: number,
    y: number,
    size: number
  ) {
    const [tileX, tileY] = this.tiles[name];
    ctx.drawImage(
      this.image,
      tileX * this.tileSize,
      tileY * this.tileSize,
      this.tileSize,
      this.tileSize,
      x,
      y,
      size,
      size
    );
  }
}

export class TileSetCollection {
  nameToTileSet: Record<string, TileSet> = {};

  constructor(public readonly tileSets: TileSet[]) {
    // validate that all tile sets have unique tile names
    for (const tileSet of tileSets) {
      for (const tileName of Object.keys(tileSet.tiles)) {
        if (this.nameToTileSet[tileName]) {
          throw new Error(`Tile name ${tileName} is not unique`);
        }
        this.nameToTileSet[tileName] = tileSet;
      }
    }
  }

  draw(
    name: string,
    ctx: CanvasRenderingContext2D,
    x: number,
    y: number,
    size: number
  ) {
    const tileSet = this.nameToTileSet[name];
    if (!tileSet) {
      throw new Error(`Tile set for ${name} not found`);
    }
    tileSet.draw(name, ctx, x, y, size);
  }

  css(name: string, size: number) {
    // Written by ChatGPT, maybe possible to simplify

    // find the tileset & tile coords
    const tileSet = this.nameToTileSet[name];
    const [col, row] = tileSet.tiles[name];

    const tileW = tileSet.tileSize;
    const tileH = tileSet.tileSize;

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
      backgroundImage: `url(${tileSet.image.src})`,
      backgroundRepeat: "no-repeat",
      // position the correct tile in the sheet:
      backgroundPosition: `-${col * tileW}px -${row * tileH}px`,
      // then grow/shrink it to fill the wrapper:
      transform: `scale(${scale})`,
      transformOrigin: "top left",
    };

    return { wrapper, inner };
  }
}
