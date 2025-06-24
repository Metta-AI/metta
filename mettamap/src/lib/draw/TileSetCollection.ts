import { CSSProperties } from "react";

import { TileSet } from "./TileSet";

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

  css(
    name: string,
    size: number
  ): { wrapper: CSSProperties; inner: CSSProperties } {
    const tileSet = this.nameToTileSet[name];
    return tileSet.css(name, size);
  }
}
