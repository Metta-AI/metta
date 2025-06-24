import { MettaObject } from "@/lib/MettaGrid";

import { loadMettaTileSets } from "./mettaTileSets";
import { TileSetCollection } from "./TileSetCollection";

export class Drawer {
  private constructor(public readonly tileSets: TileSetCollection) {}

  static async load(): Promise<Drawer> {
    const tileSets = await loadMettaTileSets();
    return new Drawer(tileSets);
  }

  drawObject(
    object: MettaObject,
    ctx: CanvasRenderingContext2D,
    x: number,
    y: number,
    size: number
  ) {
    if (object.name === "empty") {
      return;
    }
    this.tileSets.draw(object.name, ctx, x, y, size);
  }
}
