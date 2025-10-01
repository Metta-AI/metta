import { gridObjectRegistry } from "@/lib/gridObjectRegistry";

import { loadMettaTileSets } from "./mettaTileSets";
import { TileSetCollection } from "./TileSetCollection";

export class Drawer {
  private constructor(public readonly tileSets: TileSetCollection) {}

  static async load(): Promise<Drawer> {
    const tileSets = await loadMettaTileSets();
    return new Drawer(tileSets);
  }

  drawObject(
    name: string,
    ctx: CanvasRenderingContext2D,
    x: number,
    y: number,
    size: number
  ) {
    const layers = gridObjectRegistry.getLayers(name);
    for (const layer of layers) {
      const bitmap = this.tileSets.bitmap(layer.tile, layer.modulate);
      ctx.drawImage(bitmap, x, y, size, size);
    }
  }
}
