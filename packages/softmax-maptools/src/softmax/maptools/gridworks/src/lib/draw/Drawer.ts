import { loadMettaTileSets } from "./mettaTileSets";
import { TileSetCollection } from "./TileSetCollection";

// based on mettascope's colorFromId
const colorFromId = (agentId: number) => {
  const n = agentId + Math.PI + Math.E + Math.SQRT2;
  return {
    r: (n * Math.PI) % 1.0,
    g: (n * Math.E) % 1.0,
    b: (n * Math.SQRT2) % 1.0,
  };
};

type ObjectLayer = {
  tile: string;
  modulate?: { r: number; g: number; b: number };
};

type ObjectDrawer = ObjectLayer[];

const objectDrawers: Record<string, ObjectDrawer> = {
  empty: [],
  wall: [{ tile: "wall" }],
  block: [{ tile: "block" }],
  altar: [{ tile: "altar" }],
  armory: [{ tile: "armory" }],
  factory: [{ tile: "factory" }],
  lab: [{ tile: "lab" }],
  lasery: [{ tile: "lasery" }],
  temple: [{ tile: "temple" }],
  mine_red: [
    { tile: "mine" },
    { tile: "mine.color", modulate: { r: 1, g: 0, b: 0 } },
  ],
  mine_blue: [
    { tile: "mine" },
    { tile: "mine.color", modulate: { r: 0, g: 0, b: 1 } },
  ],
  mine_green: [
    { tile: "mine" },
    { tile: "mine.color", modulate: { r: 0, g: 1, b: 0 } },
  ],
  generator_red: [
    { tile: "generator" },
    { tile: "generator.color", modulate: { r: 1, g: 0, b: 0 } },
  ],
  generator_blue: [
    { tile: "generator" },
    { tile: "generator.color", modulate: { r: 0, g: 0, b: 1 } },
  ],
  generator_green: [
    { tile: "generator" },
    { tile: "generator.color", modulate: { r: 0, g: 1, b: 0 } },
  ],
  "agent.agent": [{ tile: "agent" }],
  "agent.team_1": [{ tile: "agent", modulate: colorFromId(0) }],
  "agent.team_2": [{ tile: "agent", modulate: colorFromId(1) }],
  "agent.team_3": [{ tile: "agent", modulate: colorFromId(2) }],
  "agent.team_4": [{ tile: "agent", modulate: colorFromId(3) }],
  "agent.prey": [{ tile: "agent", modulate: { r: 0, g: 1, b: 0 } }],
  "agent.predator": [{ tile: "agent", modulate: { r: 1, g: 0, b: 0 } }],
};

export const objectNames = Object.keys(objectDrawers);

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
    const layers = objectDrawers[name];
    if (!layers) {
      throw new Error(`No drawer for object ${name}`);
    }
    for (const layer of layers) {
      const bitmap = this.tileSets.bitmap(layer.tile, layer.modulate);
      ctx.drawImage(bitmap, x, y, size, size);
    }
  }
}
