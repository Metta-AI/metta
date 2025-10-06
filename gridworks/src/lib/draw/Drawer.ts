import { MettaGrid, MettaObject } from "../MettaGrid";
import { loadMettaTileSets, TILE_NAMES } from "./mettaTileSets";
import { TileSetCollection } from "./TileSetCollection";

// based on mettascope's colorFromId
function colorFromId(agentId: number) {
  const n = agentId + Math.PI + Math.E + Math.SQRT2;
  return {
    r: (n * Math.PI) % 1.0,
    g: (n * Math.E) % 1.0,
    b: (n * Math.SQRT2) % 1.0,
  };
}

type ObjectLayer = {
  tile: string;
  modulate?: { r: number; g: number; b: number };
};

type ObjectDrawer = ObjectLayer[];

const objectDrawers: Record<string, ObjectDrawer> = {
  empty: [],
  ...Object.fromEntries(
    TILE_NAMES.map((tile) => [tile, [{ tile }] as ObjectDrawer])
  ),
  "agent.agent": [{ tile: "agent" }],
  "agent.team_1": [{ tile: "agent", modulate: colorFromId(0) }],
  "agent.team_2": [{ tile: "agent", modulate: colorFromId(1) }],
  "agent.team_3": [{ tile: "agent", modulate: colorFromId(2) }],
  "agent.team_4": [{ tile: "agent", modulate: colorFromId(3) }],
  "agent.prey": [{ tile: "agent", modulate: { r: 0, g: 1, b: 0 } }],
  "agent.predator": [{ tile: "agent", modulate: { r: 1, g: 0, b: 0 } }],
};

export const objectNames = Object.keys(objectDrawers);

const BACKGROUND_COLOR = "#cfa970";

function visibleRegion(ctx: CanvasRenderingContext2D, grid: MettaGrid) {
  // Get the current transformation matrix
  const transform = ctx.getTransform();

  // Invert it to go from screen space to transformed space
  const inverse = transform.invertSelf();

  const canvas = ctx.canvas;

  const transformPoint = (x: number, y: number) => {
    return {
      x: x * inverse.a + y * inverse.c + inverse.e,
      y: x * inverse.b + y * inverse.d + inverse.f,
    };
  };

  const { x: minX, y: minY } = transformPoint(0, 0);
  const { x: maxX, y: maxY } = transformPoint(canvas.width, canvas.height);

  return {
    // -1 and +1 are just to be safe with off-by-one errors
    minX: Math.max(0, Math.floor(minX) - 1),
    minY: Math.max(0, Math.floor(minY) - 1),
    maxX: Math.min(grid.width, Math.ceil(maxX) + 1),
    maxY: Math.min(grid.height, Math.ceil(maxY) + 1),
  };
}

export class Drawer {
  private constructor(public readonly tileSets: TileSetCollection) {}

  static async load(): Promise<Drawer> {
    const tileSets = await loadMettaTileSets();
    return new Drawer(tileSets);
  }

  // Assumes that `ctx` is scaled such that the image is 1x1
  drawObject(ctx: CanvasRenderingContext2D, object: MettaObject) {
    const layers = objectDrawers[object.name];
    if (!layers) {
      throw new Error(`No drawer for object ${object.name}`);
    }
    for (const layer of layers) {
      const bitmap = this.tileSets.bitmap(layer.tile, layer.modulate);
      ctx.drawImage(bitmap, object.c, object.r, 1, 1);
    }
  }

  drawGrid(ctx: CanvasRenderingContext2D, grid: MettaGrid) {
    // Only draw the visible region of the grid - helps performance on big maps when zoomed in
    const { minX, minY, maxX, maxY } = visibleRegion(ctx, grid);

    // Clear drawing area
    ctx.fillStyle = BACKGROUND_COLOR;
    ctx.fillRect(minX, minY, maxX - minX, maxY - minY);

    // Draw the map
    for (const object of grid.objects) {
      if (
        object.c < minX ||
        object.c > maxX ||
        object.r < minY ||
        object.r > maxY
      ) {
        continue;
      }
      if (object.name === "wall") {
        // in mettascope, walls have many different types depending on the surrounding terrain.
        // until we support that, we just draw a single wall type, and highlight it with color.
        ctx.fillStyle = "#c0bcb8";
        ctx.fillRect(object.c, object.r, 1, 1);
      }
      this.drawObject(ctx, object);
    }

    // Draw grid lines
    {
      ctx.strokeStyle = "black";
      ctx.globalAlpha = 0.2;
      ctx.lineCap = "square";
      ctx.lineWidth = 0.02;

      // Draw vertical grid lines
      for (let x = minX; x <= maxX; x++) {
        ctx.beginPath();
        ctx.moveTo(x, minY);
        ctx.lineTo(x, maxY);
        ctx.stroke();
      }

      // Draw horizontal grid lines
      for (let y = minY; y <= maxY; y++) {
        ctx.beginPath();
        ctx.moveTo(minX, y);
        ctx.lineTo(maxX, y);
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
    }
  }
}
