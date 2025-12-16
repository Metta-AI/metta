import { Cell, MettaGrid, MettaObject } from "../MettaGrid";
import { loadMettaTileSets, TILE_NAMES } from "./mettaTileSets";
import { TileSetCollection } from "./TileSetCollection";

const wallPatternToTile = [
  18, 17, 4, 4, 12, 22, 4, 4, 30, 13, 41, 41, 30, 13, 41, 41, 19, 23, 5, 5, 37,
  9, 5, 5, 30, 13, 41, 41, 30, 13, 41, 41, 24, 43, 39, 39, 44, 45, 39, 39, 48,
  32, 46, 46, 48, 32, 46, 46, 24, 43, 39, 39, 44, 45, 39, 39, 48, 32, 46, 46,
  48, 32, 46, 46, 36, 10, 3, 3, 16, 40, 3, 3, 20, 27, 6, 6, 20, 27, 6, 6, 25,
  15, 2, 2, 26, 38, 2, 2, 20, 27, 6, 6, 20, 27, 6, 6, 24, 43, 39, 39, 44, 45,
  39, 39, 48, 32, 46, 46, 48, 32, 46, 46, 24, 43, 39, 39, 44, 45, 39, 39, 48,
  32, 46, 46, 48, 32, 46, 46, 28, 28, 8, 8, 21, 21, 8, 8, 33, 33, 7, 7, 33, 33,
  7, 7, 35, 35, 31, 31, 14, 14, 31, 31, 33, 33, 7, 7, 33, 33, 7, 7, 47, 47, 1,
  1, 42, 42, 1, 1, 34, 34, 0, 0, 34, 34, 0, 0, 47, 47, 1, 1, 42, 42, 1, 1, 34,
  34, 0, 0, 34, 34, 0, 0, 28, 28, 8, 8, 21, 21, 8, 8, 33, 33, 7, 7, 33, 33, 7,
  7, 35, 35, 31, 31, 14, 14, 31, 31, 33, 33, 7, 7, 33, 33, 7, 7, 47, 47, 1, 1,
  42, 42, 1, 1, 34, 34, 0, 0, 34, 34, 0, 0, 47, 47, 1, 1, 42, 42, 1, 1, 34, 34,
  0, 0, 34, 34, 0, 0,
];

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
  wall: [{ tile: "wall" }], // unused by Drawer but used by AsciiEditor
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

export const BACKGROUND_MAP_COLOR = "#000000";

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
  private constructor(
    public readonly tileSets: TileSetCollection,
    private cachedOffscreenFloorBitmap: ImageBitmap | null = null
  ) {}

  static async load(): Promise<Drawer> {
    const tileSets = await loadMettaTileSets();
    const drawer = new Drawer(tileSets);
    await drawer.generateOffscreenFloorBitmap();
    return drawer;
  }

  drawTile({
    ctx,
    tile,
    c,
    r,
    modulate,
    scale = 1,
  }: {
    ctx: CanvasRenderingContext2D;
    tile: string;
    c: number;
    r: number;
    modulate?: { r: number; g: number; b: number };
    scale?: number;
  }) {
    const bitmap = this.tileSets.bitmap(tile, modulate);
    ctx.drawImage(bitmap, c, r, scale, scale);
  }

  // Assumes that `ctx` is scaled such that the image is 1x1
  drawObject(ctx: CanvasRenderingContext2D, object: MettaObject) {
    const layers = objectDrawers[object.name];
    if (!layers) {
      throw new Error(`No drawer for object ${object.name}`);
    }
    for (const layer of layers) {
      this.drawTile({
        ctx,
        tile: layer.tile,
        c: object.c,
        r: object.r,
        modulate: layer.modulate,
      });
    }
  }

  drawWalls(ctx: CanvasRenderingContext2D, grid: MettaGrid, walls: Cell[]) {
    // Ported from worldmap.nim in mettascope
    const wallSet = new Set<number>();

    for (const wall of walls) {
      wallSet.add(wall.c + wall.r * grid.width);
    }

    const isFloor = (x: number, y: number) => {
      if (x < 0 || y < 0 || x >= grid.width || y >= grid.height) {
        return 0;
      }
      return wallSet.has(x + y * grid.width) ? 0 : 1;
    };

    for (let x = 0; x < grid.width; x++) {
      for (let y = 0; y < grid.height; y++) {
        if (isFloor(x, y)) {
          continue;
        }

        const pattern =
          1 * isFloor(x - 1, y - 1) + // NW
          2 * isFloor(x, y - 1) + // N
          4 * isFloor(x + 1, y - 1) + // NE
          8 * isFloor(x + 1, y) + // E
          16 * isFloor(x + 1, y + 1) + // SE
          32 * isFloor(x, y + 1) + // S
          64 * isFloor(x - 1, y + 1) + // SW
          128 * isFloor(x - 1, y); // W

        const tile = wallPatternToTile[pattern];

        // First draw a void layer to cover up the terrain
        ctx.fillStyle = BACKGROUND_MAP_COLOR;
        ctx.fillRect(x, y, 1, 1);

        // Then draw the wall tile
        this.drawTile({
          ctx,
          tile: `wall.${tile}`,
          c: x,
          r: y,
        });
      }
    }
  }

  private async generateOffscreenFloorBitmap() {
    if (this.cachedOffscreenFloorBitmap) {
      return this.cachedOffscreenFloorBitmap;
    }

    const bitmapSize = 10;
    const tileSize = 64;

    const getWeightedRandomInt = (weights: number[]): number => {
      const totalWeight = weights.reduce((a, b) => a + b, 0);
      let r = Math.random() * totalWeight;
      for (let i = 0; i < weights.length; i++) {
        r -= weights[i];
        if (r <= 0) {
          return i;
        }
      }
      return weights.length - 1; // Fallback
    };

    const canvas = document.createElement("canvas");
    canvas.width = bitmapSize * tileSize;
    canvas.height = bitmapSize * tileSize;
    const ctx = canvas.getContext("2d")!;

    for (let i = 0; i < bitmapSize * bitmapSize; i++) {
      const weights = [100, 50, 25, 10, 5, 2, 1];
      const tileIndex = getWeightedRandomInt(weights) + 49;
      const tileName = `wall.${tileIndex}`;
      const tileBitmap = this.tileSets.bitmap(tileName);
      const x = (i % bitmapSize) * tileSize;
      const y = Math.floor(i / bitmapSize) * tileSize;
      ctx.drawImage(tileBitmap, x, y, tileSize, tileSize);
    }

    const bitmap = await createImageBitmap(canvas);
    this.cachedOffscreenFloorBitmap = bitmap;
    return bitmap;
  }

  private drawFloor(ctx: CanvasRenderingContext2D, grid: MettaGrid) {
    const tile = this.cachedOffscreenFloorBitmap;

    if (tile === null) {
      // Invariant: generateOffscreenFloorBitmap must be called before drawFloor
      throw new Error("Offscreen floor bitmap not generated before drawFloor");
    }

    const step = 10;

    for (let x = 0; x < grid.width; x += step) {
      for (let y = 0; y < grid.height; y += step) {
        const w = Math.min(step, grid.width - x);
        const h = Math.min(step, grid.height - y);
        ctx.drawImage(tile, 0, 0, tile.width, tile.height, x, y, w, h);
      }
    }
  }

  drawGrid(ctx: CanvasRenderingContext2D, grid: MettaGrid) {
    // Preserve pixelated look when zoomed in
    ctx.imageSmoothingEnabled = false;

    // Only draw the visible region of the grid - helps performance on big maps when zoomed in
    const { minX, minY, maxX, maxY } = visibleRegion(ctx, grid);

    // Clear drawing area
    ctx.fillStyle = BACKGROUND_MAP_COLOR;
    ctx.fillRect(minX, minY, maxX - minX, maxY - minY);

    // Draw floor over entire visible region
    this.drawFloor(ctx, grid);

    // Sort objects into walls and other objects
    const objects: MettaObject[] = [];
    const walls: Cell[] = [];
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
        walls.push({ c: object.c, r: object.r });
      } else {
        objects.push(object);
      }
    }

    this.drawWalls(ctx, grid, walls);
    for (const object of objects) {
      // Draw the object itself
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
