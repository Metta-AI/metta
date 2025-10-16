import { Cell, MettaGrid, MettaObject } from "../MettaGrid";
import {
  loadMettaTileSets,
  TILE_NAMES,
  WALL_E,
  WALL_N,
  WALL_NAMES,
  WALL_S,
  WALL_W,
} from "./mettaTileSets";
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

const STATIC_OBJECT_DRAWERS: Record<string, ObjectDrawer> = {
  empty: [],
  wall: [{ tile: "wall" }], // unused by Drawer but used by AsciiEditor
  ...Object.fromEntries(
    TILE_NAMES.map((tile) => [tile, [{ tile }] as ObjectDrawer])
  ),
  "agent.agent": [{ tile: "agent" }],
  "agent.prey": [{ tile: "agent", modulate: { r: 0, g: 1, b: 0 } }],
  "agent.predator": [{ tile: "agent", modulate: { r: 1, g: 0, b: 0 } }],
};

const TEAM_OBJECT_NAMES = Array.from({ length: 10 }, (_, i) => `agent.team_${i}`);

export const objectNames = [...Object.keys(STATIC_OBJECT_DRAWERS), ...TEAM_OBJECT_NAMES];

function buildTeamDrawer(name: string): ObjectDrawer | undefined {
  if (!name.startsWith("agent.team_")) {
    return undefined;
  }

  const suffix = name.substring("agent.team_".length);
  const teamId = Number.parseInt(suffix, 10);
  if (Number.isNaN(teamId)) {
    return undefined;
  }

  const colorId = teamId === 0 ? 0 : teamId - 1;
  return [{ tile: "agent", modulate: colorFromId(colorId) }];
}

function getObjectDrawer(name: string): ObjectDrawer | undefined {
  const teamDrawer = buildTeamDrawer(name);
  if (teamDrawer) {
    return teamDrawer;
  }

  return STATIC_OBJECT_DRAWERS[name];
}

export const BACKGROUND_MAP_COLOR = "#cfa970";

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
    const layers = getObjectDrawer(object.name);
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
    const wallsGrid: boolean[][] = Array.from({ length: grid.width }, () =>
      Array.from({ length: grid.height }, () => false)
    );
    for (const wall of walls) {
      wallsGrid[wall.c][wall.r] = true;
    }

    const wallFills: Cell[] = [];

    const hasWall = (x: number, y: number) =>
      x >= 0 && x < grid.width && y >= 0 && y < grid.height && wallsGrid[x][y];

    for (let x = 0; x < grid.width; x++) {
      for (let y = 0; y < grid.height; y++) {
        if (!wallsGrid[x][y]) {
          continue;
        }
        let tile = 0;
        if (hasWall(x, y + 1)) tile = tile | WALL_S;
        if (hasWall(x + 1, y)) tile = tile | WALL_E;
        if (hasWall(x, y - 1)) tile = tile | WALL_N;
        if (hasWall(x - 1, y)) tile = tile | WALL_W;
        if (
          (tile & (WALL_S | WALL_E)) === (WALL_S | WALL_E) &&
          hasWall(x + 1, y + 1)
        ) {
          wallFills.push({ c: x, r: y });
          if (
            (tile & (WALL_N | WALL_W)) === (WALL_N | WALL_W) &&
            hasWall(x - 1, y - 1) &&
            hasWall(x - 1, y + 1) &&
            hasWall(x + 1, y - 1)
          ) {
            continue;
          }
        }
        this.drawTile({
          ctx,
          tile: WALL_NAMES[tile],
          c: x,
          r: y,
          scale: 256 / 200,
        });
      }
    }

    for (const fill of wallFills) {
      this.drawTile({
        ctx,
        tile: "wall.fill",
        c: fill.c + 0.5,
        r: fill.r + 0.3,
        scale: 256 / 200,
      });
    }
  }

  drawGrid(ctx: CanvasRenderingContext2D, grid: MettaGrid) {
    // Only draw the visible region of the grid - helps performance on big maps when zoomed in
    const { minX, minY, maxX, maxY } = visibleRegion(ctx, grid);

    // Clear drawing area
    ctx.fillStyle = BACKGROUND_MAP_COLOR;
    ctx.fillRect(minX, minY, maxX - minX, maxY - minY);

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
