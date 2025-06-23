import { ItemObjectName, MettaObject } from "@/lib/MettaGrid";

const objectNameToItemTile = {
  converter: [0, 0],
  mine: [14, 2],
  generator: [2, 2],
  altar: [12, 2],
  armory: [6, 3],
  lasery: [5, 5],
  lab: [5, 1],
  factory: [13, 0],
  temple: [7, 2],
} satisfies Record<ItemObjectName, [number, number]>;

export class Sprites {
  wall: HTMLImageElement;
  items: HTMLImageElement;
  monsters: HTMLImageElement;

  constructor(
    wall: HTMLImageElement,
    items: HTMLImageElement,
    monsters: HTMLImageElement
  ) {
    this.wall = wall;
    this.items = items;
    this.monsters = monsters;
  }

  draw(
    object: MettaObject,
    ctx: CanvasRenderingContext2D,
    x: number,
    y: number,
    size: number
  ) {
    switch (object.name) {
      case "empty":
        return;
      case "wall":
        ctx.drawImage(this.wall, x, y, size, size);
        break;
      case "agent": {
        // from raylib_renderer code:
        // orientation: 0 = Up, 1 = Down, 2 = Left, 3 = Right
        // sprites: 0 = Right, 1 = Up, 2 = Down, 3 = Left
        const tileX = [1, 2, 3, 0][
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          (object.other as any)["agent:orientation"] ?? 0
        ];
        const tileY = 0; // TODO: agent.group
        ctx.drawImage(
          this.monsters,
          tileX * 16,
          tileY * 16,
          16,
          16,
          x,
          y,
          size,
          size
        );
        break;
      }
      default: {
        const [tileX, tileY] = objectNameToItemTile[object.name];
        ctx.drawImage(
          this.items,
          tileX * 16,
          tileY * 16,
          16,
          16,
          x,
          y,
          size,
          size
        );
        break;
      }
    }
  }
}

export async function loadSprites(): Promise<Sprites> {
  const loadImage = (src: string) => {
    return new Promise<HTMLImageElement>((resolve) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.src = src;
      return img;
    });
  };
  const [wall, items, monsters] = await Promise.all([
    loadImage("/assets/wall.png"),
    loadImage("/assets/items.png"),
    loadImage("/assets/monsters.png"),
  ]);
  const sprites = new Sprites(wall, items, monsters);
  return sprites;
}
