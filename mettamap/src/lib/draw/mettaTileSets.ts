import { TileSet, TileSetCollection, TileSetSource } from "./TileSet";

export const objectsTileSet: TileSetSource = {
  src: "/assets/legacy/items.png",
  tileSize: 16,
  tiles: {
    converter: [0, 0],
    mine: [14, 2],
    generator: [2, 2],
    altar: [12, 2],
    armory: [6, 3],
    lasery: [5, 5],
    lab: [5, 1],
    factory: [13, 0],
    temple: [7, 2],
  },
};

const monstersTileSet: TileSetSource = {
  src: "/assets/legacy/monsters.png",
  tileSize: 16,
  tiles: {
    agent: [2, 0],
  },
};

const wallTileSet: TileSetSource = {
  src: "/assets/legacy/wall.png",
  tileSize: 16,
  tiles: {
    wall: [0, 0],
  },
};

// singleton for loading tile sets
let tileSetsPromise: Promise<TileSetCollection> | null = null;
export async function loadMettaTileSets(): Promise<TileSetCollection> {
  if (!tileSetsPromise) {
    tileSetsPromise = Promise.all([
      TileSet.load(objectsTileSet),
      TileSet.load(monstersTileSet),
      TileSet.load(wallTileSet),
    ]).then((tileSets) => new TileSetCollection(tileSets));
  }
  return tileSetsPromise;
}
