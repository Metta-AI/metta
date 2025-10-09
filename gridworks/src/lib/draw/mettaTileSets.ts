import { TileSet, TileSetSource } from "./TileSet";
import { TileSetCollection } from "./TileSetCollection";

export const TILE_NAMES = [
  "altar",
  "armory",
  "factory",
  "generator",
  "generator_red",
  "generator_blue",
  "generator_green",
  "mine",
  "mine_red",
  "mine_blue",
  "mine_green",
  "charger",
  "carbon_extractor",
  "oxygen_extractor",
  "germanium_extractor",
  "silicon_extractor",
  "clipped_carbon_extractor",
  "clipped_oxygen_extractor",
  "clipped_germanium_extractor",
  "clipped_silicon_extractor",
  "oxygen_ex_dep",
  "carbon_ex_dep",
  "germanium_ex_dep",
  "silicon_ex_dep",
  "assembler",
  "chest",
  "chest_carbon",
  "chest_oxygen",
  "chest_germanium",
  "chest_silicon",
  "lab",
  "lasery",
  "temple",
  "block",
  // doesn't include "agent" or "wall", they're special
];

export const WALL_NAMES = [
  "wall",
  "wall.e",
  "wall.s",
  "wall.se",
  "wall.w",
  "wall.we",
  "wall.ws",
  "wall.wse",
  "wall.n",
  "wall.ne",
  "wall.ns",
  "wall.nse",
  "wall.nw",
  "wall.nwe",
  "wall.nws",
  "wall.nwse",
];

export const WALL_E = 1;
export const WALL_S = 2;
export const WALL_W = 4;
export const WALL_N = 8;

const sources = [...TILE_NAMES, ...WALL_NAMES, "wall.fill", "agent"].map(
  (name) =>
    ({
      src: `/mettascope-assets/objects/${name}.png`,
      tileSize: 256,
      tiles: [
        {
          name,
          // right now we don't use atlases, but this might change again in the future
          x: 0,
          y: 0,
        },
      ],
    }) satisfies TileSetSource
);

// singleton for loading tile sets
let tileSetsPromise: Promise<TileSetCollection> | null = null;
export async function loadMettaTileSets(): Promise<TileSetCollection> {
  if (!tileSetsPromise) {
    tileSetsPromise = Promise.all(sources.map(TileSet.load)).then(
      (tileSets) => new TileSetCollection(tileSets)
    );
  }
  return tileSetsPromise;
}
