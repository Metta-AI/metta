import { TileSet, TileSetSource } from "./TileSet";
import { TileSetCollection } from "./TileSetCollection";

export const TILE_NAMES = [
  "assembler",
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
  "carbon_extractor.clipped",
  "oxygen_extractor.clipped",
  "germanium_extractor.clipped",
  "silicon_extractor.clipped",
  "charger_ex_dep",
  "oxygen_ex_dep",
  "carbon_ex_dep",
  "germanium_ex_dep",
  "silicon_ex_dep",
  "chest",
  "chest_carbon",
  "chest_oxygen",
  "chest_germanium",
  "chest_silicon",
  "lab",
  "lasery",
  "temple",
  "block",
  "converter",
  "trash",
  // doesn't include "agent" or "wall.*", they're special
];

const objects = [...TILE_NAMES, "agent"].map((name) => {
  // TODO: Consolidate sizes or store them along with the name
  const size = name === "assembler" ? 192 : name === "agent" ? 256 : 64;

  return {
    src: `/mettascope-assets/objects/${name}.png`,
    tileSize: size,
    tiles: [
      {
        name,
        // right now we don't use atlases, but this might change again in the future
        x: 0,
        y: 0,
      },
    ],
  };
}) satisfies TileSetSource[];

const walls = [
  {
    src: `/mettascope-assets/wall_atlas.png`,
    tileSize: 64,
    tiles: [{ name: "wall", x: 0, y: 0 }],
  },
  {
    src: `/mettascope-assets/wall_atlas.png`,
    tileSize: 64,
    tiles: Array.from({ length: 7 * 8 }, (_, i) => {
      const x = i % 7;
      const y = Math.floor(i / 7);
      return {
        name: `wall.${i}`,
        x,
        y,
      };
    }),
  },
];

const sources = [...walls, ...objects];

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
