import { TileSet, TileSetSource } from "./TileSet";
import { TileSetCollection } from "./TileSetCollection";

const sources = [
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
  "oxygen_ex_dep",
  "carbon_ex_dep",
  "germanium_ex_dep",
  "silicon_ex_dep",
  "lab",
  "lasery",
  "temple",
  "wall",
  "block",
  "agent",
].map(
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
