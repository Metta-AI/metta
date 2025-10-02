import { TileSet, TileSetSource } from "./TileSet";
import { TileSetCollection } from "./TileSetCollection";

const sources = [
  "altar",
  "assembler",
  "armory",
  "carbon_extractor",
  "charger",
  "factory",
  "germanium_extractor",
  "generator",
  "generator.color",
  "mine",
  "mine.color",
  "oxygen_extractor",
  "lab",
  "lasery",
  "silicon_extractor",
  "temple",
  "wall",
  "block",
  "agent",
].map(
  (name) =>
    ({
      src: `/assets/objects/${name}.png`,
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
