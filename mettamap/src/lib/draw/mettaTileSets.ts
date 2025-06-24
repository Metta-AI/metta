import { TileSet, TileSetCollection, TileSetSource } from "./TileSet";

const sources = [
  "agent.agent",
  "altar",
  "armory",
  "factory",
  "generator",
  "lab",
  "lasery",
  "mine",
  "temple",
  "wall",
  "block",
].map(
  (name) =>
    ({
      src: `/assets/objects/${name === "agent.agent" ? "agent" : name}.png`,
      tileSize: 256,
      tiles: {
        [name]: [0, 0],
      },
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
