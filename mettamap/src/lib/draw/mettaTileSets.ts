import { TileSet, TileSetSource } from "./TileSet";
import { TileSetCollection } from "./TileSetCollection";

const objectSources = [
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
      src: `/assets/objects/${name}.png`,
      tileSize: 256,
      tiles: [
        {
          name,
          x: 0,
          y: 0,
        },
      ],
    }) satisfies TileSetSource
);

// based on mettascope's colorFromId
const colorFromId = (agentId: number) => {
  let n = agentId + Math.PI + Math.E + Math.SQRT2;
  return {
    r: (n * Math.PI) % 1.0,
    g: (n * Math.E) % 1.0,
    b: (n * Math.SQRT2) % 1.0,
  };
};

const agentSource: TileSetSource = {
  src: "/assets/objects/agent.png",
  tileSize: 256,
  tiles: [
    {
      name: "agent.agent",
      x: 0,
      y: 0,
    },
    {
      name: "agent.team_1",
      x: 0,
      y: 0,
      modulate: colorFromId(1),
    },
    {
      name: "agent.team_2",
      x: 0,
      y: 0,
      modulate: colorFromId(2),
    },
  ],
};

const sources = [...objectSources, agentSource];

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
