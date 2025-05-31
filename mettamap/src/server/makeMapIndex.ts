import fs from "fs/promises";
import yaml from "js-yaml";

import { MAP_INDEX_FILE } from "@/server/constants";
import { getMaps } from "@/server/getMaps";
import { MapFile } from "@/server/types";

export function frontmatterToKeyValues(
  frontmatter: unknown
): Record<string, string> {
  const keyToValue: Record<string, string> = {};

  const processNode = (path: string, node: unknown) => {
    if (node === null) {
      return;
    } else if (Array.isArray(node)) {
      for (let i = 0; i < node.length; i++) {
        processNode(`${path}.${i}`, node[i]);
      }
    } else if (typeof node === "object") {
      for (const [key, value] of Object.entries(node)) {
        processNode(path ? `${path}.${key}` : key, value);
      }
    } else if (
      typeof node === "string" ||
      typeof node === "number" ||
      typeof node === "boolean"
    ) {
      keyToValue[path] = String(node);
    }
  };

  processNode("", frontmatter);

  return keyToValue;
}

class MapIndex {
  public index: Record<string, Record<string, string[]>>;

  constructor() {
    this.index = {};
  }

  process(map: MapFile) {
    const frontmatter = yaml.load(map.content.frontmatter);

    const keyToValue = frontmatterToKeyValues(frontmatter);
    for (const [key, value] of Object.entries(keyToValue)) {
      this.index[key] ??= {};
      this.index[key][value] ??= [];
      this.index[key][value].push(map.file);
    }
  }
}

export async function makeMapIndex() {
  const maps = await getMaps();

  const indexer = new MapIndex();

  for (const map of maps) {
    indexer.process(map);
  }

  // Save the index to a JSON file
  const indexFilePath = MAP_INDEX_FILE;
  await fs.writeFile(indexFilePath, JSON.stringify(indexer.index, null, 2));
  console.log(`Map index saved to ${indexFilePath}`);
}
