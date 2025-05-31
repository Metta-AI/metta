import fs from "fs/promises";
import yaml from "js-yaml";

import { MAP_DIR } from "./constants";
import { loadMapIndex } from "./loadMapIndex";
import { MapFile } from "./types";

function parseMapFile(map: string): MapFile["content"] {
  const [frontmatter, ...rest] = map.split("---");
  const mapData = rest.join("---").trim();

  // OmegaConf output format is a bit messy, so we need to clean it up
  const updatedFrontmatter = yaml.dump(yaml.load(frontmatter));

  return {
    frontmatter: updatedFrontmatter,
    data: mapData,
  };
}

export async function getMaps(
  filters?: { key: string; value: string }[]
): Promise<MapFile[]> {
  // read all files from ./maps folder
  let mapFiles: string[];

  if (filters?.length) {
    const mapIndex = await loadMapIndex();
    let mapFilesSet: Set<string> | undefined;
    for (const filter of filters) {
      const mapFiles = mapIndex[filter.key]?.[filter.value] ?? [];
      console.log(filter, mapFiles);
      mapFilesSet = mapFilesSet
        ? mapFilesSet.intersection(new Set(mapFiles))
        : new Set(mapFiles);
    }
    mapFiles = mapFilesSet ? Array.from(mapFilesSet) : [];
  } else {
    mapFiles = (await fs.readdir(MAP_DIR)).filter((file) =>
      file.endsWith(".yaml")
    );
  }

  // load each file as json
  const maps = await Promise.all(
    mapFiles.map(async (file) => {
      const map = await fs.readFile(`${MAP_DIR}/${file}`, "utf8");
      const content = parseMapFile(map);
      return {
        file,
        content,
      };
    })
  );

  return maps;
}

export async function getMap(file: string): Promise<MapFile> {
  const maps = await getMaps();
  const map = maps.find((map) => map.file === file);
  if (!map) {
    throw new Error(`Map file ${file} not found`);
  }
  return map;
}
