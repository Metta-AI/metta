import yaml from "js-yaml";
import { z } from "zod";

import { MapData, MapIndex, MapMetadata } from "./types";
import { API_URL } from "./constants";

function parseMapFile(map: string): MapData["content"] {
  const [frontmatter, ...rest] = map.split("---");
  const mapData = rest.join("---").trim();

  // OmegaConf output format is a bit messy, so we need to clean it up
  const updatedFrontmatter = yaml.dump(yaml.load(frontmatter));

  return {
    frontmatter: updatedFrontmatter,
    data: mapData,
  };
}

export async function getStoredMapDirs(): Promise<string[]> {
  const response = await fetch(
    `${API_URL}/stored-maps/dirs`
  );
  const data = await response.json();
  const parsed = z.array(z.string()).parse(data.dirs);
  return parsed;
}

export async function findStoredMaps(
  dir: string,
  filters: { key: string; value: string }[] = []
): Promise<MapMetadata[]> {
  const searchParams = new URLSearchParams({
    dir,
    filter: filters.map(f => `${f.key}=${f.value}`).join(","),
  });
  const response = await fetch(
    `${API_URL}/stored-maps/find-maps?${searchParams}`
  );
  const data = await response.json();
  const parsed = z.array(z.string()).parse(data.maps);
  return parsed.map((url) => ({ url }));

}

export async function getStoredMap(url: string): Promise<MapData> {
  const response = await fetch(
    `${API_URL}/stored-maps/get-map?url=${url}`
  );
  const data = await response.json();
  return {
    content: parseMapFile(data.content),
  };
}

export async function loadStoredMapIndex(dir: string): Promise<MapIndex> {
  const response = await fetch(`${API_URL}/stored-maps/get-index?dir=${encodeURIComponent(dir)}`)
  const data = await response.json();

  const mapIndexSchema = z.record(
    z.string(),
    z.record(z.string(), z.array(z.string()))
  );
  return mapIndexSchema.parse(data);
}

export async function listEnvs(): Promise<string[]> {
  const response = await fetch(`${API_URL}/envs`);
  const data = await response.json();
  const parsed = z.array(z.string()).parse(data.envs);
  return parsed;
}

export async function getEnv(name: string): Promise<string> {
  const response = await fetch(`${API_URL}/envs/get?name=${encodeURIComponent(name)}`);
  const data = await response.json();
  const parsed = z.object({ content: z.string() }).parse(data);
  return parsed.content;
}
