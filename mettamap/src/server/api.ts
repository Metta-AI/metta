import yaml from "js-yaml";
import { z } from "zod";

import { API_URL } from "./constants";
import { MapData, MapIndex, MapMetadata } from "./types";

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

const mettagridCfgFileMetadataSchema = z.object({
  path: z.string(),
  kind: z.string(),
});

const mettagridCfgFileSchema = z.object({
  metadata: mettagridCfgFileMetadataSchema,
  cfg: z.unknown(),
});

export type MettagridCfgFile = z.infer<typeof mettagridCfgFileSchema>;

const mettagridCfgsMetadataSchema = z.object({
  env: z.array(mettagridCfgFileMetadataSchema).optional(),
  curriculum: z.array(mettagridCfgFileMetadataSchema).optional(),
  map: z.array(mettagridCfgFileMetadataSchema).optional(),
  unknown: z.array(mettagridCfgFileMetadataSchema).optional(),
});

type MettagridCfgsMetadata = z.infer<typeof mettagridCfgsMetadataSchema>;

export async function listMettagridCfgsMetadata(): Promise<MettagridCfgsMetadata> {
  const response = await fetch(`${API_URL}/mettagrid-cfgs`);
  const data = await response.json();
  const parsed = mettagridCfgsMetadataSchema.parse(data);
  console.log(parsed);
  return parsed;
}

export async function getMettagridCfgFile(path: string): Promise<MettagridCfgFile> {
  const response = await fetch(`${API_URL}/mettagrid-cfgs/get?path=${encodeURIComponent(path)}`);
  const data = await response.json();
  return mettagridCfgFileSchema.parse(data);
}

export async function getMettagridCfgMap(path: string): Promise<{ type: 'map', data: MapData } | { type: 'error', error: string }> {
  const response = await fetch(`${API_URL}/mettagrid-cfgs/get-map?path=${encodeURIComponent(path)}`);
  const data = await response.json();
  if ("error" in data) {
    return { type: "error", error: String(data.error) };
  }
  return { type: "map", data: { content: parseMapFile(data.content) } };
}
