import * as z from "zod/v4";

import { API_URL } from "../server/constants";
import { MapIndex, MapMetadata } from "../server/types";

const sceneTreeSchema = z.object({
  type: z.string(),
  params: z.record(z.string(), z.unknown()),
  area: z.object({
    x: z.number(),
    y: z.number(),
    width: z.number(),
    height: z.number(),
  }),
  get children() {
    return z.array(sceneTreeSchema);
  },
});

export type SceneTree = z.infer<typeof sceneTreeSchema>;

const storableMapSchema = z.object({
  frontmatter: z.object({
    metadata: z.record(z.string(), z.unknown()),
    config: z.record(z.string(), z.unknown()),
    scene_tree: sceneTreeSchema.nullable(),
  }),
  data: z.string(),
});

export type StorableMap = z.infer<typeof storableMapSchema>;

export async function getStoredMapDirs(): Promise<string[]> {
  const response = await fetch(`${API_URL}/stored-maps/dirs`);
  const data = await response.json();
  const parsed = z.array(z.string()).parse(data.dirs);
  return parsed;
}

async function fetchApi<T extends z.ZodTypeAny>(
  url: string,
  schema: T
): Promise<z.infer<T>> {
  const response = await fetch(url);
  if (response.status === 500) {
    const data = await response.json();
    const detail = String(data.detail) || "Unknown error";
    throw new Error(detail);
  }
  const data = await response.json();
  return schema.parse(data);
}

export async function findStoredMaps(
  dir: string,
  filters: { key: string; value: string }[] = []
): Promise<MapMetadata[]> {
  const searchParams = new URLSearchParams({
    dir,
    filter: filters
      .map((f) => `${f.key}=${encodeURIComponent(f.value)}`)
      .join(","),
  });
  const data = await fetchApi(
    `${API_URL}/stored-maps/find-maps?${searchParams}`,
    z.object({ maps: z.array(z.string()) })
  );
  return data.maps.map((url) => ({ url }));
}

export async function getStoredMap(url: string): Promise<StorableMap> {
  return fetchApi(
    `${API_URL}/stored-maps/get-map?url=${url}`,
    storableMapSchema
  );
}

export async function loadStoredMapIndex(dir: string): Promise<MapIndex> {
  const mapIndexSchema = z.record(
    z.string(),
    z.record(z.string(), z.array(z.string()))
  );
  return await fetchApi(
    `${API_URL}/stored-maps/get-index?dir=${encodeURIComponent(dir)}`,
    mapIndexSchema
  );
}

const mettagridCfgFileMetadataSchema = z.object({
  absolute_path: z.string(),
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
  return await fetchApi(
    `${API_URL}/mettagrid-cfgs`,
    mettagridCfgsMetadataSchema
  );
}

export async function getMettagridCfgFile(
  path: string
): Promise<MettagridCfgFile> {
  return await fetchApi(
    `${API_URL}/mettagrid-cfgs/get?path=${encodeURIComponent(path)}`,
    mettagridCfgFileSchema
  );
}

export async function getMettagridCfgMap(path: string): Promise<StorableMap> {
  return await fetchApi(
    `${API_URL}/mettagrid-cfgs/get-map?path=${encodeURIComponent(path)}`,
    storableMapSchema
  );
}

export async function indexDir(dir: string): Promise<void> {
  const response = await fetch(
    `${API_URL}/stored-maps/index-dir?dir=${encodeURIComponent(dir)}`,
    { method: "POST" }
  );
  const data = await response.json();
  return data;
}

export async function getRepoRoot(): Promise<string> {
  const response = await fetch(`${API_URL}/repo-root`);
  const data = await response.json();
  const parsed = z.object({ repo_root: z.string() }).parse(data);
  return parsed.repo_root;
}
