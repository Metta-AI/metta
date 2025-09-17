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

const configMakerSchema = z.object({
  absolute_path: z.string(),
  path: z.string(),
  kind: z.string(),
  line: z.number(),
});

const viewConfigSchema = z.object({
  maker: configMakerSchema,
  config: z.object({
    value: z
      .record(z.string(), z.unknown())
      .or(z.array(z.record(z.string(), z.unknown()))),
    unset_fields: z.array(z.string()),
  }),
});

export type Config = z.infer<typeof viewConfigSchema>;

const groupedConfigMakersSchema = z.record(
  z.string(),
  z.array(configMakerSchema).optional()
);

type GroupedConfigMakers = z.infer<typeof groupedConfigMakersSchema>;

export async function listConfigMakers(): Promise<GroupedConfigMakers> {
  return await fetchApi(`${API_URL}/configs`, groupedConfigMakersSchema);
}

export async function getConfig(path: string): Promise<Config> {
  return await fetchApi(
    `${API_URL}/configs/get?path=${encodeURIComponent(path)}`,
    viewConfigSchema
  );
}

export async function getConfigMap(
  path: string,
  name?: string
): Promise<StorableMap> {
  const queryParams = new URLSearchParams();
  queryParams.set("path", path);
  if (name) {
    queryParams.set("name", name);
  }

  return await fetchApi(
    `${API_URL}/configs/get-map?${queryParams.toString()}`,
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
