import { z } from "zod";

import { API_URL } from "../server/constants";
import { MapIndex, MapMetadata } from "../server/types";

export type StorableMap = {
  frontmatter: {
    metadata: Record<string, unknown>;
    config: Record<string, unknown>;
  };
  data: string;
};

function parseStorableMap(map: unknown): StorableMap {
  const parsed = z
    .object({
      frontmatter: z.object({
        metadata: z.record(z.string(), z.unknown()),
        config: z.record(z.string(), z.unknown()),
      }),
      data: z.string(),
    })
    .parse(map);

  return parsed;
}

export async function getStoredMapDirs(): Promise<string[]> {
  const response = await fetch(`${API_URL}/stored-maps/dirs`);
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
    filter: filters
      .map((f) => `${f.key}=${encodeURIComponent(f.value)}`)
      .join(","),
  });
  const response = await fetch(
    `${API_URL}/stored-maps/find-maps?${searchParams}`
  );
  const data = await response.json();
  const parsed = z.array(z.string()).parse(data.maps);
  return parsed.map((url) => ({ url }));
}

export async function getStoredMap(url: string): Promise<StorableMap> {
  const response = await fetch(`${API_URL}/stored-maps/get-map?url=${url}`);
  const data = await response.json();
  return parseStorableMap(data);
}

export async function loadStoredMapIndex(dir: string): Promise<MapIndex> {
  const response = await fetch(
    `${API_URL}/stored-maps/get-index?dir=${encodeURIComponent(dir)}`
  );
  const data = await response.json();

  const mapIndexSchema = z.record(
    z.string(),
    z.record(z.string(), z.array(z.string()))
  );
  return mapIndexSchema.parse(data);
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
  const response = await fetch(`${API_URL}/mettagrid-cfgs`);
  const data = await response.json();
  const parsed = mettagridCfgsMetadataSchema.parse(data);
  return parsed;
}

export async function getMettagridCfgFile(
  path: string
): Promise<MettagridCfgFile> {
  const response = await fetch(
    `${API_URL}/mettagrid-cfgs/get?path=${encodeURIComponent(path)}`
  );
  const data = await response.json();
  return mettagridCfgFileSchema.parse(data);
}

export type MaybeStorableMap =
  | {
      type: "map";
      data: StorableMap;
    }
  | {
      type: "error";
      error: string;
    };

export async function getMettagridCfgMap(
  path: string
): Promise<MaybeStorableMap> {
  const response = await fetch(
    `${API_URL}/mettagrid-cfgs/get-map?path=${encodeURIComponent(path)}`
  );
  const data = await response.json();
  if ("error" in data) {
    return { type: "error", error: String(data.error) };
  }
  return { type: "map", data: parseStorableMap(data) };
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
