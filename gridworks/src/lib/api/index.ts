import * as z from "zod";

import { API_URL } from "../../server/constants";
import { fetchApi } from "./utils";

const childrenActionSchema = z.object({
  get scene() {
    return sceneSchema;
  },
  where: z
    .union([
      z.literal("full"),
      z.object({
        tags: z.array(z.string()),
      }),
    ])
    .nullable(),
  limit: z.number().nullable(),
  offset: z.number().nullable(),
  lock: z.string().nullable(),
  order_by: z.string().nullable(),
});

const sceneSchema = z.looseObject({
  type: z.string(),
  seed: z.number().nullable(),
  children: z.array(childrenActionSchema),
});

const sceneTreeSchema = z.object({
  config: sceneSchema,
  area: z.object({
    x: z.number(),
    y: z.number(),
    width: z.number(),
    height: z.number(),
  }),
  get children() {
    return z.array(sceneTreeSchema);
  },
  render_start_time: z.number(),
  render_end_time: z.number(),
  render_with_children_end_time: z.number(),
});

export type SceneTree = z.infer<typeof sceneTreeSchema>;

export const storableMapSchema = z.object({
  frontmatter: z.object({
    metadata: z.record(z.string(), z.unknown()),
    config: z.record(z.string(), z.unknown()),
    scene_tree: sceneTreeSchema.nullable(),
    char_to_name: z.record(z.string(), z.string()),
  }),
  data: z.string(),
});

export type StorableMap = z.infer<typeof storableMapSchema>;

const configMakerSchema = z.object({
  absolute_path: z.string(),
  path: z.string(),
  kind: z.string(),
  line: z.number(),
});

export const extendedConfigSchema = z.object({
  value: z
    .record(z.string(), z.unknown())
    .or(z.array(z.record(z.string(), z.unknown()))),
  unset_fields: z.array(z.string()),
});

export type ExtendedConfig = z.infer<typeof extendedConfigSchema>;

const viewConfigSchema = z.object({
  maker: configMakerSchema,
  config: extendedConfigSchema,
});

export type MakerConfig = z.infer<typeof viewConfigSchema>;

const groupedConfigMakersSchema = z.record(
  z.string(),
  z.array(configMakerSchema).optional()
);

export type GroupedConfigMakers = z.infer<typeof groupedConfigMakersSchema>;

export async function listConfigMakers(): Promise<GroupedConfigMakers> {
  return await fetchApi(`${API_URL}/configs`, groupedConfigMakersSchema);
}

export async function getConfig(path: string): Promise<MakerConfig> {
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

export async function getRepoRoot(): Promise<string> {
  const response = await fetch(`${API_URL}/repo-root`);
  const data = await response.json();
  const parsed = z.object({ repo_root: z.string() }).parse(data);
  return parsed.repo_root;
}

export async function getMettagridEncoding(): Promise<Record<string, string>> {
  const response = await fetch(`${API_URL}/mettagrid-encoding`);
  const data = await response.json();
  const parsed = z.record(z.string(), z.string()).parse(data);
  return parsed;
}
export { getJsonSchemas } from "./schemas";
