import z from "zod";

import { API_URL } from "@/server/constants";

import {
  ExtendedConfig,
  extendedConfigSchema,
  StorableMap,
  storableMapSchema,
} from "./";
import { fetchApi } from "./utils";

const missionSchema = z.object({
  name: z.string(),
  description: z.string(),
  site: z.object({
    name: z.string(),
  }),
  num_cogs: z.number().nullable(),
});

const missionsSchema = z.array(extendedConfigSchema);

type Mission = z.infer<typeof missionSchema>;

export type MissionWithFullConfig = {
  mission: Mission;
  fullConfig: ExtendedConfig;
};

const variantSchema = z.object({
  name: z.string(),
  description: z.string(),
});

export type Variant = z.infer<typeof variantSchema>;

function expandMission(mission: ExtendedConfig): MissionWithFullConfig {
  return {
    mission: missionSchema.parse(mission.value), // TODO - fallback on mismatch
    fullConfig: extendedConfigSchema.parse(mission),
  };
}

export async function getMissions(): Promise<MissionWithFullConfig[]> {
  const parsed = await fetchApi(`${API_URL}/cogames/missions`, missionsSchema);
  return parsed.map(expandMission);
}

export async function getMission(
  name: string,
  variants: string[] = []
): Promise<MissionWithFullConfig> {
  const parsed = await fetchApi(
    `${API_URL}/cogames/missions/${name}?variants=${variants.join(",")}`,
    extendedConfigSchema
  );
  return expandMission(parsed);
}

export async function getMissionEnv(
  name: string,
  variants: string[] = []
): Promise<ExtendedConfig> {
  return await fetchApi(
    `${API_URL}/cogames/missions/${name}/env?variants=${variants.join(",")}`,
    extendedConfigSchema
  );
}

export async function getMissionMap(
  name: string,
  variants: string[] = []
): Promise<StorableMap> {
  return await fetchApi(
    `${API_URL}/cogames/missions/${name}/map?variants=${variants.join(",")}`,
    storableMapSchema
  );
}

export async function getVariants(): Promise<Variant[]> {
  return await fetchApi(`${API_URL}/cogames/variants`, z.array(variantSchema));
}
