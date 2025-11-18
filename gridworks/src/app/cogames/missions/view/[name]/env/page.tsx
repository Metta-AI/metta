import { SearchParams } from "nuqs/server";

import { ConfigViewer } from "@/components/ConfigViewer";
import { getMissionEnv } from "@/lib/api/cogames";

import { loadMissionSearchParams } from "../searchParams";

export default async function MissionEnvPage({
  params,
  searchParams,
}: {
  params: Promise<{ name: string }>;
  searchParams: Promise<SearchParams>;
}) {
  const { name } = await params;
  const { variants } = await loadMissionSearchParams(searchParams);
  const env = await getMissionEnv(name, variants ?? []);

  return (
    <ConfigViewer
      value={env.value}
      unsetFields={env.unset_fields}
      kind="MettaGridConfig"
    />
  );
}
