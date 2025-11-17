import { SearchParams } from "nuqs/server";

import { ConfigViewer } from "@/components/ConfigViewer";
import { getMission } from "@/lib/api/cogames";

import { loadMissionSearchParams } from "./searchParams";

export default async function ViewMissionPage({
  params,
  searchParams,
}: {
  params: Promise<{ name: string }>;
  searchParams: Promise<SearchParams>;
}) {
  const { name } = await params;
  const { variants } = await loadMissionSearchParams(searchParams);
  const mission = await getMission(name, variants ?? []);

  return (
    <div className="pt-4">
      <ConfigViewer
        value={mission.fullConfig.value}
        unsetFields={mission.fullConfig.unset_fields}
        kind="Mission"
      />
    </div>
  );
}
