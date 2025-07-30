import Link from "next/link";

import { listMettagridCfgsMetadata } from "../../lib/api";

const KIND_DESCRIPTIONS = {
  env: "Environment configurations - define game rules, agent behavior, and environment settings",
  curriculum:
    "Training curriculums - define learning progression and task selection for training",
  map: "Map generators - define how to create and structure the game world",
  unknown: "Unknown configuration type",
} as const;

export default async function EnvsPage() {
  const cfgs = await listMettagridCfgsMetadata();

  return (
    <div className="p-4">
      <h1 className="mb-3 text-2xl font-bold">MettaGrid Configs</h1>
      <p className="mb-4 text-gray-600">
        These are configuration files that define different aspects of the
        MettaGrid environment. Click on any config to view its details and
        preview the generated map.
      </p>

      {Object.keys(cfgs).map((kind) => {
        const configs = cfgs[kind as keyof typeof cfgs];
        if (!configs || configs.length === 0) return null;

        return (
          <div key={kind} className="mb-6">
            <h2 className="mb-1 text-lg font-bold capitalize">{kind}</h2>
            <p className="mb-2 text-sm text-gray-600">
              {KIND_DESCRIPTIONS[kind as keyof typeof KIND_DESCRIPTIONS]}
            </p>
            <ul className="space-y-1">
              {configs.map((cfg) => {
                return (
                  <li key={cfg.path}>
                    <Link
                      href={`/mettagrid-cfgs/view?path=${cfg.path}`}
                      className="block py-0.5 text-blue-600 hover:text-blue-800 hover:underline"
                    >
                      {cfg.path}
                    </Link>
                  </li>
                );
              })}
            </ul>
          </div>
        );
      })}
    </div>
  );
}
