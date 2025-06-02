import Link from "next/link";

import { JsonAsYaml } from "@/components/JsonAsYaml";

import { getMettagridCfgFile } from "../../../lib/api";
import { MapFromCfg } from "./MapFromCfg";

interface EnvViewPageProps {
  searchParams: { path?: string };
}

export default async function EnvViewPage({ searchParams }: EnvViewPageProps) {
  const cfgName = searchParams.path;

  if (!cfgName) {
    throw new Error("No config name provided");
  }

  const cfg = await getMettagridCfgFile(cfgName);

  return (
    <div className="p-4">
      <div className="mb-4">
        <Link
          href="/mettagrid-cfgs"
          className="text-blue-600 hover:text-blue-800 hover:underline"
        >
          ← Back to MettaGrid configs list
        </Link>
      </div>
      <h1 className="mb-4 text-2xl font-bold">
        <span>{cfg.metadata.path}</span>
        <span className="text-xl text-gray-400"> {cfg.metadata.kind}</span>
      </h1>
      {(cfg.metadata.kind === "map" || cfg.metadata.kind === "env") && (
        <section className="mb-8">
          <h2 className="mb-4 text-xl font-bold">Generated Map</h2>
          <MapFromCfg cfg={cfg} />
        </section>
      )}
      <section className="mb-8">
        <h2 className="mb-4 text-xl font-bold">Config</h2>
        <JsonAsYaml json={cfg.cfg as Record<string, unknown>} />
      </section>
    </div>
  );
}
