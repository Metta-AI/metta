import Link from "next/link";

import { JsonAsYaml } from "@/components/JsonAsYaml";
import { configsRoute } from "@/lib/routes";

import { getMettagridCfgFile } from "../../../lib/api";
import { MapSection } from "./MapSection";
import { RunToolSection } from "./RunToolSection";

interface EnvViewPageProps {
  searchParams: Promise<{ path?: string }>;
}

export default async function EnvViewPage({ searchParams }: EnvViewPageProps) {
  const { path } = await searchParams;

  if (!path) {
    throw new Error("No config name provided");
  }

  const cfg = await getMettagridCfgFile(path);

  return (
    <div className="p-4">
      <div className="mb-4">
        <Link
          href={configsRoute()}
          className="text-blue-600 hover:text-blue-800 hover:underline"
        >
          ← Back to config makers list
        </Link>
      </div>
      <h1 className="mb-4 text-2xl font-bold">
        <span>
          <a
            className="hover:underline"
            href={`cursor://file${cfg.metadata.absolute_path}:${cfg.metadata.line}`}
          >
            {cfg.metadata.path}
          </a>
        </span>
        <span className="text-xl text-gray-400"> &rarr;</span>
        <span className="text-xl text-gray-500"> {cfg.metadata.kind}</span>
      </h1>
      {cfg.metadata.kind === "MettaGridConfig" && <MapSection cfg={cfg} />}
      {cfg.metadata.kind.endsWith("Tool") && <RunToolSection cfg={cfg} />}
      <section className="mb-8">
        <h2 className="mb-4 text-xl font-bold">Config</h2>
        <JsonAsYaml json={cfg.config as Record<string, unknown>} />
      </section>
    </div>
  );
}
