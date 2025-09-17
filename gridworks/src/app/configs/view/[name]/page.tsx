import Link from "next/link";

import { ConfigViewer } from "@/components/ConfigViewer";
import { configsRoute } from "@/lib/routes";

import { getConfig } from "../../../../lib/api";
import { MapSection } from "./MapSection";
import { RunToolSection } from "./RunToolSection";

interface ConfigViewPage {
  params: Promise<{ name: string }>;
}

export default async function ConfigViewPage({ params }: ConfigViewPage) {
  const { name } = await params;

  if (!name) {
    throw new Error("No config name provided");
  }

  const cfg = await getConfig(name);

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
            href={`cursor://file${cfg.maker.absolute_path}:${cfg.maker.line}`}
          >
            {cfg.maker.path}
          </a>
        </span>
        <span className="text-xl text-gray-400"> &rarr;</span>
        <span className="text-xl text-gray-500"> {cfg.maker.kind}</span>
      </h1>
      {cfg.maker.kind === "MettaGridConfig" && <MapSection cfg={cfg} />}
      {cfg.maker.kind.endsWith("Tool") && <RunToolSection cfg={cfg} />}
      <section className="mb-8">
        <h2 className="mb-4 text-xl font-bold">Config</h2>
        <ConfigViewer value={cfg.config.value} unsetFields={cfg.config.unset_fields} />
      </section>
    </div>
  );
}
